"""netCDF -> hive-partitioned Parquet ingestion (memory-bounded, idempotent, validated).

Output layout (``paths.parquet``)::

    process/day=2024-08-05/lot=7/wafer=3/step=etch/part-0.parquet   # 5 Hz tool traces, float32
    oes/day=2024-08-05/lot=7/wafer=3/step=etch/part-0.parquet       # 25 Hz spectra, 3648 cols
    meta/oes_dictionary.parquet   meta/process_dictionary.parquet   meta/wavelengths.parquet
    _manifest/<source>__<group>.json                                  # one record per wafer

* ``step`` is the BOSCH phase (off / ign / etch / dep) from :mod:`plasma_etch_vm.segmentation`.
* OES spectra are stored as the lossless uint16 dictionary codes (``store_oes_as: codes``, decode
  with ``meta/oes_dictionary.parquet``) or as decoded float32 (``store_oes_as: float32``, 2x size).
* Every wafer is written to a temporary directory and moved into place only after schema validation
  succeeds; a manifest record (source size + mtime + config hash) makes re-runs skip finished wafers.
* Peak memory ≈ ``chunk_rows × 3648 × (2 + 4)`` bytes per open writer block, independent of the
  number of days, so the same command scales from 1 to 10 day files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import netcdf_io as nio
from .lines import LINE_BY_NAME, apply_calibration, calibrate_wavelengths, line_intensities, line_windows
from .metrology import DATE_TO_LOT, day_from_filename, parse_group_name
from .segmentation import GasMap, assign_cycles, label_from_indicator, label_process

log = logging.getLogger(__name__)

N_OES_CHANNELS = 3648


# ----------------------------------------------------------------------------------------- utils
def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def source_signature(path: Path) -> dict:
    st = path.stat()
    return {"file": path.name, "size": st.st_size, "mtime": int(st.st_mtime)}


def oes_channel_names(n: int = N_OES_CHANNELS) -> list[str]:
    return [f"ch{i:04d}" for i in range(n)]


@dataclass
class IngestStats:
    kind: str
    source: str
    group: str
    rows: int = 0
    bytes_in: int = 0  # uncompressed bytes of the encoded arrays read
    bytes_out: int = 0  # bytes of parquet written
    seconds: float = 0.0
    steps: dict = field(default_factory=dict)
    skipped: bool = False

    @property
    def rows_per_s(self) -> float:
        return self.rows / self.seconds if self.seconds else float("nan")

    @property
    def mb_per_s(self) -> float:
        return self.bytes_in / 1e6 / self.seconds if self.seconds else float("nan")

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d.update(rows_per_s=self.rows_per_s, mb_per_s_in=self.mb_per_s)
        return d


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


class Manifest:
    """One JSON record per (source, group). Used for idempotency and throughput reporting."""

    def __init__(self, root: Path):
        self.dir = root / "_manifest"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, source: str, group: str) -> Path:
        return self.dir / f"{kind}__{Path(source).stem}__{group}.json"

    def done(self, kind: str, src: Path, group: str, cfg_hash: str) -> bool:
        p = self._path(kind, src.name, group)
        if not p.exists():
            return False
        rec = json.loads(p.read_text())
        return (
            rec.get("source") == source_signature(src)
            and rec.get("config_hash") == cfg_hash
            and all(Path(o).exists() for o in rec.get("outputs", []))
        )

    def write(self, kind: str, src: Path, group: str, cfg_hash: str, stats: IngestStats, outputs: list[Path]):
        rec = {
            "source": source_signature(src),
            "config_hash": cfg_hash,
            "outputs": [str(o) for o in outputs],
            "stats": stats.as_dict(),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._path(kind, src.name, group).write_text(json.dumps(rec, indent=2, default=str))

    def records(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted(self.dir.glob("*.json"))]


# ------------------------------------------------------------------------------ schema validation
class SchemaError(ValueError):
    pass


def validate_times(t: np.ndarray, name: str, max_gap_s: float = 30.0) -> dict:
    if not np.all(np.isfinite(t)):
        raise SchemaError(f"{name}: non-finite timestamps")
    d = np.diff(t)
    if np.any(d <= 0):
        raise SchemaError(f"{name}: timestamps not strictly increasing ({int((d <= 0).sum())} violations)")
    return {
        "dt_median_s": float(np.median(d)) if d.size else float("nan"),
        "n_gaps": int((d > 3 * np.median(d)).sum()) if d.size else 0,
        "max_gap_s": float(d.max()) if d.size else 0.0,
        "gap_warning": bool(d.size and d.max() > max_gap_s),
    }


def validate_written(part_dir: Path, expected_cols: set[str], expected_rows: int) -> None:
    rows = 0
    for f in part_dir.rglob("*.parquet"):
        md = pq.ParquetFile(f)
        cols = set(md.schema_arrow.names)
        missing = expected_cols - cols
        if missing:
            raise SchemaError(f"{f}: missing columns {sorted(missing)[:5]}...")
        rows += md.metadata.num_rows
    if rows != expected_rows:
        raise SchemaError(f"{part_dir}: wrote {rows} rows, expected {expected_rows}")


# ----------------------------------------------------------------------------------- writers
class StepWriters:
    """Lazily opened ParquetWriter per step partition inside a staging directory.

    ``writer_kwargs`` are passed to :class:`pyarrow.parquet.ParquetWriter` (encodings, dictionary
    use); ``column_encoding`` entries for columns absent from a table are dropped automatically.
    """

    def __init__(self, stage: Path, compression: str, level: int | None, **writer_kwargs):
        self.stage = stage
        self.compression = compression
        self.level = level
        self.writer_kwargs = writer_kwargs
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.rows: dict[str, int] = {}

    def write(self, step: str, table: pa.Table) -> None:
        if step not in self.writers:
            d = self.stage / f"step={step}"
            d.mkdir(parents=True, exist_ok=True)
            kw = dict(self.writer_kwargs)
            if isinstance(kw.get("column_encoding"), dict):
                present = set(table.column_names)
                kw["column_encoding"] = {k: v for k, v in kw["column_encoding"].items() if k in present}
            self.writers[step] = pq.ParquetWriter(
                d / "part-0.parquet", table.schema, compression=self.compression, compression_level=self.level, **kw
            )
            self.rows[step] = 0
        self.writers[step].write_table(table)
        self.rows[step] += table.num_rows

    def close(self) -> None:
        for w in self.writers.values():
            w.close()


class StepBuffer:
    """Accumulate row blocks per step and flush them as large row groups (better compression,
    far less per-column metadata than one row group per HDF5 block). Bounded by ``max_rows``."""

    def __init__(self, writers: StepWriters, max_rows: int):
        self.w = writers
        self.max_rows = max_rows
        self.parts: dict[str, list[dict[str, np.ndarray]]] = {}
        self.n: dict[str, int] = {}

    def add(self, step: str, cols: dict[str, np.ndarray]) -> None:
        self.parts.setdefault(step, []).append(cols)
        self.n[step] = self.n.get(step, 0) + len(next(iter(cols.values())))
        if self.n[step] >= self.max_rows:
            self.flush(step)

    def flush(self, step: str | None = None) -> None:
        for s in [step] if step else list(self.parts):
            parts = self.parts.pop(s, [])
            self.n[s] = 0
            if not parts:
                continue
            meta = {k: np.concatenate([p[k] for p in parts]) for k in ("t_s", "sample", "cycle")}
            spec = np.concatenate([p["spectrum"] for p in parts])
            cols = {k: pa.array(v) for k, v in meta.items()}
            for j in range(spec.shape[1]):
                cols[f"ch{j:04d}"] = pa.array(np.ascontiguousarray(spec[:, j]))
            self.w.write(s, pa.table(cols))


def _commit(stage: Path, final: Path) -> None:
    """Atomically replace ``final`` with ``stage`` (rename within the same volume)."""
    if final.exists():
        shutil.rmtree(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, final)


# ------------------------------------------------------------------------------------ process
def common_process_columns(process_nc: Path) -> list[str]:
    """Intersection of the parameter names across all wafer groups (31 of 44 on 2024-07-02)."""
    cols: set[str] | None = None
    for g in nio.list_groups(process_nc):
        names = set(nio.group_info(process_nc, g, "process").columns)
        cols = names if cols is None else cols & names
    return sorted(cols or [])


def ingest_process(cfg: dict, out_root: Path, manifest: Manifest, force: bool = False) -> list[IngestStats]:
    raw = Path(cfg["paths"]["raw"])
    src = raw / cfg["files"]["process"]
    decoder = nio.load_decoder(raw / cfg["files"]["process_dictionary"])
    gas = GasMap(**cfg.get("gas_map", {}))
    common = common_process_columns(src)
    chash = config_hash({"ingest": cfg["ingest"], "gas": gas.__dict__, "cols": common})
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"code": np.arange(decoder.size, dtype=np.int32), "value": decoder}),
        out_root / "meta" / "process_dictionary.parquet",
    )
    stats = []
    for group in nio.list_groups(src):
        day, wafer = parse_group_name(group)
        lot = DATE_TO_LOT[day]
        st = IngestStats("process", src.name, group)
        if not force and manifest.done("process", src, group, chash):
            st.skipped = True
            stats.append(st)
            continue
        t0 = time.perf_counter()
        t, vals, cols = nio.read_group_decoded(src, group, decoder, "process")
        st.bytes_in = vals.size * 2
        timing = validate_times(t, group)
        df = pd.DataFrame(vals, columns=cols)[common]
        df.insert(0, "t_s", t)
        df = label_process(df, gas)
        df["sample"] = np.arange(len(df), dtype=np.int32)
        final = out_root / "process" / f"day={day}" / f"lot={lot}" / f"wafer={wafer}"
        stage = final.parent / f".stage_wafer={wafer}"
        if stage.exists():
            shutil.rmtree(stage)
        w = StepWriters(stage, cfg["ingest"]["compression"], cfg["ingest"].get("compression_level"))
        for step, part in df.groupby("phase", sort=False):
            w.write(str(step), pa.Table.from_pandas(part.drop(columns="phase"), preserve_index=False))
        w.close()
        validate_written(stage, {"t_s", "cycle", "sample", *common}, len(df))
        _commit(stage, final)
        st.rows = len(df)
        st.steps = {**w.rows, **timing}
        st.seconds = time.perf_counter() - t0
        st.bytes_out = _dir_bytes(final)
        manifest.write("process", src, group, chash, st, [final])
        stats.append(st)
    return stats


# ---------------------------------------------------------------------------------------- OES
TRACE_LINES = ("F_685", "F_703", "F_690", "C2_516", "C2_563")


DARK_BAND_NM = (186.0, 250.0)  # no detectable emission (etch/dep within 0.6 count of dark)


def dark_channels(wl_cal: np.ndarray, band: tuple[float, float] = DARK_BAND_NM) -> np.ndarray:
    return np.flatnonzero((wl_cal >= band[0]) & (wl_cal <= band[1]))


def baseline_corrected_total(spec: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """Σ_channels (I − b) with b = per-spectrum median of the UV dark band (removes the common-mode
    baseline jitter of ±20 counts/channel that dominates the raw channel sum)."""
    base = np.median(spec[:, dark], axis=1) if dark.size else np.zeros(len(spec), dtype=np.float32)
    return (spec.sum(axis=1, dtype=np.float64) - spec.shape[1] * base).astype(np.float32)


def check_codes(codes: np.ndarray, decoder: np.ndarray, where: str) -> None:
    """Schema check: every stored code must index the dictionary."""
    if codes.size and int(codes.max()) >= decoder.size:
        raise SchemaError(f"{where}: code {int(codes.max())} outside dictionary (size {decoder.size})")


def _decode_columns(codes: np.ndarray, decoder: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Decode only the channels ``idx`` (others left 0) — cheap per-row line extraction."""
    sub = np.zeros(codes.shape, dtype=np.float32)
    sub[:, idx] = decoder[codes[:, idx]]
    return sub


STORE_MODES = ("codes", "counts_i16", "float32")


def encode_spectra(codes: np.ndarray, decoder: np.ndarray, store: str) -> np.ndarray:
    """Spectra as stored in Parquet.

    * ``codes``      lossless uint16 dictionary codes (decode with meta/oes_dictionary.parquet)
    * ``counts_i16`` decoded intensities rounded to integer counts (|error| ≤ 0.5 count, RMS 0.29,
                     vs. per-sample noise ≳ 14 counts); ~30 % smaller than ``codes`` with delta encoding
    * ``float32``    decoded intensities (≈2x ``codes``)
    """
    if store == "codes":
        return codes
    vals = decoder[codes]
    if store == "float32":
        return vals
    if store == "counts_i16":
        if np.abs(vals).max() >= 32767:
            raise SchemaError("decoded intensity outside int16 range; use store_oes_as=codes")
        return np.rint(vals).astype(np.int16)
    raise ValueError(f"store_oes_as must be one of {STORE_MODES}")


def oes_writer_kwargs(store: str, names: list[str]) -> dict:
    if store == "float32":
        return {"use_dictionary": False, "column_encoding": dict.fromkeys(names, "BYTE_STREAM_SPLIT")}
    if store == "counts_i16":
        return {"use_dictionary": False, "column_encoding": dict.fromkeys(names, "DELTA_BINARY_PACKED")}
    return {"use_dictionary": False, "column_encoding": dict.fromkeys(names, "BYTE_STREAM_SPLIT")}


def calibrate_from_first_block(
    src: Path, group: str, decoder: np.ndarray, wavelengths: np.ndarray, chunk_rows, min_matched: int = 5
) -> dict:
    """Wavelength calibration of one wafer from the plasma-on rows of its first HDF5 block."""
    _, _, codes = next(nio.iter_blocks(src, group, chunk_rows))
    check_codes(codes, decoder, f"{src.name}/{group}")
    spec = decoder[codes]
    tot = spec.sum(axis=1)
    lo, hi = np.percentile(tot, [1, 99])
    on = tot > lo + 0.5 * (hi - lo)
    cal = calibrate_wavelengths(spec[on].mean(axis=0) if on.any() else spec.mean(axis=0), wavelengths)
    cal["ok"] = bool(cal["n_matched"] >= min_matched and cal["rms_nm"] < 0.1)
    return cal


def oes_pass1(
    src: Path, group: str, decoder: np.ndarray, wl_cal: np.ndarray, cfg: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Pass 1 over a wafer: times, phase and cycle labels at the OES rate (~24 Hz) + a light
    per-sample trace (key line intensities and total emission). Memory: one block at a time."""
    ocfg = cfg["ingest"]["oes_phase"]
    names = tuple(dict.fromkeys((ocfg["etch_line"], ocfg["dep_line"], *TRACE_LINES)))
    wins = line_windows(wl_cal, tuple(LINE_BY_NAME[n] for n in names))
    idx = np.unique(np.concatenate([np.r_[w.peak, w.bg] for w in wins.values()]))
    dark = dark_channels(wl_cal)
    ts, tots, li_parts = [], [], []
    for _, t, codes in nio.iter_blocks(src, group, cfg["ingest"]["chunk_rows"]):
        check_codes(codes, decoder, f"{src.name}/{group}")
        li_parts.append(line_intensities(_decode_columns(codes, decoder, idx), wins))
        tots.append(baseline_corrected_total(decoder[codes], dark))
        ts.append(t)
    t = np.concatenate(ts)
    tot = np.concatenate(tots)
    li = {n: np.concatenate([p[n] for p in li_parts]) for n in wins}
    # plasma on/off from background-subtracted line signal (robust to the common-mode baseline
    # jitter of the detector, which makes the raw channel sum useless for this)
    sig = np.clip(li[ocfg["etch_line"]], 0, None) + np.clip(li[ocfg["dep_line"]], 0, None)
    lo, hi = np.percentile(sig, [1, 99])
    on = sig > lo + ocfg.get("plasma_on_frac", 0.2) * (hi - lo)
    ratio = (np.clip(li[ocfg["etch_line"]], 0, None) + 1.0) / (np.clip(li[ocfg["dep_line"]], 0, None) + 1.0)
    phase = label_from_indicator(ratio, on, min_len=ocfg.get("min_len", 3)).astype(object)
    first = np.flatnonzero(phase != "off")
    if first.size:  # the 1 s ignition step = first ``ign_s`` seconds of plasma
        k = first[0]
        phase[(t >= t[k]) & (t < t[k] + ocfg.get("ign_s", 1.0)) & (phase != "off")] = "ign"
    phase = phase.astype(str)
    cycle = assign_cycles(phase)
    trace = pd.DataFrame(
        {
            "t_s": t,
            "sample": np.arange(t.size, dtype=np.int32),
            "phase": phase,
            "cycle": cycle,
            "total": tot,
            **{f"L_{n}": v for n, v in li.items()},
        }
    )
    return t, phase, cycle, trace


def ingest_oes_file(cfg: dict, src: Path, out_root: Path, manifest: Manifest, force: bool = False) -> list[IngestStats]:
    raw = Path(cfg["paths"]["raw"])
    decoder = nio.load_decoder(raw / cfg["files"]["oes_dictionary"])
    day = day_from_filename(src)
    lot = DATE_TO_LOT[day]
    store = cfg["ingest"]["store_oes_as"]
    chash = config_hash({"ingest": cfg["ingest"]})
    meta = out_root / "meta"
    (meta / "wavelength_calibration").mkdir(parents=True, exist_ok=True)
    if not (meta / "oes_dictionary.parquet").exists():
        pq.write_table(
            pa.table({"code": np.arange(decoder.size, dtype=np.int32), "value": decoder}),
            meta / "oes_dictionary.parquet",
        )
    names = oes_channel_names()
    (meta / "oes_storage.json").write_text(json.dumps({"store_oes_as": store}))
    stats = []
    for group in nio.list_groups(src):
        wafer = int(group.split("_")[-1])
        st = IngestStats("oes", src.name, group)
        if not force and manifest.done("oes", src, group, chash):
            st.skipped = True
            stats.append(st)
            continue
        t0 = time.perf_counter()
        info = nio.group_info(src, group, "oes")
        if info.n_cols != N_OES_CHANNELS:
            raise SchemaError(f"{src.name}/{group}: {info.n_cols} channels, expected {N_OES_CHANNELS}")
        wl = nio.read_axis(src, group, "wavelengths").astype(np.float64)
        if not (180 < wl[0] < 190 and 880 < wl[-1] < 890 and np.all(np.diff(wl) > 0)):
            raise SchemaError(f"{group}: unexpected wavelength axis {wl[0]:.2f}..{wl[-1]:.2f}")
        _write_wavelengths(meta, day, wafer, wl)
        cal = calibrate_from_first_block(src, group, decoder, wl, cfg["ingest"]["chunk_rows"])
        if not cal["ok"]:
            log.warning("%s/%s: wavelength calibration weak (%s); using default", src.name, group, cal)
            cal.update(cfg["ingest"]["default_wavelength_calibration"])
        (meta / "wavelength_calibration" / f"{day}_{wafer:02d}.json").write_text(json.dumps(cal, indent=2))
        wl_cal = apply_calibration(wl, cal)
        t_all, phase, cycle, trace = oes_pass1(src, group, decoder, wl_cal, cfg)
        timing = validate_times(t_all, f"{src.name}/{group}")
        final = out_root / "oes" / f"day={day}" / f"lot={lot}" / f"wafer={wafer}"
        stage = final.parent / f".stage_wafer={wafer}"
        if stage.exists():
            shutil.rmtree(stage)
        w = StepWriters(
            stage,
            cfg["ingest"]["compression"],
            cfg["ingest"].get("compression_level"),
            **oes_writer_kwargs(store, names),
        )
        buf = StepBuffer(w, cfg["ingest"].get("row_group_rows", 20000))
        for start, t, codes in nio.iter_blocks(src, group, cfg["ingest"]["chunk_rows"]):
            check_codes(codes, decoder, f"{src.name}/{group}")
            st.bytes_in += codes.nbytes
            stop = start + len(t)
            ph = phase[start:stop]
            arr = encode_spectra(codes, decoder, store)
            samples = np.arange(start, stop, dtype=np.int32)
            for step in np.unique(ph):
                m = ph == step
                buf.add(
                    str(step), {"t_s": t[m], "sample": samples[m], "cycle": cycle[start:stop][m], "spectrum": arr[m]}
                )
        buf.flush()
        w.close()
        validate_written(stage, {"t_s", "sample", "cycle", names[0], names[-1]}, info.n_time)
        _commit(stage, final)
        tdir = out_root / "oes_trace" / f"day={day}" / f"lot={lot}" / f"wafer={wafer}"
        tdir.mkdir(parents=True, exist_ok=True)
        trace.to_parquet(tdir / "part-0.parquet", index=False)
        st.rows = info.n_time
        st.steps = {**w.rows, **timing, "wl_shift_nm": cal["shift_nm"], "wl_cal_rms_nm": cal["rms_nm"]}
        st.seconds = time.perf_counter() - t0
        st.bytes_out = _dir_bytes(final)
        manifest.write("oes", src, group, chash, st, [final, tdir])
        log.info(
            "OES %s/%s: %d rows in %.1fs (%.0f rows/s, %.1f MB/s in, %.1f MB out, Δλ=%+.3f nm)",
            src.name,
            group,
            st.rows,
            st.seconds,
            st.rows_per_s,
            st.mb_per_s,
            st.bytes_out / 1e6,
            cal["shift_nm"],
        )
        stats.append(st)
    return stats


def _write_wavelengths(meta: Path, day: str, wafer: int, wl: np.ndarray) -> None:
    """Keep one canonical (stored, uncalibrated) wavelength axis; record any wafer whose axis differs."""
    canon = meta / "wavelengths.parquet"
    tab = pa.table({"channel": np.arange(wl.size, dtype=np.int32), "wavelength_nm": wl})
    if not canon.exists():
        pq.write_table(tab, canon)
        return
    ref = pq.read_table(canon)["wavelength_nm"].to_numpy()
    if ref.shape != wl.shape or np.max(np.abs(ref - wl)) > 1e-3:
        pq.write_table(tab, meta / f"wavelengths_{day}_{wafer:02d}.parquet")
        log.warning("wavelength axis of %s wafer %d differs from canonical axis", day, wafer)


def run_ingest(cfg: dict, force: bool = False, days: list[str] | None = None) -> pd.DataFrame:
    """Ingest process traces (all wafers) + every OES day file present in ``paths.raw``."""
    out_root = Path(cfg["paths"]["parquet"])
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(out_root)
    all_stats = ingest_process(cfg, out_root, manifest, force)
    day_files = sorted(Path(cfg["paths"]["raw"]).glob("Day_*.nc"))
    if days:
        day_files = [p for p in day_files if day_from_filename(p) in days]
    for f in day_files:
        all_stats += ingest_oes_file(cfg, f, out_root, manifest, force)
    return pd.DataFrame([s.as_dict() for s in all_stats])
