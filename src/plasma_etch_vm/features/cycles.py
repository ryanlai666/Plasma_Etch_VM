"""Per-cycle aggregation of the partitioned Parquet store.

For every wafer and every BOSCH cycle ``k`` and phase (etch / dep) we compute:

* tool traces (5 Hz): mean and std of each common process parameter,
* OES (25 Hz): mean spectrum (decoded), background-subtracted line intensities computed on that
  mean spectrum, total emission, and a binned spectrum (``bin_width`` channels per bin) used by the
  PCA/NMF features and the neural models.

The OES pass streams Parquet record batches and accumulates per-cycle sums, so memory stays
O(n_cycles × n_channels) regardless of run length.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..lines import LINES, apply_calibration, line_intensities, line_windows

log = logging.getLogger(__name__)

PHASES = ("etch", "dep")


def wafer_partitions(root: Path, kind: str) -> list[tuple[str, int, int, Path]]:
    """List ``(day, lot, wafer, dir)`` partitions of ``kind`` (oes | process) in the store."""
    out = []
    for d in sorted((root / kind).glob("day=*/lot=*/wafer=*")):
        if d.name.startswith(".stage"):
            continue
        day = d.parent.parent.name.split("=")[1]
        lot = int(d.parent.name.split("=")[1])
        wafer = int(d.name.split("=")[1])
        out.append((day, lot, wafer, d))
    return out


def load_oes_meta(root: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """``(decoder, stored wavelengths, storage mode)`` from the store's meta directory."""
    dec = pq.read_table(root / "meta" / "oes_dictionary.parquet")["value"].to_numpy().astype(np.float32)
    wl = pq.read_table(root / "meta" / "wavelengths.parquet")["wavelength_nm"].to_numpy()
    sfile = root / "meta" / "oes_storage.json"
    store = json.loads(sfile.read_text())["store_oes_as"] if sfile.exists() else "codes"
    return dec, wl, store


def load_wavelength_calibration(root: Path, day: str, wafer: int) -> dict | None:
    f = root / "meta" / "wavelength_calibration" / f"{day}_{wafer:02d}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _spectrum_columns(schema_names: list[str]) -> list[str]:
    return [c for c in schema_names if c.startswith("ch")]


def bin_spectrum(spec: np.ndarray, width: int) -> np.ndarray:
    """Average ``width`` adjacent channels (last partial bin dropped). Works on (..., n_channels)."""
    n = spec.shape[-1] // width * width
    return spec[..., :n].reshape(*spec.shape[:-1], n // width, width).mean(axis=-1)


def oes_cycle_table(
    wafer_dir: Path,
    decoder: np.ndarray,
    wavelengths: np.ndarray,
    bin_width: int = 8,
    batch_rows: int = 2000,
    decoded: bool = False,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Aggregate one wafer's OES partitions to a per-(phase, cycle) table.

    Returns ``(table, wafer_mean_spectra)`` where ``wafer_mean_spectra[phase]`` is the full-resolution
    mean spectrum over all cycles of that phase (for the dashboard line viewer and full-res models).
    """
    wins = line_windows(wavelengths, LINES)
    dark = np.flatnonzero((wavelengths >= 186.0) & (wavelengths <= 250.0))
    rows, wafer_spec = [], {}
    for phase in PHASES:
        f = wafer_dir / f"step={phase}" / "part-0.parquet"
        if not f.exists():
            continue
        pf = pq.ParquetFile(f)
        ch_cols = _spectrum_columns(pf.schema_arrow.names)
        sums: dict[int, np.ndarray] = {}
        sq_tot: dict[int, float] = {}
        counts: dict[int, int] = {}
        tmin: dict[int, float] = {}
        tmax: dict[int, float] = {}
        for batch in pf.iter_batches(batch_size=batch_rows, columns=["t_s", "cycle", *ch_cols]):
            cyc = batch.column("cycle").to_numpy()
            t = batch.column("t_s").to_numpy()
            codes = np.column_stack([batch.column(c).to_numpy() for c in ch_cols])
            spec = codes.astype(np.float32) if decoded else decoder[codes]
            base = np.median(spec[:, dark], axis=1) if dark.size else np.zeros(len(spec), np.float32)
            spec = spec - base[:, None]  # per-spectrum baseline (UV dark band)
            tot = spec.sum(axis=1)
            for k in np.unique(cyc):
                m = cyc == k
                k = int(k)
                sums[k] = sums.get(k, 0) + spec[m].sum(axis=0, dtype=np.float64)
                sq_tot[k] = sq_tot.get(k, 0.0) + float((tot[m].astype(np.float64) ** 2).sum())
                counts[k] = counts.get(k, 0) + int(m.sum())
                tmin[k] = min(tmin.get(k, np.inf), float(t[m].min()))
                tmax[k] = max(tmax.get(k, -np.inf), float(t[m].max()))
        if not counts:
            continue
        ks = sorted(counts)
        mean_spec = np.stack([sums[k] / counts[k] for k in ks]).astype(np.float32)
        li = line_intensities(mean_spec, wins)
        binned = bin_spectrum(mean_spec, bin_width)
        tot_mean = mean_spec.sum(axis=1)
        tot_std = np.sqrt(
            np.maximum(np.array([sq_tot[k] / counts[k] for k in ks]) - tot_mean.astype(np.float64) ** 2, 0)
        )
        tab = pd.DataFrame(
            {
                "phase": phase,
                "cycle": ks,
                "n": [counts[k] for k in ks],
                "t_start": [tmin[k] for k in ks],
                "t_end": [tmax[k] for k in ks],
                "total": tot_mean,
                "total_std": tot_std,
            }
        )
        for name, v in li.items():
            tab[f"L_{name}"] = v
        btab = pd.DataFrame(binned, columns=[f"b{j:03d}" for j in range(binned.shape[1])])
        rows.append(pd.concat([tab, btab], axis=1))
        w = np.array([counts[k] for k in ks], dtype=np.float64)
        wafer_spec[phase] = (mean_spec * w[:, None]).sum(axis=0) / w.sum()
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return table, wafer_spec


def process_cycle_table(wafer_dir: Path, params: list[str] | None = None) -> pd.DataFrame:
    """Per-(phase, cycle) mean / std of every process parameter for one wafer."""
    parts = []
    for phase in PHASES:
        f = wafer_dir / f"step={phase}" / "part-0.parquet"
        if not f.exists():
            continue
        df = pq.read_table(f).to_pandas()
        cols = params or [c for c in df.columns if c not in ("t_s", "cycle", "sample")]
        g = df.groupby("cycle")
        agg = g[cols].mean().add_prefix("P_")
        sd = g[cols].std().add_prefix("Psd_")
        tt = g["t_s"].agg(["min", "max", "size"]).rename(columns={"min": "t_start", "max": "t_end", "size": "n"})
        out = pd.concat([tt, agg, sd], axis=1).reset_index()
        out.insert(0, "phase", phase)
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_cycle_tables(store: Path, out_dir: Path, bin_width: int = 8, overwrite: bool = False) -> dict:
    """Build ``cycles_process.parquet``, ``cycles_oes.parquet`` and ``wafer_spectra.parquet``.

    Per-wafer outputs are cached under ``out_dir/_cache`` so adding new day files only processes the
    new wafers (idempotent, incremental).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "_cache"
    cache.mkdir(exist_ok=True)
    timings = {"process_s": 0.0, "oes_s": 0.0, "oes_wafers": 0, "process_wafers": 0}

    t0 = time.perf_counter()
    proc_parts = []
    for day, lot, wafer, d in wafer_partitions(store, "process"):
        c = cache / f"process_{day}_{wafer:02d}.parquet"
        if overwrite or not c.exists():
            tab = process_cycle_table(d)
            tab.insert(0, "wafer", wafer)
            tab.insert(0, "lot", lot)
            tab.insert(0, "day", day)
            tab.to_parquet(c, index=False)
            timings["process_wafers"] += 1
        proc_parts.append(pd.read_parquet(c))
    if proc_parts:
        pd.concat(proc_parts, ignore_index=True).to_parquet(out_dir / "cycles_process.parquet", index=False)
    timings["process_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    oes_parts, spec_rows = [], []
    oes_wafers = wafer_partitions(store, "oes")
    if oes_wafers:
        decoder, wl, mode = load_oes_meta(store)
        for day, lot, wafer, d in oes_wafers:
            c = cache / f"oes_{day}_{wafer:02d}.parquet"
            cs = cache / f"spec_{day}_{wafer:02d}.parquet"
            if overwrite or not (c.exists() and cs.exists()):
                tw = time.perf_counter()
                wl_cal = apply_calibration(wl, load_wavelength_calibration(store, day, wafer))
                tab, wspec = oes_cycle_table(d, decoder, wl_cal, bin_width=bin_width, decoded=mode != "codes")
                tab.insert(0, "wafer", wafer)
                tab.insert(0, "lot", lot)
                tab.insert(0, "day", day)
                tab.to_parquet(c, index=False)
                srows = [
                    {"day": day, "lot": lot, "wafer": wafer, "phase": ph, **{f"ch{i:04d}": v for i, v in enumerate(s)}}
                    for ph, s in wspec.items()
                ]
                pd.DataFrame(srows).to_parquet(cs, index=False)
                timings["oes_wafers"] += 1
                log.info("cycle table %s wafer %d: %.1fs", day, wafer, time.perf_counter() - tw)
            oes_parts.append(pd.read_parquet(c))
            spec_rows.append(pd.read_parquet(cs))
        pd.concat(oes_parts, ignore_index=True).to_parquet(out_dir / "cycles_oes.parquet", index=False)
        pd.concat(spec_rows, ignore_index=True).to_parquet(out_dir / "wafer_spectra.parquet", index=False)
        cal0 = load_wavelength_calibration(store, oes_wafers[0][0], oes_wafers[0][2])
        pd.DataFrame(
            {"channel": np.arange(wl.size), "wavelength_stored_nm": wl, "wavelength_nm": apply_calibration(wl, cal0)}
        ).to_parquet(out_dir / "wavelengths.parquet", index=False)
    timings["oes_s"] = time.perf_counter() - t0
    return timings
