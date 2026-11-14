"""End-to-end: synthetic netCDF -> partitioned Parquet -> cycle tables -> wafer features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from plasma_etch_vm import netcdf_io as nio
from plasma_etch_vm.ingest import Manifest, SchemaError, encode_spectra, ingest_oes_file, run_ingest

from .conftest import N_CYCLES, write_day


def test_readers_decode_blocks(raw_dir):
    dec = nio.load_decoder(raw_dir / "Dictionary_OES.nc")
    assert nio.list_groups(raw_dir / "Day_2024_08_05.nc") == ["Wafer_01", "Wafer_02"]
    assert nio.hdf5_chunk_rows(raw_dir / "Day_2024_08_05.nc", "Wafer_01") == 200
    blocks = list(nio.iter_blocks(raw_dir / "Day_2024_08_05.nc", "Wafer_01", "auto"))
    assert all(b[2].shape[0] <= 200 for b in blocks)
    assert blocks[0][2].dtype == np.uint16 and dec[blocks[0][2]].dtype == np.float32
    info = nio.group_info(raw_dir / "Process_data.nc", "Day_2024_08_05_Wafer_01", "process")
    assert "Gas5Flow" in info.columns  # prefix stripped


def test_ingest_partitions_and_rows(ingested):
    cfg, stats = ingested
    root = Path(cfg["paths"]["parquet"])
    oes = stats[stats["kind"] == "oes"]
    assert len(oes) == 2 and (oes["rows"] > 0).all()
    steps = {p.name for p in (root / "oes" / "day=2024-08-05" / "lot=7" / "wafer=1").iterdir()}
    assert {"step=etch", "step=dep", "step=off"} <= steps
    n = sum(
        pq.ParquetFile(f).metadata.num_rows
        for f in (root / "oes" / "day=2024-08-05" / "lot=7" / "wafer=1").rglob("*.parquet")
    )
    assert n == int(oes.iloc[0]["rows"])
    cal = json.loads((root / "meta" / "wavelength_calibration" / "2024-08-05_01.json").read_text())
    assert cal["ok"] and abs(cal["shift_nm"] - 1.22) < 0.05
    proc = pd.read_parquet(root / "process" / "day=2024-08-05" / "lot=7" / "wafer=1" / "step=etch")
    assert {"t_s", "cycle", "Gas5Flow"} <= set(proc.columns)


def test_ingest_is_idempotent(ingested):
    cfg, _ = ingested
    again = run_ingest(cfg)
    assert again["skipped"].all()


def test_oes_phase_and_cycles(ingested):
    cfg, _ = ingested
    tr = pd.read_parquet(Path(cfg["paths"]["parquet"]) / "oes_trace" / "day=2024-08-05" / "lot=7" / "wafer=1")
    assert tr.loc[tr["phase"] == "etch", "cycle"].nunique() == N_CYCLES


def test_schema_validation_rejects_bad_codes(raw_dir, ingest_cfg, tmp_path):
    bad = tmp_path / "Day_2024_08_05.nc"
    write_day(bad, n_wafers=1, bad_code=True)
    with pytest.raises(SchemaError):
        ingest_oes_file(ingest_cfg, bad, tmp_path / "store", Manifest(tmp_path / "store"), force=True)


def test_counts_i16_quantisation_error():
    table = np.arange(-10, 10, 0.07, dtype=np.float32)
    codes = np.arange(table.size, dtype=np.uint16)[None, :]
    err = encode_spectra(codes, table, "counts_i16").astype(np.float32) - table[codes]
    assert np.abs(err).max() <= 0.5 + 1e-6


def test_cycles_and_wafer_features(ingested):
    from plasma_etch_vm.features.cycles import build_cycle_tables
    from plasma_etch_vm.features.transitions import transition_features
    from plasma_etch_vm.features.wafer import oes_features, tool_features

    cfg, _ = ingested
    store, feat = Path(cfg["paths"]["parquet"]), Path(cfg["paths"]["features"])
    t = build_cycle_tables(store, feat, bin_width=8)
    assert t["oes_wafers"] == 2 and t["process_wafers"] == 2
    co = pd.read_parquet(feat / "cycles_oes.parquet")
    assert co[co["phase"] == "etch"]["L_F_685"].mean() > co[co["phase"] == "dep"]["L_F_685"].mean()
    assert co[co["phase"] == "dep"]["L_C2_516"].mean() > co[co["phase"] == "etch"]["L_C2_516"].mean()
    o = oes_features(
        co, ["F_685", "C2_516"], {"c2f": ["dep", "C2_516", "F_685"], "x": ["dep", "F_685", "etch", "F_685"]}
    )
    assert "R_dep_c2f_mean" in o.columns and "R_dep_etch_x_mean" in o.columns
    tf = tool_features(pd.read_parquet(feat / "cycles_process.parquet"))
    assert any(c.startswith("T_etch_PlatenRFLoadCapacitor") for c in tf.columns)
    assert not any("Gas8Flow" in c for c in tf.columns)  # constant channel dropped
    tr = {("2024-08-05", 1): pd.read_parquet(store / "oes_trace" / "day=2024-08-05" / "lot=7" / "wafer=1")}
    x = transition_features(tr)
    assert np.isfinite(x["X_F_fall_s_mean"].iloc[0]) and 0 < x["X_F_fall_s_mean"].iloc[0] < 1.0


def test_store_queries(ingested):
    from plasma_etch_vm.store import Store

    cfg, _ = ingested
    s = Store(cfg["paths"]["parquet"])
    counts = s.sql("SELECT step, count(*) AS n FROM oes GROUP BY step").set_index("step")["n"]
    ls = s.line_series("F_685", "etch")
    assert len(ls) == counts["etch"] and ls["L_F_685"].median() > 1000
    agg = s.scan("process").group_by("wafer").len().collect()
    assert agg.height == 2
