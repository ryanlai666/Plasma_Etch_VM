"""Query layer over the hive-partitioned Parquet store (DuckDB SQL + Polars lazy scans).

The store is partitioned ``kind/day=/lot=/wafer=/step=`` and the OES table is *wide* (one column per
detector channel), so a query that needs a few emission lines reads only those column chunks of only
the matching partitions (partition + projection pushdown) — e.g. the F I 685.6 nm window over all
wafers is ~0.2 % of the spectral data.

    from plasma_etch_vm.store import Store
    s = Store("data/parquet")
    s.sql("SELECT day, wafer, count(*) FROM oes WHERE step = 'etch' GROUP BY ALL")
    s.line_series("F_685", step="etch")          # per-sample line intensity, all wafers
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .lines import LINE_BY_NAME, apply_calibration, line_windows


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    # ------------------------------------------------------------------------------ duckdb
    def _glob(self, kind: str) -> str:
        return (self.root / kind / "**" / "*.parquet").as_posix()

    def connect(self, union_by_name: bool = False):
        """DuckDB connection with one view per table.

        All partitions of a table share one schema (validated at ingestion), so ``union_by_name`` is off
        by default: with ~1 000 files × 3 651 columns, unifying schemas by name dominated query time.
        """
        import duckdb

        con = duckdb.connect()
        for kind in ("oes", "process", "oes_trace"):
            if (self.root / kind).exists():
                con.execute(
                    f"CREATE VIEW {kind} AS SELECT * FROM read_parquet('{self._glob(kind)}', "
                    f"hive_partitioning = true, union_by_name = {str(union_by_name).lower()})"
                )
        return con

    def sql(self, query: str) -> pd.DataFrame:
        con = self.connect()
        try:
            return con.execute(query).df()
        finally:
            con.close()

    # ------------------------------------------------------------------------------ polars
    def scan(self, kind: str):
        """Polars LazyFrame over one table (hive partitions become columns)."""
        import polars as pl

        return pl.scan_parquet(self._glob(kind), hive_partitioning=True)

    # ------------------------------------------------------------------------------ helpers
    def wavelengths(self, calibrated: bool = True) -> np.ndarray:
        import json

        wl = pd.read_parquet(self.root / "meta" / "wavelengths.parquet")["wavelength_nm"].to_numpy()
        cals = sorted((self.root / "meta" / "wavelength_calibration").glob("*.json"))
        if calibrated and cals:
            wl = apply_calibration(wl, json.loads(cals[0].read_text()))
        return wl

    def line_channels(self, line: str) -> tuple[list[str], list[str]]:
        w = line_windows(self.wavelengths(), (LINE_BY_NAME[line],))[line]
        return [f"ch{i:04d}" for i in w.peak], [f"ch{i:04d}" for i in w.bg]

    def line_series(self, line: str, step: str = "etch") -> pd.DataFrame:
        """Per-sample background-subtracted line intensity for all wafers, reading only the
        line's peak + background channels of the requested step partitions (DuckDB)."""
        peak, bg = self.line_channels(line)
        cols = ", ".join(peak + bg)
        df = self.sql(f"SELECT day, lot, wafer, t_s, cycle, {cols} FROM oes WHERE step = '{step}'")
        val = df[peak].to_numpy(np.float32).max(axis=1) - np.quantile(df[bg].to_numpy(np.float32), 0.2, axis=1)
        return df[["day", "lot", "wafer", "t_s", "cycle"]].assign(**{f"L_{line}": val})


def query_benchmark(root: str | Path, line: str = "F_685") -> dict:
    """Time a full-table aggregate vs. a projected line query (projection / partition pushdown)."""
    s = Store(root)
    out = {}
    t0 = time.perf_counter()
    n = s.sql("SELECT count(*) AS n FROM oes")["n"].iloc[0]
    out["count_rows"] = int(n)
    out["count_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ls = s.line_series(line, "etch")
    out["line_query_rows"] = len(ls)
    out["line_query_seconds"] = time.perf_counter() - t0
    peak, bg = s.line_channels(line)
    out["line_query_columns"] = len(peak) + len(bg)
    t0 = time.perf_counter()
    lf = s.scan("process")
    import polars as pl

    agg = (
        lf.filter(pl.col("step") == "etch")
        .group_by(["day", "wafer"])
        .agg(pl.col("Pressure").mean().alias("p_mean"), pl.col("PlatenRFLoadPower").mean().alias("platen_w"))
        .collect()
    )
    out["polars_process_agg_rows"] = agg.height
    out["polars_process_agg_seconds"] = time.perf_counter() - t0
    return out
