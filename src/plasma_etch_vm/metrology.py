"""Wafer metrology -> VM targets.

Sources (all values in µm, see docs/data_schema.md):
* ``Si_Oxide_etch_89_points.csv`` – 89 sites, measured Feb 2025 (interferometer + profilometer).
  Primary target table (88 wafers).
* ``Si_Oxide_etch_9_points.csv`` – 9 sites, measured on the day of the experiment
  (reflectometer + Dektak). Secondary target table (lots 8 and 10 missing).
* ``Lot_status.xlsx`` – lot -> date and conditioning recipe ("3C - Si" = 3 conditioning repeats on a
  blank Si wafer; "C"/no suffix = on the chuck).

Wafer key used everywhere: ``wafer_id = "YYYY-MM-DD_WW"`` (identical to the CSV ``experiment_key``).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

LOT_DATES = {
    1: "2024-07-02",
    2: "2024-07-05",
    3: "2024-07-09",
    4: "2024-07-11",
    5: "2024-07-19",
    6: "2024-08-01",
    7: "2024-08-05",
    8: "2024-08-07",
    9: "2024-08-21",
    10: "2024-08-22",
}
DATE_TO_LOT = {v: k for k, v in LOT_DATES.items()}

WAFER_RADIUS_MM = 100.0


def wafer_id(day: str, wafer: int) -> str:
    """``wafer_id('2024-08-05', 3) -> '2024-08-05_03'``."""
    return f"{day}_{int(wafer):02d}"


def parse_group_name(name: str) -> tuple[str, int]:
    """Parse ``Day_2024_08_05_Wafer_03`` (process file) or ``Wafer_03`` + day (OES file)."""
    m = re.fullmatch(r"Day_(\d{4})_(\d{2})_(\d{2})_Wafer_(\d+)", name)
    if not m:
        raise ValueError(f"not a process group name: {name}")
    return f"{m[1]}-{m[2]}-{m[3]}", int(m[4])


def day_from_filename(path: str | Path) -> str:
    m = re.search(r"Day_(\d{4})_(\d{2})_(\d{2})", Path(path).name)
    if not m:
        raise ValueError(f"cannot parse day from {path}")
    return f"{m[1]}-{m[2]}-{m[3]}"


def load_lot_status(path: Path) -> pd.DataFrame:
    """Lot table with parsed conditioning design (n_cond in {1,3,9}, cond_substrate in {chuck,Si,SiO2})."""
    raw = pd.read_excel(path, header=None)
    rows = []
    for _, r in raw.iterrows():
        m = re.fullmatch(r"Lot\s+(\d+)", str(r[0]).strip())
        if not m:
            continue
        lot = int(m[1])
        typ = str(r[4]).strip()
        n_cond = int(re.match(r"(\d+)C", typ)[1])
        sub = "chuck"
        if "SiO2" in typ:
            sub = "SiO2"
        elif "Si" in typ:
            sub = "Si"
        rows.append(
            {
                "lot": lot,
                "day": LOT_DATES[lot],
                "n_wafers": int(r[2]),
                "measurements": str(r[3]).strip(),
                "conditioning": typ,
                "n_cond": n_cond,
                "cond_substrate": sub,
            }
        )
    return pd.DataFrame(rows)


def _attach_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["wafer_id"] = df["experiment_key"].astype(str)
    df["day"] = df["wafer_id"].str.slice(0, 10)
    df["wafer"] = df["wafer_id"].str.slice(11).astype(int)
    df["lot"] = df["day"].map(DATE_TO_LOT).astype(int)
    df["x_mm"] = df["X"] / 1000.0
    df["y_mm"] = df["Y"] / 1000.0
    df["r_mm"] = np.hypot(df["x_mm"], df["y_mm"])
    df["r_norm"] = df["r_mm"] / WAFER_RADIUS_MM
    df["theta"] = np.arctan2(df["y_mm"], df["x_mm"])
    return df


def load_points(path: Path) -> pd.DataFrame:
    """Load a 9- or 89-point CSV as a tidy point table with ids and polar coordinates.

    Rows whose ``experiment_key`` is missing are dropped (they cannot be linked to a wafer).
    """
    df = pd.read_csv(path)
    df = df[df["experiment_key"].notna()]
    df = _attach_ids(df)
    df["selectivity"] = df["si_etch"] / df["oxide_etch"]
    keep = [
        "wafer_id",
        "day",
        "lot",
        "wafer",
        "x_mm",
        "y_mm",
        "r_mm",
        "r_norm",
        "theta",
        "preox_thickness",
        "postox_thickness",
        "stepheight",
        "oxide_etch",
        "si_etch",
        "selectivity",
    ]
    if "loc_id" in df.columns:
        keep.insert(4, "loc_id")
    if "postox_thickness_nan" in df.columns:
        df["post_ox_interpolated"] = df["postox_thickness_nan"].isna()
        keep.append("post_ox_interpolated")
    return df[keep].reset_index(drop=True)


# The 9-point table uses a coordinate frame rotated by 180° w.r.t. the 89-point table: after
# (x, y) -> (-x, -y) the site-level Si etch depths of the two campaigns correlate at r = 0.995
# (657 matched site rows), versus r = -0.06 without rotation (see docs/data_schema.md).
NINE_POINT_ROTATION_DEG = 180


def rotate_points(points: pd.DataFrame, deg: int = NINE_POINT_ROTATION_DEG) -> pd.DataFrame:
    """Rotate site coordinates by a multiple of 90° (and recompute polar coordinates)."""
    if deg % 90:
        raise ValueError("only multiples of 90° are supported")
    k = (deg // 90) % 4
    out = points.copy()
    x, y = out["x_mm"].to_numpy(), out["y_mm"].to_numpy()
    for _ in range(k):
        x, y = -y, x
    out["x_mm"], out["y_mm"] = x + 0.0, y + 0.0
    out["theta"] = np.arctan2(out["y_mm"], out["x_mm"])
    return out


def wafer_targets(points: pd.DataFrame, n_cycles: pd.Series | None = None) -> pd.DataFrame:
    """Aggregate point measurements to wafer-level VM targets.

    * ``si_etch_mean``        mean Si etch depth [µm]
    * ``si_nu_halfrange_pct`` within-wafer non-uniformity, 100·(max−min)/(2·mean) [%]
    * ``si_nu_std_pct``       1-σ non-uniformity, 100·std/mean [%]
    * ``si_nu_p90_pct``       robust non-uniformity, 100·(P95−P5)/(2·median) [%]
    * ``si_edge_center``      mean depth at r>=0.75R minus mean at r<=0.25R [µm] (radial profile)
    * ``ox_etch_mean``        mean SiO2 mask loss [µm]
    * ``selectivity``         si_etch_mean / ox_etch_mean
    * ``si_rate_um_per_cycle`` si_etch_mean / n_cycles (nominal 100 if unknown)
    """
    g = points.groupby("wafer_id")
    out = pd.DataFrame(
        {
            "day": g["day"].first(),
            "lot": g["lot"].first(),
            "wafer": g["wafer"].first(),
            "n_points": g.size(),
            "si_etch_mean": g["si_etch"].mean(),
            "si_etch_std": g["si_etch"].std(ddof=1),
            "si_etch_min": g["si_etch"].min(),
            "si_etch_max": g["si_etch"].max(),
            "ox_etch_mean": g["oxide_etch"].mean(),
            "ox_etch_std": g["oxide_etch"].std(ddof=1),
        }
    )
    out["si_nu_halfrange_pct"] = 100 * (out["si_etch_max"] - out["si_etch_min"]) / (2 * out["si_etch_mean"])
    # robust variant: 5th-95th percentile half-range relative to the median (insensitive to one bad site)
    q05, q95, med = g["si_etch"].quantile(0.05), g["si_etch"].quantile(0.95), g["si_etch"].median()
    out["si_nu_p90_pct"] = 100 * (q95 - q05) / (2 * med)
    out["si_nu_std_pct"] = 100 * out["si_etch_std"] / out["si_etch_mean"]
    out["selectivity"] = out["si_etch_mean"] / out["ox_etch_mean"]
    edge = points[points["r_norm"] >= 0.75].groupby("wafer_id")["si_etch"].mean()
    center = points[points["r_norm"] <= 0.25].groupby("wafer_id")["si_etch"].mean()
    out["si_edge_center"] = edge - center
    cyc = n_cycles.reindex(out.index).fillna(100.0) if n_cycles is not None else 100.0
    out["si_rate_um_per_cycle"] = out["si_etch_mean"] / cyc
    return out.reset_index()


TARGET_COLUMNS = [
    "si_etch_mean",
    "si_nu_halfrange_pct",
    "si_nu_p90_pct",
    "si_nu_std_pct",
    "si_edge_center",
    "ox_etch_mean",
    "selectivity",
]
