"""Step-transition timing from the ~24 Hz OES trace (``oes_trace`` partitions).

At every SF6 -> C4F8 switch the F I emission decays and the C2 Swan emission rises; at every
C4F8 -> SF6 switch the reverse happens. The 10–90 % transition times reflect the gas-exchange
(residence) time τ ≈ pV/Q of the chamber plus the MFC / valve response, and drift with chamber
state. Also extracted: F overshoot at the start of the etch step (the high-bias polymer
break-through sub-step) relative to the etch plateau.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from ..metrology import wafer_id
from .wafer import drift_stats


def _crossing_time(t: np.ndarray, y: np.ndarray, level: float, rising: bool) -> float:
    """First time ``y`` crosses ``level`` (linear interpolation); NaN if it never does."""
    above = y >= level if rising else y <= level
    idx = np.flatnonzero(above)
    if idx.size == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return float(t[0])
    y0, y1 = y[i - 1], y[i]
    frac = 0.0 if y1 == y0 else (level - y0) / (y1 - y0)
    return float(t[i - 1] + np.clip(frac, 0, 1) * (t[i] - t[i - 1]))


def transition_time(t: np.ndarray, y: np.ndarray, lo_level: float, hi_level: float, rising: bool) -> float:
    """10–90 % transition time between two plateau levels."""
    if not np.isfinite(lo_level) or not np.isfinite(hi_level) or hi_level - lo_level <= 0:
        return np.nan
    a = lo_level + 0.1 * (hi_level - lo_level)
    b = lo_level + 0.9 * (hi_level - lo_level)
    if rising:
        return _crossing_time(t, y, b, True) - _crossing_time(t, y, a, True)
    return _crossing_time(t, y, a, False) - _crossing_time(t, y, b, False)


def _runs(phase: np.ndarray) -> list[tuple[int, int, str]]:
    change = np.flatnonzero(np.r_[True, phase[1:] != phase[:-1]])
    ends = np.r_[change[1:], phase.size]
    return [(int(s), int(e), str(phase[s])) for s, e in zip(change, ends)]


def wafer_transitions(
    trace: pd.DataFrame, f_col: str = "L_F_685", c_col: str = "L_C2_516", pad: int = 12, overshoot_s: float = 0.5
) -> pd.DataFrame:
    """Per-cycle transition metrics for one wafer trace."""
    t = trace["t_s"].to_numpy(np.float64)
    ph = trace["phase"].to_numpy()
    cyc = trace["cycle"].to_numpy()
    F = trace[f_col].to_numpy(np.float64)
    C = trace[c_col].to_numpy(np.float64)
    runs = _runs(ph)
    rows = []
    for (s0, e0, p0), (s1, e1, p1) in pairwise(runs):
        if {p0, p1} != {"etch", "dep"} or (e0 - s0) < 6 or (e1 - s1) < 6:
            continue
        # plateau levels from the second half of each run
        f0 = np.median(F[(s0 + e0) // 2 : e0])
        f1 = np.median(F[(s1 + e1) // 2 : e1])
        c0 = np.median(C[(s0 + e0) // 2 : e0])
        c1 = np.median(C[(s1 + e1) // 2 : e1])
        a, b = max(s0, s1 - pad), min(e1, s1 + pad)
        tw = t[a:b]
        row = {"cycle": int(cyc[s1]) if p1 == "etch" else int(cyc[s0]), "switch": f"{p0}->{p1}"}
        if p0 == "etch":  # etch -> dep: F falls, C2 rises
            row["F_fall_s"] = transition_time(tw, F[a:b], f1, f0, rising=False)
            row["C2_rise_s"] = transition_time(tw, C[a:b], c0, c1, rising=True)
        else:  # dep -> etch: F rises, C2 falls
            row["F_rise_s"] = transition_time(tw, F[a:b], f0, f1, rising=True)
            row["C2_fall_s"] = transition_time(tw, C[a:b], c1, c0, rising=False)
            early = (t >= t[s1]) & (t < t[s1] + overshoot_s)
            row["F_overshoot"] = float(np.mean(F[early]) / f1) if f1 > 0 and early.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def transition_features(traces: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    """Wafer-level ``X_<metric>_<stat>`` features from ``{(day, wafer): trace}``."""
    out = {}
    for (day, wafer), tr in traces.items():
        per = wafer_transitions(tr)
        r = {}
        for m in ("F_fall_s", "C2_rise_s", "F_rise_s", "C2_fall_s", "F_overshoot"):
            if m not in per:
                continue
            g = per[["cycle", m]].dropna()
            g = g[g["cycle"] >= 1]
            for s, v in drift_stats(g["cycle"].to_numpy(), g[m].to_numpy()).items():
                r[f"X_{m}_{s}"] = v
        out[wafer_id(day, wafer)] = r
    return pd.DataFrame.from_dict(out, orient="index").sort_index()
