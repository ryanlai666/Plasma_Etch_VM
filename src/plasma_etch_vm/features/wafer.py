"""Wafer-level feature table from the per-cycle tables.

Feature groups (column prefixes) — models select groups by name (configs/features.yaml):

* ``tool``       ``T_<phase>_<param>_<stat>``  tool-parameter statistics over cycles
* ``timing``     ``TM_tool_*``                 cycle count, phase durations (tool log)
* ``oes``        ``O_<phase>_<line>_<stat>``   normalised line intensities over cycles
* ``oes_timing`` ``TM_oes_*``                  cycle count, phase durations (OES clock)
* ``ratio``      ``R_<phase>_<name>_<stat>``   line / cross-phase log-ratios (no actinometer gas present)
* ``transition`` ``X_<metric>_<stat>``         10-90 % step-switch times, F overshoot
* ``context``  ``C_*``                       slot in lot (wafers since conditioning), conditioning design
* ``spectrum`` ``S_<phase>_bNNN``            wafer-mean binned spectrum (normalised) -> PCA/NMF in-fold

Statistics over the ~100 cycles: ``mean``, ``std``, ``slope`` (change per 100 cycles from a linear
fit — intra-wafer drift) and ``d`` (mean of last 10 cycles − mean of first 10).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrology import wafer_id

STATS = ("mean", "std", "slope", "d")


def drift_stats(cycle: np.ndarray, y: np.ndarray, edge: int = 10) -> dict[str, float]:
    cycle = np.asarray(cycle, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    cycle, y = cycle[ok], y[ok]
    if y.size == 0:
        return dict.fromkeys(STATS, np.nan)
    order = np.argsort(cycle)
    cycle, y = cycle[order], y[order]
    slope = np.polyfit(cycle, y, 1)[0] * 100 if y.size >= 3 and np.ptp(cycle) > 0 else np.nan
    e = min(edge, max(1, y.size // 3))
    return {
        "mean": float(y.mean()),
        "std": float(y.std(ddof=1)) if y.size > 1 else np.nan,
        "slope": float(slope),
        "d": float(y[-e:].mean() - y[:e].mean()),
    }


def _main_cycles(tab: pd.DataFrame, min_frac: float = 0.5) -> pd.DataFrame:
    """Drop partial cycles (cycle 0 / ramp-up, truncated tail) with too few samples."""
    if tab.empty:
        return tab
    tab = tab[tab["cycle"] >= 1]
    med = tab.groupby("phase")["n"].transform("median")
    return tab[tab["n"] >= min_frac * med]


def _wid(df: pd.DataFrame) -> pd.Series:
    return pd.Series([wafer_id(d, w) for d, w in zip(df["day"], df["wafer"])], index=df.index)


def tool_features(cyc_proc: pd.DataFrame, min_rel_std: float = 1e-4) -> pd.DataFrame:
    """Statistics of per-cycle tool-parameter means, per phase. Drops (near-)constant parameters."""
    cyc_proc = _main_cycles(cyc_proc.copy())
    cyc_proc["wafer_id"] = _wid(cyc_proc)
    pcols = [c for c in cyc_proc.columns if c.startswith("P_")]
    keep = []
    for c in pcols:
        v = cyc_proc[c].to_numpy(np.float64)
        scale = np.nanmean(np.abs(v)) + 1e-12
        if np.nanstd(v) / scale > min_rel_std:
            keep.append(c)
    rows = {}
    for (wid, phase), g in cyc_proc.groupby(["wafer_id", "phase"]):
        r = rows.setdefault(wid, {})
        for c in keep:
            for s, v in drift_stats(g["cycle"].to_numpy(), g[c].to_numpy()).items():
                r[f"T_{phase}_{c[2:]}_{s}"] = v
        sd_cols = [c for c in g.columns if c.startswith("Psd_") and "P_" + c[4:] in keep]
        for c in sd_cols:
            r[f"T_{phase}_{c[4:]}_withinsd"] = float(g[c].mean())
        dur = (g["t_end"] - g["t_start"]).to_numpy()
        r[f"TM_tool_{phase}_n"] = float(len(g))
        r[f"TM_tool_{phase}_samples_mean"] = float(g["n"].mean())
        r[f"TM_tool_{phase}_span_mean_s"] = float(np.mean(dur))
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def oes_features(cyc_oes: pd.DataFrame, lines: list[str], ratios: dict[str, list]) -> pd.DataFrame:
    """Line intensities (normalised by total emission) and log-ratios per phase, aggregated over cycles.

    ``ratios``: name -> ``[phase, num_line, den_line]`` (same phase) or
    ``[num_phase, num_line, den_phase, den_line]`` (cross-phase, matched per cycle).
    """
    cyc = _main_cycles(cyc_oes.copy())
    cyc["wafer_id"] = _wid(cyc)
    eps = 1.0
    rows = {}
    for (wid, phase), g in cyc.groupby(["wafer_id", "phase"]):
        r = rows.setdefault(wid, {})
        k = g["cycle"].to_numpy()
        tot = g["total"].to_numpy(np.float64)
        for s, v in drift_stats(k, tot / 1e6).items():
            r[f"O_{phase}_total_{s}"] = v
        r[f"O_{phase}_totalsd_mean"] = float((g["total_std"] / g["total"]).mean())
        for ln in lines:
            col = f"L_{ln}"
            if col not in g:
                continue
            y = g[col].to_numpy(np.float64) / tot * 1e3  # per-mille of total emission
            for s, v in drift_stats(k, y).items():
                r[f"O_{phase}_{ln}_{s}"] = v
        dur = (g["t_end"] - g["t_start"]).to_numpy()
        r[f"TM_oes_{phase}_n"] = float(len(g))
        r[f"TM_oes_{phase}_span_mean_s"] = float(np.mean(dur))
        r[f"TM_oes_{phase}_span_std_s"] = float(np.std(dur))
    for name, spec in ratios.items():
        if len(spec) == 3:
            num_phase, num, den_phase, den = spec[0], spec[1], spec[0], spec[2]
        else:
            num_phase, num, den_phase, den = spec
        if f"L_{num}" not in cyc or f"L_{den}" not in cyc:
            continue
        a = cyc[cyc["phase"] == num_phase][["wafer_id", "cycle", f"L_{num}"]].rename(columns={f"L_{num}": "num"})
        b = cyc[cyc["phase"] == den_phase][["wafer_id", "cycle", f"L_{den}"]].rename(columns={f"L_{den}": "den"})
        m = a.merge(b, on=["wafer_id", "cycle"])
        tag = num_phase if num_phase == den_phase else f"{num_phase}_{den_phase}"
        for wid, g in m.groupby("wafer_id"):
            y = np.log((g["num"].clip(lower=0) + eps) / (g["den"].clip(lower=0) + eps))
            for s, v in drift_stats(g["cycle"].to_numpy(), y.to_numpy()).items():
                rows[wid][f"R_{tag}_{name}_{s}"] = v
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def spectrum_features(wafer_spectra: pd.DataFrame, bin_width: int = 8, phases=("etch", "dep")) -> pd.DataFrame:
    """Wafer-mean spectra, binned and area-normalised, as ``S_<phase>_bNNN`` columns."""
    from .cycles import bin_spectrum

    ws = wafer_spectra.copy()
    ws["wafer_id"] = _wid(ws)
    ch = [c for c in ws.columns if c.startswith("ch")]
    parts = []
    for ph in phases:
        g = ws[ws["phase"] == ph].set_index("wafer_id")
        if g.empty:
            continue
        x = bin_spectrum(g[ch].to_numpy(np.float64), bin_width)
        x = x / np.clip(x.sum(axis=1, keepdims=True), 1e-9, None) * x.shape[1]
        parts.append(pd.DataFrame(x, index=g.index, columns=[f"S_{ph}_b{j:03d}" for j in range(x.shape[1])]))
    return pd.concat(parts, axis=1).sort_index() if parts else pd.DataFrame()


def context_features(wafer_ids: list[str], lot_status: pd.DataFrame) -> pd.DataFrame:
    """Run context known before metrology: slot in lot and conditioning design (one-hot)."""
    ls = lot_status.set_index("day")
    rows = {}
    for wid in wafer_ids:
        day, slot = wid[:10], int(wid[11:])
        lot = ls.loc[day]
        rows[wid] = {
            "C_slot": float(slot),
            "C_log_slot": float(np.log(slot)),
            "C_n_cond": float(lot["n_cond"]),
            "C_cond_chuck": float(lot["cond_substrate"] == "chuck"),
            "C_cond_Si": float(lot["cond_substrate"] == "Si"),
            "C_cond_SiO2": float(lot["cond_substrate"] == "SiO2"),
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


GROUP_PREFIXES = {
    "tool": ("T_",),
    "timing": ("TM_tool_",),
    "oes": ("O_",),
    "oes_timing": ("TM_oes_",),
    "ratio": ("R_",),
    "transition": ("X_",),
    "context": ("C_",),
    "cycle_count": ("TM_tool_etch_n", "TM_tool_dep_n"),  # number of BOSCH steps logged (99 vs 100)
    "spectrum": ("S_",),
}


def select_groups(columns: list[str], groups: list[str]) -> list[str]:
    prefixes = tuple(p for g in groups for p in GROUP_PREFIXES[g])
    return [c for c in columns if c.startswith(prefixes)]
