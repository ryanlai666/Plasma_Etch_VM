"""BOSCH cycle / phase segmentation.

Recipe (Readme.pdf): 1 s ignition, then 100 cycles of 4.5 s SF6 etch + 1.5 s C4F8 passivation.
The gas channels are anonymous (``Gas1Flow`` ... ``Gas8Flow``); from the traces the 600 sccm channel
switching with a 4.5 s / 6 s duty cycle is SF6 (``Gas5Flow``) and the 300 sccm channel with a
1.5 s / 6 s duty cycle is C4F8 (``Gas4Flow``). The mapping is configurable (configs/data.yaml).

Two independent segmenters are provided:

* :func:`label_process` – from tool traces (5 Hz): plasma on/off from source RF power, etch vs. dep
  from normalised SF6 / C4F8 flows.
* :func:`label_from_indicator` – from any 1-D signal that is high in the etch phase (e.g. the OES
  ratio F 703.7 nm / CF2 band), thresholded with Otsu's method. Lets the 25 Hz OES stream be
  segmented without relying on clock alignment with the tool log.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PHASES = ("off", "ign", "etch", "dep")


@dataclass(frozen=True)
class GasMap:
    sf6: str = "Gas5Flow"
    c4f8: str = "Gas4Flow"
    sf6_nominal: float = 600.0
    c4f8_nominal: float = 300.0
    source_power: str = "SourceRFLoadPower"
    plasma_on_w: float = 500.0
    min_frac: float = 0.2


def otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    """Otsu threshold of a 1-D sample (ignores NaN)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0 or np.ptp(x) == 0:
        return float(np.nanmean(x)) if x.size else 0.0
    hist, edges = np.histogram(x, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    mt = (hist * centers).sum()
    m1 = (mt - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between))])


def _smooth_labels(labels: np.ndarray, min_len: int) -> np.ndarray:
    """Merge runs shorter than ``min_len`` into the preceding run (removes 1-sample glitches)."""
    if min_len <= 1 or labels.size == 0:
        return labels
    lab = labels.copy()
    change = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    ends = np.r_[change[1:], lab.size]
    for s, e in zip(change, ends):
        if e - s < min_len and s > 0:
            lab[s:e] = lab[s - 1]
    return lab


def assign_cycles(phase: np.ndarray) -> np.ndarray:
    """Cycle index: increments at every *onset* of an etch phase (0 before the first etch, -1 when off).

    Each cycle therefore = one SF6 etch phase + the C4F8 passivation phase that follows it.
    """
    phase = np.asarray(phase)
    is_etch = phase == "etch"
    onset = is_etch & np.r_[True, ~is_etch[:-1]]
    cyc = np.cumsum(onset).astype(np.int32)
    cyc[phase == "off"] = -1
    return cyc


def label_process(df: pd.DataFrame, gas: GasMap | None = None, min_len: int = 2) -> pd.DataFrame:
    """Label 5 Hz process rows with ``phase`` in {off, ign, etch, dep} and ``cycle``.

    ``ign`` = plasma on before the first etch/dep switching (the 1 s ignition step).
    """
    gas = gas or GasMap()
    f_sf6 = df[gas.sf6].to_numpy(np.float64) / gas.sf6_nominal
    f_c4 = df[gas.c4f8].to_numpy(np.float64) / gas.c4f8_nominal
    on = df[gas.source_power].to_numpy(np.float64) > gas.plasma_on_w
    phase = np.where(f_sf6 >= f_c4, "etch", "dep").astype(object)
    phase[np.maximum(f_sf6, f_c4) < gas.min_frac] = "ign"
    phase[~on] = "off"
    phase = _smooth_labels(phase, min_len)
    # 'ign' only makes sense before the first cycle; later low-flow plasma samples inherit neighbours
    first_cycle = np.flatnonzero((phase == "etch") | (phase == "dep"))
    if first_cycle.size:
        k = first_cycle[0]
        late_ign = np.flatnonzero(phase[k:] == "ign") + k
        for i in late_ign:
            phase[i] = phase[i - 1]
    out = df.copy()
    out["phase"] = phase.astype(str)
    out["cycle"] = assign_cycles(out["phase"].to_numpy())
    return out


def label_from_indicator(indicator: np.ndarray, plasma_on: np.ndarray | None = None, min_len: int = 3) -> np.ndarray:
    """Etch/dep labels from an indicator that is high during etch (Otsu split on its log)."""
    x = np.log(np.clip(np.asarray(indicator, dtype=np.float64), 1e-6, None))
    on = np.ones_like(x, dtype=bool) if plasma_on is None else np.asarray(plasma_on, dtype=bool)
    thr = otsu_threshold(x[on])
    phase = np.where(x >= thr, "etch", "dep").astype(object)
    phase[~on] = "off"
    return _smooth_labels(phase, min_len).astype(str)


def segments(time_s: np.ndarray, phase: np.ndarray, cycle: np.ndarray) -> pd.DataFrame:
    """Run-length table of contiguous phases: phase, cycle, t_start, t_end, duration_s, n."""
    phase = np.asarray(phase)
    if phase.size == 0:
        return pd.DataFrame(columns=["phase", "cycle", "t_start", "t_end", "duration_s", "n"])
    change = np.flatnonzero(np.r_[True, phase[1:] != phase[:-1]])
    ends = np.r_[change[1:], phase.size]
    dt = float(np.median(np.diff(time_s))) if time_s.size > 1 else 0.0
    rows = []
    for s, e in zip(change, ends):
        rows.append(
            {
                "phase": phase[s],
                "cycle": int(cycle[s]),
                "t_start": float(time_s[s]),
                "t_end": float(time_s[e - 1]) + dt,
                "duration_s": float(time_s[e - 1] - time_s[s]) + dt,
                "n": int(e - s),
            }
        )
    return pd.DataFrame(rows)


def cycle_summary(seg: pd.DataFrame) -> dict[str, float]:
    """Timing features of the BOSCH sequence (counts, mean/std phase durations, total times)."""
    etch = seg[seg["phase"] == "etch"]
    dep = seg[seg["phase"] == "dep"]
    on = seg[seg["phase"] != "off"]
    return {
        "n_cycles": float(len(etch)),
        "etch_dur_mean_s": float(etch["duration_s"].mean()) if len(etch) else np.nan,
        "etch_dur_std_s": float(etch["duration_s"].std()) if len(etch) > 1 else np.nan,
        "dep_dur_mean_s": float(dep["duration_s"].mean()) if len(dep) else np.nan,
        "dep_dur_std_s": float(dep["duration_s"].std()) if len(dep) > 1 else np.nan,
        "etch_time_total_s": float(etch["duration_s"].sum()),
        "dep_time_total_s": float(dep["duration_s"].sum()),
        "plasma_time_total_s": float(on["duration_s"].sum()),
    }
