"""Neural VM models (PyTorch).

* :class:`SpectralCNN1D` – 1-D CNN on window-averaged OES spectra, input ``(B, n_phase, n_bins)``.
  Trained multi-instance: each wafer contributes one instance per window of cycles; wafer
  predictions are the mean over its windows.
* :class:`StatFETransformer` – transformer with *statistical feature embedding* and *window
  positional encoding* following Lee & Kwon, arXiv:2407.06682: every token is a (sensor, window)
  pair whose embedding is the vector of ``d`` window statistics (mean, max, min, var, range, slope);
  the sinusoidal encoding is indexed by the *window* so all sensors of a window share it; one encoder
  block; flattened output -> linear regression head. ``d_model=None`` keeps the paper's design
  (embedding dim = number of statistics); an optional linear projection is available.

Dataset builders turn the per-cycle tables (:mod:`plasma_etch_vm.features.cycles`) into tensors.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..metrology import wafer_id

STAT_NAMES = ("mean", "max", "min", "var", "range", "slope")


# ------------------------------------------------------------------------------------- models
class SpectralCNN1D(nn.Module):
    def __init__(self, in_channels: int, n_targets: int, width: int = 32, depth: int = 4, dropout: float = 0.2):
        super().__init__()
        layers, c = [], in_channels
        for i in range(depth):
            out = width * min(2**i, 4)
            layers += [
                nn.Conv1d(c, out, kernel_size=7, padding=3),
                nn.BatchNorm1d(out),
                nn.GELU(),
                nn.MaxPool1d(2),
            ]
            c = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c, 64), nn.GELU(), nn.Linear(64, n_targets))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return self.head(h)


def window_positional_encoding(n_sensors: int, n_windows: int, d: int) -> torch.Tensor:
    """P[pos, 2i] = sin(win(pos)/10000^(2i/d)), P[pos, 2i+1] = cos(...), win(pos) = ceil(pos/p) (1-based)."""
    pos = torch.arange(1, n_sensors * n_windows + 1, dtype=torch.float32)
    win = torch.ceil(pos / n_sensors)[:, None]
    i = torch.arange(d, dtype=torch.float32)[None, :]
    div = torch.pow(10000.0, (2 * torch.div(i, 2, rounding_mode="floor")) / d)
    angle = win / div
    pe = torch.where(i.long() % 2 == 0, torch.sin(angle), torch.cos(angle))
    return pe  # (p*w, d)


class StatFETransformer(nn.Module):
    def __init__(
        self,
        n_sensors: int,
        n_windows: int,
        n_stats: int,
        n_targets: int,
        d_model: int | None = None,
        n_heads: int = 1,
        n_layers: int = 1,
        ff_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tokens = n_sensors * n_windows
        d = d_model or n_stats
        self.proj = nn.Identity() if d_model is None else nn.Linear(n_stats, d_model)
        self.register_buffer("pe", window_positional_encoding(n_sensors, n_windows, d), persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(self.n_tokens * d, n_targets))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, p*w, n_stats)
        h = self.proj(x) + self.pe
        return self.head(self.encoder(h))


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ----------------------------------------------------------------------------------- datasets
def _window_index(cycle: np.ndarray, n_windows: int, max_cycle: int) -> np.ndarray:
    return np.clip(((cycle - 1) * n_windows) // max_cycle, 0, n_windows - 1)


def spectra_windows(
    cyc_oes: pd.DataFrame, n_windows: int = 10, max_cycle: int = 100, phases=("etch", "dep")
) -> tuple[np.ndarray, list[str]]:
    """Window-averaged binned spectra -> ``X (n_wafers, n_windows, n_phases, n_bins)`` + wafer ids.

    Each spectrum is area-normalised then log-scaled (robust to slow intensity drift of the optics).
    """
    bcols = [c for c in cyc_oes.columns if c.startswith("b") and c[1:].isdigit()]
    cyc = cyc_oes[cyc_oes["cycle"] >= 1].copy()
    cyc["win"] = _window_index(cyc["cycle"].to_numpy(), n_windows, max_cycle)
    ids, xs = [], []
    for (day, wafer), g in cyc.groupby(["day", "wafer"]):
        x = np.zeros((n_windows, len(phases), len(bcols)), dtype=np.float32)
        for j, ph in enumerate(phases):
            gp = g[g["phase"] == ph]
            m = gp.groupby("win")[bcols].mean().reindex(range(n_windows)).ffill().bfill()
            x[:, j, :] = m.to_numpy(np.float32)
        x = np.clip(x, 1e-3, None)
        x = x / x.mean(axis=-1, keepdims=True)
        xs.append(np.log(x))
        ids.append(wafer_id(day, wafer))
    return np.stack(xs), ids


def _slope(y: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=np.float64)
    x -= x.mean()
    return float((x * (y - y.mean())).sum() / (x**2).sum())


def statfe_tokens(
    cyc_tables: list[pd.DataFrame],
    signal_prefixes=("P_", "L_"),
    n_windows: int = 10,
    max_cycle: int = 100,
    phases=("etch", "dep"),
    min_rel_std: float = 1e-4,
) -> tuple[np.ndarray, list[str], list[str]]:
    """StatFE input tensor ``(n_wafers, p*w, 6)`` from per-cycle tables (process and/or OES).

    A *sensor* is a (phase, signal) pair, e.g. ``etch:P_Pressure`` or ``dep:L_CF2_252``; the window
    statistics are taken over the per-cycle values inside each window of ``max_cycle/n_windows``
    cycles. Token order is window-major (all sensors of window 1, then window 2, ...) as in the paper.
    """
    long_parts, wafer_sets = [], []
    for tab in cyc_tables:
        sig = [c for c in tab.columns if c.startswith(tuple(signal_prefixes))]
        t = tab[(tab["cycle"] >= 1) & tab["phase"].isin(phases)]
        m = t.melt(id_vars=["day", "wafer", "phase", "cycle"], value_vars=sig, var_name="signal")
        long_parts.append(m)
        wafer_sets.append(set(zip(t["day"], t["wafer"])))
    # only wafers that have data from *every* requested source (no zero-filled sensors)
    common = set.intersection(*wafer_sets) if wafer_sets else set()
    long = pd.concat(long_parts, ignore_index=True)
    long = long[[(d, w) in common for d, w in zip(long["day"], long["wafer"])]]
    long["sensor"] = long["phase"] + ":" + long["signal"]
    # drop constant sensors
    sd = long.groupby("sensor")["value"].agg(lambda v: np.nanstd(v) / (np.nanmean(np.abs(v)) + 1e-12))
    sensors = sorted(sd[sd > min_rel_std].index)
    long = long[long["sensor"].isin(sensors)]
    long["win"] = _window_index(long["cycle"].to_numpy(), n_windows, max_cycle)
    sidx = {s: i for i, s in enumerate(sensors)}
    ids, xs = [], []
    for (day, wafer), g in long.groupby(["day", "wafer"]):
        x = np.zeros((n_windows, len(sensors), len(STAT_NAMES)), dtype=np.float32)
        for (s, w), gg in g.groupby(["sensor", "win"]):
            y = gg.sort_values("cycle")["value"].to_numpy(np.float64)
            y = y[np.isfinite(y)]
            if y.size == 0:
                continue
            x[w, sidx[s]] = [y.mean(), y.max(), y.min(), y.var(), np.ptp(y), _slope(y)]
        xs.append(x.reshape(n_windows * len(sensors), len(STAT_NAMES)))
        ids.append(wafer_id(day, wafer))
    return np.stack(xs), ids, sensors


class TokenScaler:
    """Per-(token, stat) standardisation fitted on training wafers only."""

    def fit(self, x: np.ndarray):
        self.mu_ = x.mean(axis=0, keepdims=True)
        self.sd_ = x.std(axis=0, keepdims=True) + 1e-6
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mu_) / self.sd_).astype(np.float32)


def positional_check(n_sensors: int = 3, n_windows: int = 2, d: int = 6) -> bool:
    """All sensors of a window share one encoding vector (used in tests)."""
    pe = window_positional_encoding(n_sensors, n_windows, d)
    same = torch.allclose(pe[0], pe[n_sensors - 1])
    diff = not torch.allclose(pe[0], pe[n_sensors])
    return bool(same and diff and math.isfinite(float(pe.sum())))
