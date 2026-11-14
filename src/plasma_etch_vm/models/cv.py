"""Group-aware cross-validation.

Wafers of one lot share a day, a conditioning recipe and a chamber state, so random K-fold would
leak lot-level information into the test fold. Default: leave-one-lot-out (LOLO, 10 folds on the
full dataset). With fewer groups than ``min_groups`` (the 1-day smoke test has one lot) we fall back
to leave-one-wafer-out and flag the result as *not* leakage-safe.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, LeaveOneOut


def make_splits(
    groups: np.ndarray, scheme: str = "lolo", n_splits: int = 5, min_groups: int = 3
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Return ``(splits, scheme_used)``.

    ``scheme``: ``lolo`` (leave-one-group-out) | ``group_kfold`` | ``loo``.
    """
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    idx = np.arange(len(groups))
    if scheme != "loo" and n_groups < min_groups:
        scheme = "loo"
    if scheme == "lolo":
        splitter = LeaveOneGroupOut()
        splits = list(splitter.split(idx, groups=groups))
    elif scheme == "group_kfold":
        splitter = GroupKFold(n_splits=min(n_splits, n_groups))
        splits = list(splitter.split(idx, groups=groups))
    elif scheme == "loo":
        splits = list(LeaveOneOut().split(idx))
    else:
        raise ValueError(f"unknown CV scheme {scheme}")
    return splits, scheme


def assert_no_group_leak(splits, groups: np.ndarray) -> None:
    groups = np.asarray(groups)
    for tr, te in splits:
        overlap = set(groups[tr]) & set(groups[te])
        if overlap:
            raise AssertionError(f"group leakage between train and test: {sorted(overlap)[:5]}")


def iter_inner_calibration(
    train_idx: np.ndarray, groups: np.ndarray, cal_frac: float = 0.25, seed: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Split a training fold into proper-train / calibration *by group* (for split conformal)."""
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)[train_idx]
    ug = np.unique(g)
    if len(ug) >= 4:
        n_cal = max(1, round(cal_frac * len(ug)))
        cal_groups = rng.choice(ug, size=n_cal, replace=False)
        cal_mask = np.isin(g, cal_groups)
    else:
        cal_mask = np.zeros(len(train_idx), dtype=bool)
        cal_mask[rng.choice(len(train_idx), size=max(1, round(cal_frac * len(train_idx))), replace=False)] = True
    yield train_idx[~cal_mask], train_idx[cal_mask]
