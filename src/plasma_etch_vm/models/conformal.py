"""Conformal prediction intervals.

* :func:`split_conformal_quantile` – finite-sample split-conformal quantile of absolute residuals
  (Vovk et al.; Lei et al. 2018): q = the ceil((n+1)(1-α))-th smallest calibration score.
* :class:`NormalizedConformal` – locally adaptive variant with scores |y-ŷ|/σ̂(x).
* :func:`cv_plus_interval` – CV+ (Barber et al. 2021, "jackknife+") intervals using out-of-fold
  residuals and the fold models' test predictions; uses all data for both fitting and calibration,
  which matters with ~90 wafers.

Coverage guarantees assume exchangeability of calibration and test wafers; with lot-wise drift this
holds only approximately, so empirical coverage under leave-one-lot-out is always reported.
"""

from __future__ import annotations

import math

import numpy as np


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    n = scores.size
    if n == 0:
        return float("inf")
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")  # not enough calibration points for this alpha
    return float(scores[k - 1])


def split_conformal_interval(
    y_cal: np.ndarray, pred_cal: np.ndarray, pred_test: np.ndarray, alpha: float = 0.1
) -> tuple[np.ndarray, np.ndarray, float]:
    q = split_conformal_quantile(np.abs(np.asarray(y_cal) - np.asarray(pred_cal)), alpha)
    pred_test = np.asarray(pred_test, dtype=np.float64)
    return pred_test - q, pred_test + q, q


class NormalizedConformal:
    """Scores ``|y - ŷ| / (σ̂ + eps)``; σ̂ from any difficulty estimate (e.g. a residual model)."""

    def __init__(self, alpha: float = 0.1, eps: float = 1e-6):
        self.alpha = alpha
        self.eps = eps
        self.q_: float | None = None

    def fit(self, y_cal, pred_cal, sigma_cal):
        s = np.abs(np.asarray(y_cal) - np.asarray(pred_cal)) / (np.asarray(sigma_cal) + self.eps)
        self.q_ = split_conformal_quantile(s, self.alpha)
        return self

    def interval(self, pred, sigma):
        half = self.q_ * (np.asarray(sigma) + self.eps)
        return np.asarray(pred) - half, np.asarray(pred) + half


def cv_plus_interval(
    oof_residuals: np.ndarray, fold_of: np.ndarray, test_preds_by_fold: np.ndarray, alpha: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """CV+ interval for new points.

    ``oof_residuals[i]`` = |y_i − ŷ_{-fold(i)}(x_i)|, ``fold_of[i]`` = fold of training point i,
    ``test_preds_by_fold[k, j]`` = prediction of fold-k model for test point j.
    Lower = ⌊α(n+1)⌋-th smallest of ŷ_{-k(i)}(x) − R_i, upper = ⌈(1−α)(n+1)⌉-th smallest of ŷ + R_i.
    """
    r = np.asarray(oof_residuals, dtype=np.float64)
    n = r.size
    preds = np.asarray(test_preds_by_fold, dtype=np.float64)[np.asarray(fold_of)]  # (n, n_test)
    lo_vals = np.sort(preds - r[:, None], axis=0)
    hi_vals = np.sort(preds + r[:, None], axis=0)
    k_lo = math.floor(alpha * (n + 1))
    k_hi = math.ceil((1 - alpha) * (n + 1))
    lo = lo_vals[k_lo - 1] if k_lo >= 1 else np.full(preds.shape[1], -np.inf)
    hi = hi_vals[k_hi - 1] if k_hi <= n else np.full(preds.shape[1], np.inf)
    return lo, hi


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    w = np.asarray(hi) - np.asarray(lo)
    return float(np.mean(w[np.isfinite(w)])) if np.isfinite(w).any() else float("inf")
