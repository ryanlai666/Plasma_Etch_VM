"""Cross-validated VM benchmark: feature sets × models × targets, group-aware CV, conformal intervals.

For each outer fold (default leave-one-lot-out):

1. the training lots are split *by lot* into proper-train / calibration,
2. the model is fit on proper-train, absolute residuals on the calibration lots give the
   split-conformal quantile q, and the test lot gets ``ŷ ± q`` (split conformal, honest),
3. the model is re-fit on the full training fold for the point prediction (reported metrics).

Everything is logged to MLflow (``tracking.uri``, default ``file:./mlruns``).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .conformal import coverage, mean_width, split_conformal_interval
from .cv import assert_no_group_leak, iter_inner_calibration, make_splits
from .tabular import build_model

log = logging.getLogger(__name__)


@dataclass
class FoldResult:
    oof: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    fold: np.ndarray
    fit_seconds: float


def regression_metrics(y: np.ndarray, pred: np.ndarray, y_train_mean: np.ndarray | None = None) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    out = {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)) if y.size > 1 else float("nan"),
        "mape_pct": float(np.mean(np.abs((y - pred) / np.where(y == 0, np.nan, y)))) * 100,
        "n": int(y.size),
    }
    if y_train_mean is not None:
        base = np.asarray(y_train_mean, float)[ok]
        rmse_base = float(np.sqrt(mean_squared_error(y, base)))
        out["rmse_vs_mean_baseline"] = out["rmse"] / rmse_base if rmse_base > 0 else float("nan")
    return out


def cross_validate(
    model, X: np.ndarray, y: np.ndarray, groups: np.ndarray, splits, alpha: float = 0.1, seed: int = 0
) -> FoldResult:
    n = len(y)
    oof = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    fold = np.full(n, -1)
    t0 = time.perf_counter()
    for k, (tr, te) in enumerate(splits):
        tr = tr[np.isfinite(y[tr])]
        for ptr, cal in iter_inner_calibration(tr, groups, seed=seed + k):
            m_cal = clone(model).fit(X[ptr], y[ptr])
            lo_c, hi_c, _ = split_conformal_interval(
                y[cal], m_cal.predict(X[cal]).ravel(), m_cal.predict(X[te]).ravel(), alpha
            )
        m = clone(model).fit(X[tr], y[tr])
        p = np.asarray(m.predict(X[te])).ravel()
        oof[te] = p
        # centre the conformal band on the full-fold prediction, keep its half-width
        half = (hi_c - lo_c) / 2
        lo[te], hi[te] = p - half, p + half
        fold[te] = k
    return FoldResult(oof, lo, hi, fold, time.perf_counter() - t0)


def _train_mean_baseline(y: np.ndarray, splits) -> np.ndarray:
    base = np.full(len(y), np.nan)
    for tr, te in splits:
        base[te] = np.nanmean(y[tr])
    return base


def run_benchmark(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
    feature_columns: dict[str, list[str]],
    mlflow_run: bool = True,
) -> pd.DataFrame:
    """``features`` / ``targets`` indexed by wafer_id; ``feature_columns``: set name -> column list."""
    out_dir.mkdir(parents=True, exist_ok=True)
    common = features.index.intersection(targets.index)
    features, targets = features.loc[common], targets.loc[common]
    groups = targets["lot"].to_numpy()
    splits, scheme = make_splits(
        groups, cfg["cv"]["scheme"], cfg["cv"].get("n_splits", 5), cfg["cv"].get("min_groups", 3)
    )
    if scheme != "loo":
        assert_no_group_leak(splits, groups)
    alpha = cfg["conformal"]["alpha"]
    seed = cfg.get("seed", 0)

    mlflow = None
    if mlflow_run:
        from ..tracking import start_run

        mlflow = start_run(cfg, run_name=cfg.get("run_name", "tabular_benchmark"))

    rows, oof_tables = [], []
    for fs_name, cols in feature_columns.items():
        X = features[cols].to_numpy(np.float64)
        for model_name in cfg["models"]:
            if model_name not in cfg.get("model_feature_sets", {}).get(fs_name, cfg["models"]):
                continue
            model = build_model(model_name, cfg.get("model_params", {}).get(model_name), seed)
            for tgt in cfg["targets"]:
                y = targets[tgt].to_numpy(np.float64)
                res = cross_validate(model, X, y, groups, splits, alpha, seed)
                met = regression_metrics(y, res.oof, _train_mean_baseline(y, splits))
                met.update(
                    coverage=coverage(y, res.lo, res.hi),
                    interval_width=mean_width(res.lo, res.hi),
                    fit_seconds=res.fit_seconds,
                )
                row = {
                    "feature_set": fs_name,
                    "model": model_name,
                    "target": tgt,
                    "n_features": X.shape[1],
                    "cv": scheme,
                    "n_folds": len(splits),
                    **met,
                }
                rows.append(row)
                oof_tables.append(
                    pd.DataFrame(
                        {
                            "wafer_id": common,
                            "lot": groups,
                            "feature_set": fs_name,
                            "model": model_name,
                            "target": tgt,
                            "y": y,
                            "pred": res.oof,
                            "lo": res.lo,
                            "hi": res.hi,
                            "fold": res.fold,
                        }
                    )
                )
                log.info(
                    "%-10s %-9s %-22s rmse=%.4g r2=%.3f cov=%.2f (%.1fs)",
                    fs_name,
                    model_name,
                    tgt,
                    met["rmse"],
                    met["r2"],
                    met["coverage"],
                    res.fit_seconds,
                )
                if mlflow is not None:
                    key = f"{fs_name}.{model_name}.{tgt}"
                    mlflow.log_metrics(
                        {
                            f"{key}.{k}": float(v)
                            for k, v in met.items()
                            if isinstance(v, (int, float)) and np.isfinite(v)
                        }
                    )
    results = pd.DataFrame(rows)
    oof = pd.concat(oof_tables, ignore_index=True) if oof_tables else pd.DataFrame()
    results.to_csv(out_dir / "benchmark_results.csv", index=False)
    oof.to_parquet(out_dir / "oof_predictions.parquet", index=False)
    (out_dir / "benchmark_meta.json").write_text(
        json.dumps(
            {
                "cv": scheme,
                "n_folds": len(splits),
                "n_wafers": len(common),
                "n_lots": len(np.unique(groups)),
                "alpha": alpha,
                "feature_sets": {k: len(v) for k, v in feature_columns.items()},
            },
            indent=2,
        )
    )
    if mlflow is not None:
        mlflow.log_params({"cv": scheme, "n_wafers": len(common), "n_lots": len(np.unique(groups)), "alpha": alpha})
        mlflow.log_artifact(str(out_dir / "benchmark_results.csv"))
        mlflow.log_artifact(str(out_dir / "oof_predictions.parquet"))
        mlflow.end_run()
    return results
