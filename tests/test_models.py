"""Models: group CV, conformal intervals, tabular factory, benchmark, NN shapes + checkpoint/resume."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plasma_etch_vm.models.conformal import coverage, cv_plus_interval, split_conformal_quantile
from plasma_etch_vm.models.cv import assert_no_group_leak, iter_inner_calibration, make_splits
from plasma_etch_vm.models.tabular import build_model


def synthetic_table(n_lots=6, per_lot=8, p=20, seed=0):
    rng = np.random.default_rng(seed)
    n = n_lots * per_lot
    X = rng.normal(size=(n, p))
    lot = np.repeat(np.arange(1, n_lots + 1), per_lot)
    slot = np.tile(np.arange(1, per_lot + 1), n_lots)
    y = 44 - 0.1 * slot + 0.3 * X[:, 0] + rng.normal(0, 0.05, n)
    ids = [f"2024-07-{lt:02d}_{s:02d}" for lt, s in zip(lot, slot)]
    feats = pd.DataFrame(X, columns=[f"T_etch_p{i}_mean" for i in range(p)], index=ids)
    feats["C_slot"] = slot.astype(float)
    tg = pd.DataFrame({"lot": lot, "si_etch_mean": y, "ox_etch_mean": 0.6 + 0.01 * X[:, 1]}, index=ids)
    return feats, tg


def test_lolo_splits_have_no_leakage():
    groups = np.repeat([1, 2, 3, 4], 5)
    splits, scheme = make_splits(groups, "lolo")
    assert scheme == "lolo" and len(splits) == 4
    assert_no_group_leak(splits, groups)
    _, scheme = make_splits(np.ones(6), "lolo")
    assert scheme == "loo"  # single lot -> flagged fallback
    many = np.repeat([1, 2, 3, 4, 5, 6], 3)  # >= 4 lots: calibration split is by lot
    tr, cal = next(iter_inner_calibration(np.arange(18), many))
    assert len(cal) and not set(many[tr]) & set(many[cal])
    few = np.repeat([1, 2, 3], 4)  # < 4 lots: documented fallback to a wafer-level split
    tr, cal = next(iter_inner_calibration(np.arange(12), few))
    assert len(tr) + len(cal) == 12 and not set(tr) & set(cal)


def test_split_conformal_quantile_finite_sample():
    s = np.arange(1, 20, dtype=float)  # n = 19
    assert split_conformal_quantile(s, 0.1) == 18.0  # ceil(20*0.9) = 18th smallest
    assert np.isinf(split_conformal_quantile(s[:5], 0.1))


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(200):
        cal, test = rng.normal(size=100), rng.normal(size=50)
        q = split_conformal_quantile(np.abs(cal), 0.1)
        covs.append(coverage(test, -q * np.ones(50), q * np.ones(50)))
    assert 0.87 < np.mean(covs) < 0.95


def test_cv_plus_shapes():
    lo, hi = cv_plus_interval(np.ones(10), np.repeat([0, 1], 5), np.zeros((2, 3)), alpha=0.2)
    assert lo.shape == (3,) and np.all(hi >= lo)


@pytest.mark.parametrize("name", ["mean", "ridge", "pls", "gbm", "pca_ridge", "nmf_ridge"])
def test_model_factory_fit_predict(name):
    feats, tg = synthetic_table()
    X = feats.to_numpy()
    if name == "nmf_ridge":
        X = np.abs(X)
    m = build_model(name, {"n_estimators": 20} if name == "gbm" else None).fit(X[:30], tg["si_etch_mean"].iloc[:30])
    assert np.asarray(m.predict(X[30:])).ravel().shape == (len(X) - 30,)


def test_benchmark_runs_and_beats_mean(tmp_path):
    from plasma_etch_vm.models.benchmark import run_benchmark

    feats, tg = synthetic_table()
    cfg = {
        "cv": {"scheme": "lolo", "min_groups": 3},
        "conformal": {"alpha": 0.2},
        "seed": 0,
        "models": ["mean", "ridge"],
        "targets": ["si_etch_mean"],
    }
    res = run_benchmark(feats, tg, cfg, tmp_path, {"all": list(feats.columns)}, mlflow_run=False)
    r = res.set_index("model")
    assert r.loc["ridge", "rmse"] < 0.5 * r.loc["mean", "rmse"]
    assert (tmp_path / "oof_predictions.parquet").exists()


def test_statfe_positional_encoding_and_forward():
    torch = pytest.importorskip("torch")
    from plasma_etch_vm.models.nn import SpectralCNN1D, StatFETransformer, positional_check

    assert positional_check()
    m = StatFETransformer(n_sensors=5, n_windows=4, n_stats=6, n_targets=2)
    assert m(torch.randn(3, 20, 6)).shape == (3, 2)
    m2 = StatFETransformer(n_sensors=5, n_windows=4, n_stats=6, n_targets=2, d_model=16, n_heads=2)
    assert m2(torch.randn(3, 20, 6)).shape == (3, 2)
    c = SpectralCNN1D(in_channels=2, n_targets=3, width=8, depth=2)
    assert c(torch.randn(4, 2, 64)).shape == (4, 3)


def test_statfe_tokens_from_cycle_tables():
    pytest.importorskip("torch")
    from plasma_etch_vm.models.nn import statfe_tokens

    rows = []
    for w in (1, 2):
        for ph in ("etch", "dep"):
            for k in range(1, 21):
                rows.append(
                    {
                        "day": "2024-08-05",
                        "wafer": w,
                        "phase": ph,
                        "cycle": k,
                        "P_Pressure": 0.04 + 0.001 * k + w,
                        "P_const": 1.0,
                    }
                )
    X, ids, sensors = statfe_tokens([pd.DataFrame(rows)], n_windows=4, max_cycle=20)
    assert X.shape == (2, 4 * len(sensors), 6) and "etch:P_Pressure" in sensors and "etch:P_const" not in sensors
    assert ids == ["2024-08-05_01", "2024-08-05_02"]


def test_train_cv_checkpoint_resume(tmp_path):
    pytest.importorskip("torch")
    from plasma_etch_vm.models.train_nn import train_cv

    rng = np.random.default_rng(0)
    n_wafers = 12
    ids = [f"2024-07-{1 + i // 4:02d}_{1 + i % 4:02d}" for i in range(n_wafers)]
    X = rng.normal(size=(n_wafers, 3 * 2, 6)).astype(np.float32)
    tg = pd.DataFrame(
        {"lot": [1 + i // 4 for i in range(n_wafers)], "si_etch_mean": rng.normal(44, 0.3, n_wafers)}, index=ids
    )
    cfg = {
        "model": {"type": "statfe", "n_windows": 2, "n_heads": 1, "n_layers": 1, "ff_dim": 8},
        "targets": ["si_etch_mean"],
        "cv": {"scheme": "lolo", "min_groups": 3},
        "train": {"epochs": 3, "batch_size": 4, "lr": 1e-3, "device": "cpu"},
        "checkpoint": {"dir": str(tmp_path / "ck"), "every_minutes": 20},
        "run_name": "t",
        "seed": 0,
    }
    out = train_cv(cfg, X, ids, {"n_sensors": 3}, tg, tmp_path / "run")
    assert out["stats"]["n_folds"] == 3 and (tmp_path / "ck" / "latest.pt").exists()
    first = pd.read_parquet(tmp_path / "run" / "nn_oof_predictions.parquet")
    # resume: finished folds are read back, predictions identical
    out2 = train_cv(cfg, X, ids, {"n_sensors": 3}, tg, tmp_path / "run")
    second = pd.read_parquet(tmp_path / "run" / "nn_oof_predictions.parquet")
    np.testing.assert_allclose(first["pred"], second["pred"])
    assert out2["results"]["n"].iloc[0] == n_wafers
