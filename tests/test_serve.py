"""API and dashboard figure builders (CPU only)."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from plasma_etch_vm.models.tabular import build_model


@pytest.fixture()
def bundle_path(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    cols = ["T_etch_Pressure_mean", "C_slot"]
    X = rng.normal(size=(30, 2))
    y = 44 + X[:, 0]
    bundle = {
        "feature_columns": cols,
        "feature_set": "tool_context",
        "model_name": "ridge",
        "models": {"si_etch_mean": build_model("ridge").fit(X, y)},
        "conformal_halfwidth": {"si_etch_mean": {"0.1": 0.25, "0.2": 0.2}},
        "train_wafers": ["2024-07-02_01"],
        "train_feature_medians": dict.fromkeys(cols, 0.0),
        "train_feature_min": dict.fromkeys(cols, -3.0),
        "train_feature_max": dict.fromkeys(cols, 3.0),
        "target_units": {"si_etch_mean": "µm"},
        "created": "test",
    }
    p = tmp_path / "bundle.joblib"
    joblib.dump(bundle, p)
    store = pd.DataFrame({"T_etch_Pressure_mean": [0.5], "C_slot": [1.0]}, index=["2024-07-02_01"])
    store.to_parquet(tmp_path / "store.parquet")
    monkeypatch.setenv("PEVM_BUNDLE", str(p))
    monkeypatch.setenv("PEVM_FEATURE_STORE", str(tmp_path / "store.parquet"))
    from plasma_etch_vm.serve import api

    api.load_bundle.cache_clear()
    api.load_store.cache_clear()
    yield p
    api.load_bundle.cache_clear()
    api.load_store.cache_clear()


def test_api_predict(bundle_path):
    from fastapi.testclient import TestClient

    from plasma_etch_vm.serve.api import app

    c = TestClient(app)
    assert c.get("/health").json()["model_loaded"]
    assert c.get("/model").json()["n_features"] == 2
    r = c.post("/predict", json={"features": {"T_etch_Pressure_mean": 1.0, "C_slot": 2.0}, "alpha": 0.1}).json()
    p = r["predictions"]["si_etch_mean"]
    assert p["upper"] - p["lower"] == pytest.approx(0.5) and p["unit"] == "µm"
    r2 = c.post("/predict", json={"features": {"T_etch_Pressure_mean": 10.0}}).json()
    assert r2["n_features_missing"] == 1 and r2["n_features_out_of_range"] == 1 and len(r2["warnings"]) == 2
    r3 = c.get("/predict/2024-07-02_01").json()
    assert any("training set" in w for w in r3["warnings"])
    assert c.get("/predict/unknown").status_code == 404


def test_figure_builders():
    from plasma_etch_vm.dashboard import figures as F

    d = pd.DataFrame(
        {
            "wafer_id": ["2024-07-02_01", "2024-07-02_05"],
            "lot": [1, 1],
            "y": [44.0, 43.5],
            "pred": [43.9, 43.6],
            "lo": [43.6, 43.3],
            "hi": [44.2, 43.9],
            "fold": [0, 0],
        }
    )
    fig = F.pred_vs_measured(d, "si_etch_mean", static=True)
    assert len(fig.data) == 2 and fig.layout.margin.t > 40
    x = np.linspace(500, 790, 100)
    f2 = F.spectrum(x, {"etch": np.ones(100), "dep": np.ones(100)}, [("F_685", 685.6)], (500, 790))
    assert len(f2.layout.annotations) == 1


def test_api_v2_hybrid_cvplus(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from plasma_etch_vm.serve import api

    rng = np.random.default_rng(1)
    cols = ["TM_tool_etch_n", "C_slot"]
    X = np.c_[rng.choice([99.0, 100.0], 40), rng.integers(1, 11, 40)]
    rate = 0.44 - 0.001 * X[:, 1]
    full = build_model("ridge").fit(X, rate)
    folds = [build_model("ridge").fit(X[i::2], rate[i::2]) for i in range(2)]
    bundle = {
        "kind": "hybrid_cvplus",
        "target": "si_etch_mean",
        "feature_columns": cols,
        "cycles_column": "TM_tool_etch_n",
        "full_model": full,
        "fold_models": folds,
        "fold_of": np.arange(40) % 2,
        "residuals": np.full(40, 0.1),
        "model_name": "hybrid",
        "train_wafers": [],
        "train_feature_medians": dict.fromkeys(cols, 1.0),
        "note": "v2",
        "created": "t",
    }
    joblib.dump(bundle, tmp_path / "v2.joblib")
    monkeypatch.setenv("PEVM_BUNDLE", str(tmp_path / "v2.joblib"))
    api.load_bundle.cache_clear()
    r = TestClient(api.app).post("/predict", json={"features": {"TM_tool_etch_n": 100.0, "C_slot": 5.0}}).json()
    api.load_bundle.cache_clear()
    p = r["predictions"]["si_etch_mean"]
    assert p["lower"] < p["prediction"] < p["upper"] and abs(p["prediction"] - 100 * (0.44 - 0.005)) < 0.01
