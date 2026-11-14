"""Virtual-metrology prediction API.

Run::

    pevm export                                    # writes outputs/model_bundle.joblib
    uvicorn plasma_etch_vm.serve.api:app --port 8000

Environment:
    PEVM_BUNDLE         path to the joblib bundle (default outputs/model_bundle.joblib = pre-declared ridge;
                        outputs/model_bundle_v2.joblib = v2 candidate hybrid with CV+ intervals, Si depth only)
    PEVM_FEATURE_STORE  wafer feature table for ``GET /predict/{wafer_id}``
                        (default data/features/wafer_features.parquet; optional)

Endpoints:
    GET  /health                 liveness + whether a model is loaded
    GET  /model                  model card (feature set, targets, conformal half-widths)
    POST /predict                predict from a feature dict (missing features -> training median)
    GET  /predict/{wafer_id}     predict a wafer from the local feature store (+ measured values)

Intervals: ŷ ± q_α, with q_α the split-conformal quantile of leave-one-lot-out out-of-fold absolute
residuals (approximately (1−α) coverage for a new wafer from a new lot).
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ..config import project_path

app = FastAPI(
    title="Plasma-etch virtual metrology",
    version="0.1.0",
    description="BOSCH etch depth / uniformity / selectivity from tool traces with conformal intervals",
)


class PredictRequest(BaseModel):
    features: dict[str, float | None] = Field(..., description="feature name -> value (see GET /model)")
    alpha: float = Field(0.1, description="miscoverage level; available: see /model conformal_halfwidth")


class TargetPrediction(BaseModel):
    prediction: float
    lower: float | None
    upper: float | None
    unit: str


class PredictResponse(BaseModel):
    predictions: dict[str, TargetPrediction]
    alpha: float
    n_features_missing: int
    n_features_out_of_range: int
    warnings: list[str]
    latency_ms: float
    measured: dict[str, float] | None = None


@lru_cache(maxsize=1)
def load_bundle(path: str | None = None) -> dict[str, Any]:
    import joblib

    p = Path(path or os.environ.get("PEVM_BUNDLE", project_path("outputs/model_bundle.joblib")))
    if not p.exists():
        raise FileNotFoundError(f"model bundle not found at {p}; run `pevm export` first")
    return joblib.load(p)


@lru_cache(maxsize=1)
def load_store() -> pd.DataFrame | None:
    p = Path(os.environ.get("PEVM_FEATURE_STORE", project_path("data/features/wafer_features.parquet")))
    return pd.read_parquet(p) if p.exists() else None


@lru_cache(maxsize=1)
def load_targets() -> pd.DataFrame | None:
    p = project_path("data/features/wafer_targets.parquet")
    return pd.read_parquet(p) if p.exists() else None


def predict_features(bundle: dict, features: dict[str, float | None], alpha: float = 0.1) -> dict:
    t0 = time.perf_counter()
    cols = bundle["feature_columns"]
    med = bundle["train_feature_medians"]
    x, missing, oor = [], 0, 0
    lo_r, hi_r = bundle.get("train_feature_min", {}), bundle.get("train_feature_max", {})
    for c in cols:
        v = features.get(c)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            missing += 1
            v = med.get(c, np.nan)
        elif c in lo_r and (v < lo_r[c] or v > hi_r[c]):
            oor += 1
        x.append(v)
    X = np.asarray(x, dtype=np.float64)[None, :]
    preds = {}
    if bundle.get("kind") == "hybrid_cvplus":  # v2 candidate: depth = n_cycles x rate, CV+ interval
        from ..models.conformal import cv_plus_interval

        ncyc = X[0, cols.index(bundle["cycles_column"])]
        p = float(np.asarray(bundle["full_model"].predict(X)).ravel()[0] * ncyc)
        by_fold = np.array([[float(np.asarray(m.predict(X)).ravel()[0] * ncyc)] for m in bundle["fold_models"]])
        lo, hi = cv_plus_interval(bundle["residuals"], bundle["fold_of"], by_fold, alpha)
        preds[bundle["target"]] = {"prediction": p, "lower": float(lo[0]), "upper": float(hi[0]), "unit": "µm"}
        bundle = {**bundle, "models": {}, "conformal_halfwidth": {bundle["target"]: {str(alpha): 0.0}}}
    for tgt, model in bundle["models"].items():
        p = float(np.asarray(model.predict(X)).ravel()[0])
        q = bundle.get("conformal_halfwidth", {}).get(tgt, {}).get(str(alpha))
        preds[tgt] = {
            "prediction": p,
            "lower": p - q if q is not None and np.isfinite(q) else None,
            "upper": p + q if q is not None and np.isfinite(q) else None,
            "unit": bundle.get("target_units", {}).get(tgt, ""),
        }
    warnings = []
    if missing:
        warnings.append(f"{missing} of {len(cols)} features missing; imputed with training medians")
    if oor:
        warnings.append(f"{oor} features outside the training range; intervals may under-cover")
    if str(alpha) not in next(iter(bundle.get("conformal_halfwidth", {"": {}}).values()), {}):
        warnings.append(f"no conformal half-width stored for alpha={alpha}; intervals omitted")
    return {
        "predictions": preds,
        "alpha": alpha,
        "n_features_missing": missing,
        "n_features_out_of_range": oor,
        "warnings": warnings,
        "latency_ms": (time.perf_counter() - t0) * 1e3,
    }


@app.get("/health")
def health() -> dict:
    try:
        b = load_bundle()
        return {"status": "ok", "model_loaded": True, "targets": [b["target"]] if b.get("kind") else list(b["models"])}
    except FileNotFoundError as exc:
        return {"status": "degraded", "model_loaded": False, "detail": str(exc)}


@app.get("/model")
def model_card() -> dict:
    try:
        b = load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    if b.get("kind") == "hybrid_cvplus":
        return {
            "kind": b["kind"],
            "model": b["model_name"],
            "targets": [b["target"]],
            "note": b["note"],
            "n_features": len(b["feature_columns"]),
            "feature_columns": b["feature_columns"],
            "n_train_wafers": len(b["train_wafers"]),
            "created": b["created"],
        }
    return {
        "feature_set": b["feature_set"],
        "model": b["model_name"],
        "targets": list(b["models"]),
        "target_units": b.get("target_units", {}),
        "n_features": len(b["feature_columns"]),
        "feature_columns": b["feature_columns"],
        "conformal_halfwidth": b.get("conformal_halfwidth", {}),
        "n_train_wafers": len(b.get("train_wafers", [])),
        "created": b.get("created"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> dict:
    try:
        b = load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    return predict_features(b, req.features, req.alpha)


@app.get("/predict/{wafer_id}", response_model=PredictResponse)
def predict_wafer(wafer_id: str, alpha: float = Query(0.1)) -> dict:
    try:
        b = load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    store = load_store()
    if store is None or wafer_id not in store.index:
        raise HTTPException(404, f"wafer {wafer_id} not in feature store")
    row = store.loc[wafer_id]
    feats = {c: (None if pd.isna(row.get(c)) else float(row.get(c))) for c in b["feature_columns"]}
    out = predict_features(b, feats, alpha)
    tg = load_targets()
    if tg is not None and wafer_id in tg.index:
        tnames = [b["target"]] if b.get("kind") else list(b["models"])
        out["measured"] = {t: float(tg.loc[wafer_id, t]) for t in tnames if t in tg.columns}
    if wafer_id in set(b.get("train_wafers", [])):
        out["warnings"].append("wafer was in the training set (in-sample prediction)")
    return out
