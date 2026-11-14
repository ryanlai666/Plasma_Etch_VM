"""High-level pipeline steps used by the CLI, the API and the dashboard."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .features.cycles import wafer_partitions
from .features.transitions import transition_features
from .features.wafer import (
    GROUP_PREFIXES,
    context_features,
    oes_features,
    select_groups,
    spectrum_features,
    tool_features,
)
from .metrology import TARGET_COLUMNS, load_lot_status, load_points, wafer_targets

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------- features
def load_targets(cfg: dict, paths: dict, table: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(wafer_targets, points)`` for the configured metrology table (89 or 9 points)."""
    key = table or cfg.get("metrology", {}).get("points", "points_89")
    pts = load_points(paths["raw"] / cfg["files"][key])
    return wafer_targets(pts).set_index("wafer_id"), pts


def build_feature_table(cfg: dict, paths: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    feat_dir = paths["features"]
    fcfg = cfg["features"]
    t0 = time.perf_counter()
    lot_status = load_lot_status(paths["raw"] / cfg["files"]["lot_status"])
    parts = []
    cp = feat_dir / "cycles_process.parquet"
    if cp.exists():
        parts.append(tool_features(pd.read_parquet(cp)))
    co = feat_dir / "cycles_oes.parquet"
    if co.exists():
        parts.append(oes_features(pd.read_parquet(co), fcfg["lines"], fcfg.get("ratios", {})))
        parts.append(spectrum_features(pd.read_parquet(feat_dir / "wafer_spectra.parquet"), fcfg["bin_width"]))
        if fcfg.get("transitions", True):
            traces = {
                (day, wafer): pd.read_parquet(d / "part-0.parquet")
                for day, _, wafer, d in wafer_partitions(paths["parquet"], "oes_trace")
            }
            parts.append(transition_features(traces))
    feats = pd.concat(parts, axis=1)
    feats = feats.join(context_features(list(feats.index), lot_status))
    feats.index.name = "wafer_id"
    targets, _ = load_targets(cfg, paths)
    groups = {g: select_groups(list(feats.columns), [g]) for g in GROUP_PREFIXES}
    feat_dir.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(feat_dir / "wafer_features.parquet")
    targets.to_parquet(feat_dir / "wafer_targets.parquet")
    (feat_dir / "feature_groups.json").write_text(json.dumps(groups, indent=1))
    log.info("feature table %s built in %.1fs", feats.shape, time.perf_counter() - t0)
    return feats, targets, groups


def load_feature_table(paths: dict):
    fd = paths["features"]
    feats = pd.read_parquet(fd / "wafer_features.parquet")
    targets = pd.read_parquet(fd / "wafer_targets.parquet")
    groups = json.loads((fd / "feature_groups.json").read_text())
    return feats, targets, groups


def feature_set_columns(feats: pd.DataFrame, groups: dict, sets: dict[str, list[str]]) -> dict[str, list[str]]:
    return {name: [c for g in gs for c in groups.get(g, [])] for name, gs in sets.items()}


# ------------------------------------------------------------------------------------ benchmark
def run_tabular_benchmark(cfg: dict, paths: dict, mlflow_run: bool = True) -> pd.DataFrame:
    """Benchmark each feature set on the wafers where that set is available (≥1 non-NaN feature)."""
    from .models.benchmark import run_benchmark

    feats, targets, groups = load_feature_table(paths)
    sets = feature_set_columns(feats, groups, cfg["feature_sets"])
    out_root = paths["outputs"] / "benchmark"
    all_res = []
    for name, cols in sets.items():
        if not cols:
            log.warning("feature set %s is empty (no data for its groups yet) - skipped", name)
            continue
        have = feats[cols].notna().any(axis=1)
        f = feats.loc[have, cols]
        sub_cfg = {**cfg, "run_name": f"{cfg.get('run_name', 'tabular')}_{name}"}
        res = run_benchmark(f, targets, sub_cfg, out_root / name, {name: cols}, mlflow_run=mlflow_run)
        res["n_wafers"] = int(f.index.intersection(targets.index).size)
        all_res.append(res)
    results = pd.concat(all_res, ignore_index=True)
    results.to_csv(out_root / "benchmark_all.csv", index=False)
    write_benchmark_markdown(results, out_root / "benchmark_all.md")
    return results


def write_benchmark_markdown(results: pd.DataFrame, path: Path) -> None:
    cols = [
        "feature_set",
        "model",
        "target",
        "n_wafers",
        "cv",
        "rmse",
        "mae",
        "r2",
        "rmse_vs_mean_baseline",
        "coverage",
        "interval_width",
    ]
    cols = [c for c in cols if c in results]
    df = results[cols].copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: f"{v:.4g}")
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------- point-level model
def run_point_benchmark(cfg: dict, paths: dict) -> pd.DataFrame:
    """Per-site Si etch depth (89 points): wafer-level prediction × learned radial profile.

    * ``site_mean``      training-set mean depth at each (x, y) site (no wafer information)
    * ``ridge_x_shape``  ridge prediction of the wafer mean × training mean shape (depth / wafer mean)
                         at each site — separates rate (chamber state) from the radial loading profile
    * ``gbm_point``      LightGBM on [wafer features + x, y, r, r²] directly
    """
    from sklearn.base import clone

    from .models.benchmark import regression_metrics
    from .models.cv import make_splits
    from .models.tabular import build_model

    feats, targets, groups = load_feature_table(paths)
    cols = [c for g in cfg["point_model"]["feature_groups"] for c in groups.get(g, [])]
    _, pts = load_targets(cfg, paths, "points_89")
    pts = pts[pts["wafer_id"].isin(feats.index)].copy()
    pts["site"] = pts["x_mm"].round(1).astype(str) + "," + pts["y_mm"].round(1).astype(str)
    wids = np.array(sorted(pts["wafer_id"].unique()))
    lot_of = pts.groupby("wafer_id")["lot"].first().reindex(wids).to_numpy()
    splits, scheme = make_splits(lot_of, cfg["cv"]["scheme"], min_groups=cfg["cv"].get("min_groups", 3))
    Xw = feats.loc[wids, cols]
    yw = targets.loc[wids, "si_etch_mean"].to_numpy()
    pred = {m: np.full(len(pts), np.nan) for m in ("site_mean", "ridge_x_shape", "gbm_point")}
    ridge = build_model("ridge", cfg.get("model_params", {}).get("ridge"))
    gbm = build_model("gbm", dict(cfg.get("model_params", {}).get("gbm", {})))
    for tr, te in splits:
        trw, tew = set(wids[tr]), set(wids[te])
        ptr = pts[pts["wafer_id"].isin(trw)]
        mte = pts["wafer_id"].isin(tew).to_numpy()
        site_mean = ptr.groupby("site")["si_etch"].mean()
        shape = (ptr["si_etch"] / ptr.groupby("wafer_id")["si_etch"].transform("mean")).groupby(ptr["site"]).mean()
        pred["site_mean"][mte] = pts.loc[mte, "site"].map(site_mean).to_numpy()
        m = clone(ridge).fit(Xw.iloc[tr].to_numpy(), yw[tr])
        wmean = pd.Series(m.predict(Xw.iloc[te].to_numpy()).ravel(), index=wids[te])
        pred["ridge_x_shape"][mte] = (pts.loc[mte, "wafer_id"].map(wmean) * pts.loc[mte, "site"].map(shape)).to_numpy()

        def design(df):
            x = feats.loc[df["wafer_id"], cols].to_numpy()
            sp = df[["x_mm", "y_mm", "r_mm"]].to_numpy()
            return np.column_stack([x, sp, (df["r_mm"].to_numpy() ** 2)[:, None]])

        g = clone(gbm).fit(design(ptr), ptr["si_etch"].to_numpy())
        pred["gbm_point"][mte] = g.predict(design(pts[mte]))
    rows = []
    for name, p in pred.items():
        met = regression_metrics(pts["si_etch"].to_numpy(), p, pred["site_mean"])
        rows.append(
            {"model": name, "target": "si_etch_point", "cv": scheme, "n_wafers": len(wids), "n_points": len(pts), **met}
        )
    res = pd.DataFrame(rows)
    out = paths["outputs"] / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "point_benchmark.csv", index=False)
    pts.assign(**{f"pred_{k}": v for k, v in pred.items()}).to_parquet(out / "point_oof.parquet", index=False)
    return res


# ------------------------------------------------------------------------------------------- NN
def save_nn_inputs(path: Path, X: np.ndarray, ids: list[str], meta: dict) -> None:
    """Pack model inputs (X, wafer ids, meta) into one compressed .npz (no pickle)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X.astype(np.float32), ids=np.array(ids, dtype=str), meta=np.array(json.dumps(meta)))


def load_nn_inputs(path: str | Path) -> tuple[np.ndarray, list[str], dict]:
    z = np.load(path, allow_pickle=False)
    return z["X"], [str(i) for i in z["ids"]], json.loads(str(z["meta"]))


def run_nn(cfg: dict, paths: dict, mlflow_run: bool = True) -> dict:
    from .models.train_nn import build_inputs, train_cv

    t0 = time.perf_counter()
    if cfg["model"].get("inputs_npz"):
        # precomputed per-wafer tensors (scripts/pack_nn_inputs.py) -> identical inputs on any machine (e.g. Colab)
        X, ids, meta = load_nn_inputs(cfg["model"]["inputs_npz"])
    else:
        fd = paths["features"]
        cyc_oes = pd.read_parquet(fd / "cycles_oes.parquet") if (fd / "cycles_oes.parquet").exists() else None
        cyc_proc = pd.read_parquet(fd / "cycles_process.parquet") if (fd / "cycles_process.parquet").exists() else None
        X, ids, meta = build_inputs(cfg, cyc_oes, cyc_proc)
    prep_s = time.perf_counter() - t0
    if cfg.get("targets_parquet"):
        targets = pd.read_parquet(cfg["targets_parquet"])
    else:
        targets, _ = load_targets(cfg, paths)
    out_dir = paths["outputs"] / "nn" / cfg.get("run_name", cfg["model"]["type"])
    if cfg.get("checkpoint", {}).get("dir"):
        cfg["checkpoint"]["dir"] = str(Path(cfg["checkpoint"]["dir"]))
    mlf = None
    if mlflow_run:
        from .tracking import start_run

        mlf = start_run(cfg, run_name=cfg.get("run_name", cfg["model"]["type"]))
    out = train_cv(cfg, X, ids, meta, targets, out_dir)
    out["stats"]["input_shape"] = list(X.shape)
    out["stats"]["prep_seconds"] = prep_s
    (out_dir / "nn_stats.json").write_text(json.dumps(out["stats"], indent=2, default=str))
    if mlf is not None:
        for r in out["results"].itertuples():
            mlf.log_metrics(
                {
                    f"{r.target}.{k}": float(getattr(r, k))
                    for k in ("rmse", "mae", "r2", "coverage")
                    if np.isfinite(getattr(r, k))
                }
            )
        mlf.log_metrics({"peak_vram_mb": out["stats"]["peak_vram_mb"], "n_params": out["stats"]["n_params"]})
        mlf.log_artifact(str(out_dir / "nn_results.csv"))
        mlf.end_run()
    return out


# ------------------------------------------------------------------------------------ interpret
def run_interpret(cfg: dict, paths: dict) -> pd.DataFrame:
    from .models.cv import make_splits
    from .models.interpret import cv_permutation_importance, shap_tree_importance
    from .models.tabular import build_model

    feats, targets, groups = load_feature_table(paths)
    icfg = cfg["interpret"]
    cols = [c for g in cfg["feature_sets"][icfg["feature_set"]] for c in groups.get(g, [])]
    f = feats[cols].dropna(how="all")
    common = f.index.intersection(targets.index)
    f, tg = f.loc[common], targets.loc[common]
    splits, _ = make_splits(tg["lot"].to_numpy(), cfg["cv"]["scheme"], min_groups=cfg["cv"].get("min_groups", 3))
    out = paths["outputs"] / "interpret"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for tgt in icfg["targets"]:
        for mname in icfg["models"]:
            model = build_model(mname, dict(cfg.get("model_params", {}).get(mname, {})))
            y = tg[tgt].to_numpy()
            pi = cv_permutation_importance(
                model, f, y, splits, n_repeats=icfg.get("n_repeats", 5), top=icfg.get("top", 20)
            )
            pi.insert(0, "method", "perm_lolo")
            pi.insert(0, "model", mname)
            pi.insert(0, "target", tgt)
            frames.append(pi)
            if mname == "gbm":
                try:
                    sh = shap_tree_importance(model, f, y, top=icfg.get("top", 20))
                    sh = sh.rename(columns={"mean_abs_shap": "importance_mean"})
                    sh.insert(0, "method", "shap_full_fit")
                    sh.insert(0, "model", mname)
                    sh.insert(0, "target", tgt)
                    frames.append(sh)
                except Exception as exc:  # shap optional
                    log.warning("SHAP failed: %s", exc)
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(out / "feature_importance.csv", index=False)
    return res


# --------------------------------------------------------------------------------------- export
def export_bundle(cfg: dict, paths: dict) -> Path:
    """Fit the serving models on all wafers; conformal half-widths from LOLO OOF residuals."""
    import joblib
    from sklearn.base import clone

    from .models.conformal import split_conformal_quantile
    from .models.tabular import build_model

    feats, targets, groups = load_feature_table(paths)
    ecfg = cfg["export"]
    cols = [c for g in cfg["feature_sets"][ecfg["feature_set"]] for c in groups.get(g, [])]
    f = feats[cols].dropna(how="all")
    common = f.index.intersection(targets.index)
    oof_path = paths["outputs"] / "benchmark" / ecfg["feature_set"] / "oof_predictions.parquet"
    oof = pd.read_parquet(oof_path) if oof_path.exists() else None
    models, q = {}, {}
    for tgt in ecfg["targets"]:
        mname = ecfg["model"]
        m = clone(build_model(mname, dict(cfg.get("model_params", {}).get(mname, {})))).fit(
            f.loc[common].to_numpy(), targets.loc[common, tgt].to_numpy()
        )
        models[tgt] = m
        if oof is not None:
            o = oof[(oof["model"] == mname) & (oof["target"] == tgt)]
            res = np.abs(o["y"] - o["pred"]).to_numpy()
            q[tgt] = {str(a): split_conformal_quantile(res, a) for a in ecfg.get("alphas", [0.1, 0.2])}
    bundle = {
        "feature_columns": cols,
        "feature_set": ecfg["feature_set"],
        "model_name": ecfg["model"],
        "models": models,
        "conformal_halfwidth": q,
        "train_wafers": list(common),
        "train_feature_medians": f.loc[common].median().to_dict(),
        "train_feature_min": f.loc[common].min().to_dict(),
        "train_feature_max": f.loc[common].max().to_dict(),
        "target_units": {
            "si_etch_mean": "µm",
            "si_nu_halfrange_pct": "%",
            "si_nu_std_pct": "%",
            "si_nu_p90_pct": "%",
            "si_edge_center": "µm",
            "ox_etch_mean": "µm",
            "selectivity": "-",
        },
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = paths["outputs"] / "model_bundle.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    card = {
        k: v
        for k, v in bundle.items()
        if k not in ("models", "train_feature_medians", "train_feature_min", "train_feature_max")
    }
    card["n_features"] = len(cols)
    (paths["outputs"] / "model_card.json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    return out


def export_bundle_v2(cfg: dict, paths: dict) -> Path:
    """'v2 candidate' bundle: Si depth = n_cycles x GBM rate(tool + context), CV+ intervals (Barber et al. 2021).

    Stores the full-data rate model, the 10 leave-one-lot-out fold models and their out-of-fold residuals so the
    API can form CV+ intervals for a new wafer. Chosen after the first results (nested-by-lot selection over 22
    configs picked it in 10/10 folds) — the pre-declared ridge bundle stays the default.
    """
    import joblib
    from sklearn.base import clone

    from .models.tabular import build_model

    feats, targets, groups = load_feature_table(paths)
    cols = [c for g in ("tool", "timing", "context") for c in groups.get(g, [])]
    common = feats.index.intersection(targets["si_etch_mean"].dropna().index)
    X = feats.loc[common, cols].to_numpy(float)
    n = feats.loc[common, "TM_tool_etch_n"].to_numpy(float)
    y = targets.loc[common, "si_etch_mean"].to_numpy(float)
    lots = targets.loc[common, "lot"].to_numpy()
    gbm = build_model("gbm", dict(cfg.get("model_params", {}).get("gbm", {})))
    full = clone(gbm).fit(X, y / n)
    fold_models, fold_of, oof = [], np.full(len(y), -1), np.full(len(y), np.nan)
    for k, lt in enumerate(np.unique(lots)):
        te = lots == lt
        m = clone(gbm).fit(X[~te], y[~te] / n[~te])
        oof[te] = np.asarray(m.predict(X[te])).ravel() * n[te]
        fold_of[te] = k
        fold_models.append(m)
    f = feats.loc[common, cols]
    bundle = {
        "kind": "hybrid_cvplus",
        "target": "si_etch_mean",
        "feature_columns": cols,
        "cycles_column": "TM_tool_etch_n",
        "full_model": full,
        "fold_models": fold_models,
        "fold_of": fold_of,
        "residuals": np.abs(y - oof),
        "model_name": "hybrid n_cycles x LightGBM rate (tool + context)",
        "feature_set": "tool_context",
        "train_wafers": list(common),
        "train_feature_medians": f.median().to_dict(),
        "train_feature_min": f.min().to_dict(),
        "train_feature_max": f.max().to_dict(),
        "target_units": {"si_etch_mean": "µm"},
        "note": "v2 candidate, chosen after the first results; default served model remains the pre-declared ridge",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = paths["outputs"] / "model_bundle_v2.joblib"
    joblib.dump(bundle, out)
    card = {k: v for k, v in bundle.items() if k in ("kind", "target", "model_name", "feature_set", "note", "created")}
    card.update(n_features=len(cols), n_train_wafers=len(common), lolo_oof_rmse=float(np.sqrt(np.mean((y - oof) ** 2))))
    (paths["outputs"] / "model_card_v2.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return out


__all__ = [
    "TARGET_COLUMNS",
    "build_feature_table",
    "export_bundle",
    "load_feature_table",
    "run_interpret",
    "run_nn",
    "run_point_benchmark",
    "run_tabular_benchmark",
]
