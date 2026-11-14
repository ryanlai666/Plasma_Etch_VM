"""Uncertainty and robustness analyses behind REPORT.md (lot-bootstrap CIs, nested selection, CV+, physics).

    python scripts/review_analyses.py [--n-boot 2000] [--skip-nested]

Writes outputs/review/review_analyses.md (+ .json, bootstrap CSV). Single process (memory-safe).

1. variance decomposition of Si depth (within- vs between-lot) and what slot + cycle count explain;
2. lot-resampled bootstrap CIs on every (feature set, model) RMSE and paired differences vs. the pre-declared
   reference (ridge, tool+context) and vs. the slot/cycles/conditioning OLS baseline; lots won;
3. nested-by-lot model selection over all configs (inner leave-one-lot-out on the 9 training lots);
4. CV+ intervals nested by lot (Barber et al. 2021) vs. the split-conformal intervals of the benchmark;
5. do OES features explain the within-lot residuals of the tool model? (leave-one-lot-out residual model);
6. measurement-noise floor from the 9- vs 89-point campaigns;
7. hybrid model depth = n_cycles × rate(features);
8. absolute vs. share-of-total line intensities vs. slot, depth slope per lot;
9. idle gaps between wafers (OES epoch clock) and the lot-8 residuals.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.base import clone  # noqa: E402

from plasma_etch_vm.config import load_config  # noqa: E402
from plasma_etch_vm.metrology import load_lot_status, load_points, rotate_points  # noqa: E402
from plasma_etch_vm.models.conformal import cv_plus_interval  # noqa: E402
from plasma_etch_vm.models.tabular import build_model  # noqa: E402

OUT = ROOT / "outputs" / "review"
BENCH = ROOT / "outputs" / "benchmark"
FEAT = ROOT / "data" / "features"
STORE = ROOT / "data" / "parquet"
TARGETS = ["si_etch_mean", "si_nu_p90_pct", "si_edge_center", "ox_etch_mean", "selectivity", "si_nu_halfrange_pct"]
REF = ("tool_context", "ridge")
BASE = ("slot_cycles_cond", "ols")


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def f(v, nd=3) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def md(df: pd.DataFrame, nd=3) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f(v, nd) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- data loading
def load_oof() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(BENCH.glob("*/oof_predictions.parquet"))]
    o = pd.concat(parts, ignore_index=True)
    o["config"] = o["feature_set"] + "·" + o["model"]
    return o


# ------------------------------------------------------------------------------- bootstrap
def lot_bootstrap(oof: pd.DataFrame, target: str, n_boot: int, seed: int = 0) -> pd.DataFrame:
    """RMSE CI per config + paired ΔRMSE vs REF and BASE, resampling whole lots with replacement."""
    o = oof[(oof["target"] == target) & (oof["model"] != "mean")]
    wide = o.pivot_table(index=["wafer_id", "lot"], columns="config", values="pred").reset_index()
    y = o.drop_duplicates("wafer_id").set_index("wafer_id")["y"].reindex(wide["wafer_id"]).to_numpy()
    lots = wide["lot"].to_numpy()
    configs = [c for c in wide.columns if "·" in c]
    P = wide[configs].to_numpy()
    ulots = np.unique(lots)
    idx_by_lot = {lt: np.flatnonzero(lots == lt) for lt in ulots}
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(configs)))
    for b in range(n_boot):
        pick = rng.choice(ulots, size=len(ulots), replace=True)
        ii = np.concatenate([idx_by_lot[lt] for lt in pick])
        boots[b] = np.sqrt(np.mean((P[ii] - y[ii, None]) ** 2, axis=0))
    point = np.sqrt(np.mean((P - y[:, None]) ** 2, axis=0))
    ref_name, base_name = "·".join(REF), "·".join(BASE)
    rows = []
    for j, c in enumerate(configs):
        row = {
            "config": c,
            "RMSE": point[j],
            "RMSE lo": np.percentile(boots[:, j], 2.5),
            "RMSE hi": np.percentile(boots[:, j], 97.5),
        }
        for tag, name in (("ref", ref_name), ("base", base_name)):
            if name in configs:
                k = configs.index(name)
                d = boots[:, j] - boots[:, k]
                row[f"Δ vs {tag}"] = point[j] - point[k]
                row[f"Δ {tag} lo"] = np.percentile(d, 2.5)
                row[f"Δ {tag} hi"] = np.percentile(d, 97.5)
                per_lot = [
                    rmse(y[idx_by_lot[lt]], P[idx_by_lot[lt], j]) < rmse(y[idx_by_lot[lt]], P[idx_by_lot[lt], k])
                    for lt in ulots
                ]
                row[f"lots won vs {tag}"] = f"{int(np.sum(per_lot))}/{len(ulots)}"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("RMSE")


# ------------------------------------------------------------------- nested selection + CV+
def inner_lolo(model, X, y, lots, tr, te):
    """Inner leave-one-lot-out on the training wafers: OOF predictions, fold ids, fold models' test preds."""
    ltr = lots[tr]
    inner_lots = np.unique(ltr)
    oof = np.full(len(tr), np.nan)
    fold_of = np.full(len(tr), -1)
    test_preds = np.empty((len(inner_lots), len(te)))
    for k, lt in enumerate(inner_lots):
        m_in = ltr != lt
        mdl = clone(model).fit(X[tr][m_in], y[tr][m_in])
        oof[~m_in] = np.asarray(mdl.predict(X[tr][~m_in])).ravel()
        fold_of[~m_in] = k
        test_preds[k] = np.asarray(mdl.predict(X[te])).ravel()
    return oof, fold_of, test_preds


HYBRIDS = (  # depth = n_cycles x rate(features); (name, model, feature groups)
    ("hybrid_rate_tool_context", "ridge", ["tool", "timing", "context"]),
    ("hybrid_rate_tool_context", "gbm", ["tool", "timing", "context"]),
    ("hybrid_rate_context", "ols", ["context"]),
)


def nested_and_cvplus(feats, targets, groups, sets, model_sets, params, target, alpha=0.1, with_hybrids=True):
    """Nested-by-lot selection over every config (benchmark configs + hybrids) and CV+ intervals per config.

    Hybrid configs fit the per-cycle rate y / n_cycles and predict n_cycles x rate; their residuals, CV+ scores and
    selection scores are all computed in depth units, so they compete on equal terms.
    """
    y_all = targets[target]
    common = feats.index.intersection(y_all.dropna().index)
    lots = targets.loc[common, "lot"].to_numpy()
    y = y_all.loc[common].to_numpy(float)
    ncyc = feats.loc[common, "TM_tool_etch_n"].to_numpy(float)
    configs = []  # (name, model, cols, is_rate)
    for fs, gs in sets.items():
        cols = [c for g in gs for c in groups.get(g, [])]
        for m in model_sets.get(fs, []):
            if m != "mean":
                configs.append((fs, m, cols, False))
    if with_hybrids and target == "si_etch_mean":
        for name, m, gs in HYBRIDS:
            configs.append((name, m, [c for g in gs for c in groups[g]], True))
    out = {"per_config": {}, "selection": []}
    sel_pred = np.full(len(y), np.nan)
    for fs, m, _c, _r in configs:
        out["per_config"][f"{fs}·{m}"] = {
            "pred": np.full(len(y), np.nan),
            "lo": np.full(len(y), np.nan),
            "hi": np.full(len(y), np.nan),
            "inner_rmse": {},
        }
    for lt in np.unique(lots):
        te = np.flatnonzero(lots == lt)
        tr = np.flatnonzero(lots != lt)
        best, best_score = None, np.inf
        for fs, m, cols, is_rate in configs:
            X = feats.loc[common, cols].to_numpy(float)
            model = build_model(m, dict(params.get(m, {}) or {}))
            yfit = y / ncyc if is_rate else y
            oof, fold_of, tp = inner_lolo(model, X, yfit, lots, tr, te)
            if is_rate:
                oof, tp = oof * ncyc[tr], tp * ncyc[te][None, :]
            score = rmse(y[tr], oof)
            rec = out["per_config"][f"{fs}·{m}"]
            rec["inner_rmse"][int(lt)] = score
            full = clone(model).fit(X[tr], yfit[tr])
            pred = np.asarray(full.predict(X[te])).ravel()
            rec["pred"][te] = pred * ncyc[te] if is_rate else pred
            lo, hi = cv_plus_interval(np.abs(y[tr] - oof), fold_of, tp, alpha)
            rec["lo"][te], rec["hi"][te] = lo, hi
            if score < best_score:
                best, best_score = f"{fs}·{m}", score
        sel_pred[te] = out["per_config"][best]["pred"][te]
        out["selection"].append({"held-out lot": int(lt), "selected config": best, "inner RMSE": best_score})
    out["y"], out["lots"], out["wafer_id"], out["selected_pred"] = y, lots, list(common), sel_pred
    return out


# ------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-nested", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg = load_config("models/tabular.yaml")
    feats = pd.read_parquet(FEAT / "wafer_features.parquet")
    targets = pd.read_parquet(FEAT / "wafer_targets.parquet")
    groups = json.loads((FEAT / "feature_groups.json").read_text())
    oof = load_oof()
    res: dict = {}
    doc = ["# Review analyses (generated by scripts/review_analyses.py)", ""]

    # 1. variance decomposition --------------------------------------------------------------
    d = targets[["lot", "wafer", "si_etch_mean"]].join(feats[["TM_tool_etch_n", "C_slot"]], how="left")
    y = d["si_etch_mean"]
    within = y - d.groupby("lot")["si_etch_mean"].transform("mean")
    share = float((within**2).sum() / ((y - y.mean()) ** 2).sum())
    Z = pd.DataFrame({"slot": d["C_slot"], "cyc": d["TM_tool_etch_n"]})
    Zw = Z - Z.groupby(d["lot"]).transform("mean")
    coef, *_ = np.linalg.lstsq(Zw.to_numpy(), within.to_numpy(), rcond=None)
    resid = within.to_numpy() - Zw.to_numpy() @ coef
    n_cfg = oof[(oof["target"] == "si_etch_mean") & (oof["model"] != "mean")]["config"].nunique()
    res["variance"] = {
        "n_wafers": len(y),
        "total_sd": float(y.std(ddof=1)),
        "within_lot_share_of_variance": share,
        "within_lot_sd": float(within.std(ddof=1)),
        "within_lot_resid_sd_after_slot_cycles_insample": float(resid.std(ddof=1)),
        "slope_um_per_slot": float(coef[0]),
        "um_per_cycle": float(coef[1]),
        "n_configs_benchmarked_si_depth": int(n_cfg),
    }
    v = res["variance"]
    doc += [
        "## 1. Variance decomposition of Si depth",
        "",
        f"{v['n_wafers']} wafers: total SD {f(v['total_sd'])} µm; **{100 * share:.0f} % of the variance is within lots** "
        f"(within-lot SD {f(v['within_lot_sd'])} µm). Within-lot least squares on slot + logged cycle count (in-sample) "
        f"leaves SD {f(v['within_lot_resid_sd_after_slot_cycles_insample'])} µm (slope {f(v['slope_um_per_slot'], 4)} "
        f"µm/slot, {f(v['um_per_cycle'], 3)} µm/cycle). {n_cfg} (feature set, model) configs were benchmarked for depth.",
        "",
    ]

    # 2. lot bootstrap on every comparison ---------------------------------------------------
    doc += ["## 2. Lot-resampled 95 % CIs (2 000 bootstrap resamples of the 10 lots)", ""]
    res["bootstrap"] = {}
    for t in TARGETS:
        if t not in set(oof["target"]):
            continue
        b = lot_bootstrap(oof, t, a.n_boot)
        b.to_csv(OUT / f"bootstrap_{t}.csv", index=False)
        res["bootstrap"][t] = b.to_dict(orient="records")
        show = pd.DataFrame(
            {
                "config": b["config"],
                "RMSE [95 % CI]": [
                    f"{r:.3f} [{lo:.3f}, {hi:.3f}]" for r, lo, hi in zip(b["RMSE"], b["RMSE lo"], b["RMSE hi"])
                ],
                "Δ vs ridge·tool+ctx [CI]": [
                    f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]" for x, lo, hi in zip(b["Δ vs ref"], b["Δ ref lo"], b["Δ ref hi"])
                ],
                "lots won vs ref": b["lots won vs ref"],
                "Δ vs OLS slot+cycles+cond [CI]": [
                    f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                    for x, lo, hi in zip(b["Δ vs base"], b["Δ base lo"], b["Δ base hi"])
                ],
                "lots won vs base": b["lots won vs base"],
            }
        )
        doc += [f"### {t}", "", md(show), ""]

    # 2b. explicit paired comparisons ---------------------------------------------------------
    pairs = [
        ("tool_context·gbm", "tool_context·ridge"),
        ("oes_tool_context·gbm", "tool_context·gbm"),
        ("oes_tool_context·ridge", "tool_context·ridge"),
        ("tool_context·ridge", "slot_cycles_cond·ols"),
        ("oes_spectrum·pls", "slot_cycles_cond·ols"),
    ]
    prow = []
    for t in ("si_etch_mean", "si_nu_p90_pct", "si_edge_center"):
        o = oof[oof["target"] == t]
        wide = o.pivot_table(index=["wafer_id", "lot"], columns="config", values="pred").reset_index()
        yy = o.drop_duplicates("wafer_id").set_index("wafer_id")["y"].reindex(wide["wafer_id"]).to_numpy()
        lt_arr = wide["lot"].to_numpy()
        ul = np.unique(lt_arr)
        rng = np.random.default_rng(3)
        picks = [rng.choice(ul, len(ul)) for _ in range(a.n_boot)]
        for c1, c2 in pairs:
            if c1 not in wide or c2 not in wide:
                continue
            p1, p2 = wide[c1].to_numpy(), wide[c2].to_numpy()
            ds = []
            for pk in picks:
                ii = np.concatenate([np.flatnonzero(lt_arr == q) for q in pk])
                ds.append(rmse(yy[ii], p1[ii]) - rmse(yy[ii], p2[ii]))
            won = sum(rmse(yy[lt_arr == q], p1[lt_arr == q]) < rmse(yy[lt_arr == q], p2[lt_arr == q]) for q in ul)
            prow.append(
                {
                    "target": t,
                    "A": c1,
                    "B": c2,
                    "RMSE A": rmse(yy, p1),
                    "RMSE B": rmse(yy, p2),
                    "Δ A−B [95 % CI]": f"{rmse(yy, p1) - rmse(yy, p2):+.3f} "
                    f"[{np.percentile(ds, 2.5):+.3f}, {np.percentile(ds, 97.5):+.3f}]",
                    "lots A wins": f"{won}/{len(ul)}",
                }
            )
    res["pairs"] = prow
    doc += ["## 2b. Paired comparisons (lot bootstrap)", "", md(pd.DataFrame(prow), 3), ""]

    # 6 (early, cheap). measurement-noise floor ------------------------------------------------
    raw = ROOT / "data" / "raw"
    p89 = load_points(raw / "Si_Oxide_etch_89_points.csv")
    p9 = rotate_points(load_points(raw / "Si_Oxide_etch_9_points.csv"), 180)
    w9 = p9.groupby("wafer_id")["si_etch"].mean()
    w89 = p89.groupby("wafer_id")["si_etch"].mean()
    j = pd.concat([w9.rename("m9"), w89.rename("m89")], axis=1, join="inner")
    p9r = p9.assign(x_mm=p9["x_mm"].round(1), y_mm=p9["y_mm"].round(1))
    p89r = p89.assign(x_mm=p89["x_mm"].round(1), y_mm=p89["y_mm"].round(1))
    mm = p9r.merge(p89r, on=["wafer_id", "x_mm", "y_mm"], suffixes=("_9", "_89"))
    same = mm.groupby("wafer_id")[["si_etch_9", "si_etch_89"]].mean()
    d_full = (j["m89"] - j["m9"]).to_numpy()
    d_same = (same["si_etch_89"] - same["si_etch_9"]).to_numpy()
    res["noise_floor"] = {
        "n_wafers": len(j),
        "offset_full_mean": float(d_full.mean()),
        "sd_diff_wafer_means": float(d_full.std(ddof=1)),
        "n_wafers_same_sites": len(same),
        "offset_same_sites": float(d_same.mean()),
        "sd_diff_same_9_sites": float(d_same.std(ddof=1)),
        "implied_single_campaign_sd_same_sites": float(d_same.std(ddof=1) / np.sqrt(2)),
    }
    nf = res["noise_floor"]
    doc += [
        "## 6. Measurement-noise floor",
        "",
        f"9-point (rotated 180°) vs 89-point wafer means, {nf['n_wafers']} wafers: offset {f(nf['offset_full_mean'])} µm, "
        f"SD of the difference **{f(nf['sd_diff_wafer_means'])} µm**. Restricted to the same 9 sites in both campaigns: "
        f"offset {f(nf['offset_same_sites'])} µm, SD {f(nf['sd_diff_same_9_sites'])} µm → if both campaigns are equally "
        f"noisy, one campaign's wafer-mean noise ≈ {f(nf['implied_single_campaign_sd_same_sites'])} µm. Model RMSEs of "
        "0.15–0.17 µm are therefore ~2–3× the metrology noise, not at it.",
        "",
    ]

    # 7. hybrid depth = cycles × rate ----------------------------------------------------------
    common = feats.index.intersection(targets.index)
    lots = targets.loc[common, "lot"].to_numpy()
    yv = targets.loc[common, "si_etch_mean"].to_numpy(float)
    ncyc = feats.loc[common, "TM_tool_etch_n"].to_numpy(float)
    rate = yv / ncyc
    hy = {}
    for name, gs, m in (
        ("direct OLS slot+cycles (no conditioning)", None, "ols"),
        ("rate(slot+cond) OLS × cycles", ["context"], "ols"),
        ("rate(tool+ctx) ridge × cycles", ["tool", "timing", "context"], "ridge"),
        ("rate(tool+ctx) GBM × cycles", ["tool", "timing", "context"], "gbm"),
    ):
        direct = gs is None
        cols = ["C_slot", "TM_tool_etch_n"] if direct else [c for g in gs for c in groups[g]]
        X = feats.loc[common, cols].to_numpy(float)
        pred = np.full(len(yv), np.nan)
        for lt in np.unique(lots):
            te, tr = lots == lt, lots != lt
            yfit = yv if direct else rate
            mdl = build_model(m, dict(cfg.get("model_params", {}).get(m, {}) or {})).fit(X[tr], yfit[tr])
            pred[te] = np.asarray(mdl.predict(X[te])).ravel() * (1.0 if direct else ncyc[te])
        hy[name] = pred
    hyb_oof = []
    for name, pred in hy.items():
        hyb_oof.append(
            pd.DataFrame(
                {
                    "wafer_id": common,
                    "lot": lots,
                    "y": yv,
                    "pred": pred,
                    "target": "si_etch_mean",
                    "feature_set": "hybrid",
                    "model": name,
                }
            )
        )
    hyb = pd.concat(hyb_oof)
    hyb.to_parquet(OUT / "hybrid_oof.parquet", index=False)
    hyb["config"] = "hybrid·" + hyb["model"]
    o2 = pd.concat([oof[oof["target"] == "si_etch_mean"], hyb], ignore_index=True)
    bh = lot_bootstrap(o2, "si_etch_mean", a.n_boot)
    bh = bh[
        bh["config"].str.startswith("hybrid") | bh["config"].isin(["·".join(REF), "·".join(BASE), "tool_context·gbm"])
    ]
    res["hybrid"] = bh.to_dict(orient="records")
    doc += [
        "## 7. Hybrid model: depth = n_cycles (tool log) × rate(features)",
        "",
        md(
            pd.DataFrame(
                {
                    "config": bh["config"],
                    "RMSE [95 % CI]": [
                        f"{r:.3f} [{lo:.3f}, {hi:.3f}]" for r, lo, hi in zip(bh["RMSE"], bh["RMSE lo"], bh["RMSE hi"])
                    ],
                    "Δ vs ridge·tool+ctx [CI]": [
                        f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                        for x, lo, hi in zip(bh["Δ vs ref"], bh["Δ ref lo"], bh["Δ ref hi"])
                    ],
                    "lots won vs ref": bh["lots won vs ref"],
                }
            )
        ),
        "",
    ]

    # 8. absolute vs share line intensities, depth slope per lot -------------------------------
    co = pd.read_parquet(FEAT / "cycles_oes.parquet")
    co = co[co["cycle"] >= 1]
    med = co.groupby("phase")["n"].transform("median")
    co = co[co["n"] >= 0.5 * med]
    co["wafer_id"] = co["day"] + "_" + co["wafer"].astype(int).map("{:02d}".format)
    per = {}
    for ph, col in (("dep", "L_C2_516"), ("dep", "total"), ("etch", "L_F_685"), ("etch", "total"), ("dep", "L_F_685")):
        per[f"{col}_{ph}_abs"] = co[co["phase"] == ph].groupby("wafer_id")[col].mean()
    W = pd.DataFrame(per)
    W["C2_share_dep"] = W["L_C2_516_dep_abs"] / W["total_dep_abs"]
    W["F_share_etch"] = W["L_F_685_etch_abs"] / W["total_etch_abs"]
    W["F_dep_over_F_etch"] = W["L_F_685_dep_abs"] / W["L_F_685_etch_abs"]
    W["slot"] = W.index.str.slice(11).astype(int)
    W["lot"] = W.index.str.slice(0, 10)
    rows = []
    for c in [
        "L_C2_516_dep_abs",
        "total_dep_abs",
        "C2_share_dep",
        "L_F_685_etch_abs",
        "total_etch_abs",
        "F_share_etch",
        "F_dep_over_F_etch",
    ]:
        x = W[c]
        xw = x - W.groupby("lot")[c].transform("mean")
        sw = W["slot"] - W.groupby("lot")["slot"].transform("mean")
        slope = float((xw * sw).sum() / (sw**2).sum())
        rows.append(
            {
                "signal": c,
                "within-lot r(slot)": float(np.corrcoef(xw, sw)[0, 1]),
                "% change slot 1→10": float(100 * slope * 9 / x.mean()),
                "window-transmission independent": "yes" if c == "F_dep_over_F_etch" else "no",
            }
        )
    res["line_abs_vs_share"] = rows
    lot_slopes = []
    tt = targets.join(feats[["C_slot"]], how="left")
    lstat = load_lot_status(raw / "Lot_status.xlsx").set_index("lot")
    for lt, g in tt.groupby("lot"):
        s, _intercept = np.polyfit(g["C_slot"], g["si_etch_mean"], 1)
        lot_slopes.append(
            {"lot": int(lt), "conditioning": lstat.loc[lt, "conditioning"], "wafers": len(g), "slope µm/slot": float(s)}
        )
    res["depth_slope_per_lot"] = lot_slopes
    doc += [
        "## 8. Absolute vs share-of-total line intensities (96 wafers, within-lot)",
        "",
        md(pd.DataFrame(rows), 3),
        "",
        "Depth slope vs slot per lot:",
        "",
        md(pd.DataFrame(lot_slopes), 3),
        "",
    ]

    # 9. idle gaps between wafers (OES epoch clock) ---------------------------------------------
    starts = []
    for p in sorted((STORE / "oes_trace").glob("day=*/lot=*/wafer=*/part-0.parquet")):
        day = p.parent.parent.parent.name.split("=")[1]
        w = int(p.parent.name.split("=")[1])
        tr = pd.read_parquet(p, columns=["t_s", "phase"])
        on = tr[tr["phase"] != "off"]["t_s"]
        starts.append(
            {
                "day": day,
                "wafer": w,
                "rec_start": float(tr["t_s"].iloc[0]),
                "plasma_start": float(on.iloc[0]) if len(on) else np.nan,
            }
        )
    S = pd.DataFrame(starts).sort_values(["day", "wafer"])
    S["gap_s"] = S.groupby("day")["plasma_start"].diff()
    med_gap = float(S["gap_s"].median())
    long = S[S["gap_s"] > 1.3 * med_gap]
    res["idle_gaps"] = {"median_s": med_gap, "long": long[["day", "wafer", "gap_s"]].to_dict(orient="records")}
    lot8 = oof[(oof["target"] == "si_etch_mean") & (oof["lot"] == 8)].copy()
    lot8["resid"] = lot8["y"] - lot8["pred"]
    lot8["slot"] = lot8["wafer_id"].str.slice(11).astype(int)
    keep = [
        "context·ridge",
        "slot_cycles_cond·ols",
        "tool_context·ridge",
        "tool_context·gbm",
        "oes_spectrum·pls",
        "oes_physics·gbm",
    ]
    piv = lot8[lot8["config"].isin(keep)].pivot_table(index="slot", columns="config", values="resid")
    res["lot8_residuals"] = piv.round(3).reset_index().to_dict(orient="records")
    doc += [
        "## 9. Idle time between wafers (plasma-on start to plasma-on start, OES clock)",
        "",
        f"Median gap {med_gap:.0f} s. Gaps > 1.3 × median:",
        "",
        md(long[["day", "wafer", "gap_s"]].rename(columns={"gap_s": "gap before wafer (s)"}), 0)
        if len(long)
        else "none",
        "",
        "Lot 8 (2024-08-07) out-of-fold residuals (measured − predicted, µm):",
        "",
        md(piv.reset_index(), 3),
        "",
    ]

    # 5. do OES features explain the tool model's residuals? -----------------------------------
    ref_oof = oof[(oof["target"] == "si_etch_mean") & (oof["config"] == "·".join(REF))].set_index("wafer_id")
    r = (ref_oof["y"] - ref_oof["pred"]).reindex(common).to_numpy()
    rr = []
    for fs, gs, m in (
        ("oes_physics", ["oes", "ratio", "oes_timing", "transition"], "ridge"),
        ("oes_physics", ["oes", "ratio", "oes_timing", "transition"], "pls"),
        ("oes_spectrum", ["spectrum"], "pls"),
        ("oes_spectrum", ["spectrum"], "pca_ridge"),
    ):
        cols = [c for g in gs for c in groups[g]]
        X = feats.loc[common, cols].to_numpy(float)
        pr = np.full(len(r), np.nan)
        for lt in np.unique(lots):
            te, tr = lots == lt, lots != lt
            mdl = build_model(m, dict(cfg.get("model_params", {}).get(m, {}) or {})).fit(X[tr], r[tr])
            pr[te] = np.asarray(mdl.predict(X[te])).ravel()
        rng = np.random.default_rng(1)
        ul = np.unique(lots)
        bd = []
        for _ in range(a.n_boot):
            ii = np.concatenate([np.flatnonzero(lots == lt) for lt in rng.choice(ul, len(ul))])
            bd.append(rmse(r[ii], pr[ii]) - rmse(r[ii], 0 * r[ii]))
        rr.append(
            {
                "residual model": f"{fs}·{m}",
                "residual RMSE before": rmse(r, 0 * r),
                "after OES correction": rmse(r, pr),
                "Δ [95 % CI]": f"{rmse(r, pr) - rmse(r, 0 * r):+.3f} [{np.percentile(bd, 2.5):+.3f}, {np.percentile(bd, 97.5):+.3f}]",
                "OOF R² of residual": 1 - np.sum((r - pr) ** 2) / np.sum((r - r.mean()) ** 2),
            }
        )
    oes_cols = [c for g in ["oes", "ratio", "transition"] for c in groups[g]]
    cor = []
    for c in oes_cols:
        x = feats.loc[common, c].to_numpy(float)
        ok = np.isfinite(x)
        if ok.sum() > 10 and np.std(x[ok]) > 0:
            cor.append((c, float(np.corrcoef(x[ok], r[ok])[0, 1])))
    cor = sorted(cor, key=lambda z: -abs(z[1]))[:5]
    res["oes_explains_residuals"] = {"models": rr, "top_feature_correlations": cor, "n_features_tested": len(oes_cols)}
    doc += [
        "## 5. Do OES features explain the within-lot residuals of the tool + context ridge model?",
        "",
        md(pd.DataFrame(rr), 3),
        "",
        f"Largest |r| between a single OES feature and the residual (of {len(oes_cols)} tested; with ~88 wafers |r| ≈ 0.21 "
        "is p ≈ 0.05 before any multiple-testing correction): " + ", ".join(f"`{c}` {v:+.2f}" for c, v in cor),
        "",
    ]

    # 3 + 4. nested selection and CV+ --------------------------------------------------------------
    if not a.skip_nested:
        tn = time.time()
        sets = cfg["feature_sets"]
        msets = cfg["model_feature_sets"]
        nest = nested_and_cvplus(feats, targets, groups, sets, msets, cfg.get("model_params", {}), "si_etch_mean")
        yv2, lots2 = nest["y"], nest["lots"]
        sel = pd.DataFrame(nest["selection"])
        sp = nest["selected_pred"]
        # bootstrap for the nested-selected predictions vs the pre-declared ridge
        ref_pred = nest["per_config"]["·".join(REF)]["pred"]
        rng = np.random.default_rng(2)
        ul = np.unique(lots2)
        bs, bd = [], []
        for _ in range(a.n_boot):
            ii = np.concatenate([np.flatnonzero(lots2 == lt) for lt in rng.choice(ul, len(ul))])
            bs.append(rmse(yv2[ii], sp[ii]))
            bd.append(rmse(yv2[ii], sp[ii]) - rmse(yv2[ii], ref_pred[ii]))
        res["nested_selection"] = {
            "rmse": rmse(yv2, sp),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "delta_vs_ridge": rmse(yv2, sp) - rmse(yv2, ref_pred),
            "delta_ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
            "selections": nest["selection"],
            "seconds": time.time() - tn,
        }
        ns = res["nested_selection"]
        cvp = []
        split = oof[oof["target"] == "si_etch_mean"]
        for name, rec in nest["per_config"].items():
            lo, hi = rec["lo"], rec["hi"]
            cov = (yv2 >= lo) & (yv2 <= hi)
            per_lot_cov = pd.Series(cov).groupby(lots2).mean()
            s_o = split[split["config"] == name]
            s_w = (s_o["hi"] - s_o["lo"]).to_numpy()
            s_cov = ((s_o["y"] >= s_o["lo"]) & (s_o["y"] <= s_o["hi"])).to_numpy()
            s_lot = pd.Series(s_cov, index=s_o["lot"].to_numpy()).groupby(level=0).mean()
            half_by_lot = pd.Series((hi - lo) / 2).groupby(lots2).mean()
            s_half_by_lot = pd.Series(s_w / 2, index=s_o["lot"].to_numpy()).groupby(level=0).mean()
            cvp.append(
                {
                    "config": name,
                    "CV+ coverage": float(cov.mean()),
                    "CV+ min lot cov": float(per_lot_cov.min()),
                    "CV+ mean width": float(np.mean(hi - lo)),
                    "CV+ half-width range": f"{half_by_lot.min():.3f}–{half_by_lot.max():.3f}",
                    "split coverage": float(s_cov.mean()) if len(s_cov) else np.nan,
                    "split min lot cov": float(s_lot.min()) if len(s_lot) else np.nan,
                    "split mean width": float(np.mean(s_w)) if len(s_w) else np.nan,
                    "split half-width range": f"{s_half_by_lot.min():.3f}–{s_half_by_lot.max():.3f}"
                    if len(s_w)
                    else "–",
                }
            )
        cvp = pd.DataFrame(cvp).sort_values("CV+ mean width")
        rows_x = []
        for name, rec in nest["per_config"].items():
            rows_x.append(
                pd.DataFrame(
                    {
                        "wafer_id": nest["wafer_id"],
                        "lot": lots2,
                        "y": yv2,
                        "config": name,
                        "pred": rec["pred"],
                        "cvplus_lo": rec["lo"],
                        "cvplus_hi": rec["hi"],
                    }
                )
            )
        rows_x.append(
            pd.DataFrame(
                {
                    "wafer_id": nest["wafer_id"],
                    "lot": lots2,
                    "y": yv2,
                    "config": "nested_selection",
                    "pred": sp,
                    "cvplus_lo": np.nan,
                    "cvplus_hi": np.nan,
                }
            )
        )
        pd.concat(rows_x, ignore_index=True).to_parquet(OUT / "nested_cvplus_oof.parquet", index=False)
        res["cv_plus"] = cvp.to_dict(orient="records")
        doc += [
            "## 3. Nested-by-lot model selection (Si depth)",
            "",
            f"For each held-out lot, every config ({len(nest['per_config'])}) is scored by inner leave-one-lot-out on the 9 "
            f"training lots and the best one predicts the held-out lot. Nested RMSE **{f(ns['rmse'])} µm** "
            f"[{f(ns['ci'][0])}, {f(ns['ci'][1])}]; Δ vs the pre-declared ridge {ns['delta_vs_ridge']:+.3f} µm "
            f"[{ns['delta_ci'][0]:+.3f}, {ns['delta_ci'][1]:+.3f}]. ({ns['seconds'] / 60:.1f} min CPU)",
            "",
            md(sel, 3),
            "",
            "## 4. CV+ (nested by lot) vs split-conformal intervals, Si depth, nominal 90 %",
            "",
            "Split conformal = benchmark (calibration on round(0.25 × 9) = 2 of the 9 training lots, ≈ 20 wafers). "
            "CV+ uses the inner leave-one-lot-out residuals of all 9 training lots.",
            "",
            md(cvp, 3),
            "",
        ]
    res["seconds_total"] = time.time() - t0
    (OUT / "review_analyses.md").write_text("\n".join(doc) + "\n", encoding="utf-8")
    (OUT / "review_analyses.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"wrote {OUT / 'review_analyses.md'} in {res['seconds_total']:.0f}s")


if __name__ == "__main__":
    main()
