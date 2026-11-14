"""Build the result tables / figures for REPORT.md directly from run outputs (no hand-typed numbers).

    python scripts/summarize_results.py

Reads outputs/benchmark, outputs/nn, outputs/interpret, outputs/campaign_state.json, outputs/campaign.log,
data/parquet/_manifest, data/features. Writes outputs/summary/summary.md (+ .json) and interactive
figure previews in outputs/figures/; the static docs/figures/*.png come from scripts/make_figures.py.
Sections that lack inputs are skipped.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "outputs"
SUM = OUT / "summary"
FEAT = ROOT / "data" / "features"
STORE = ROOT / "data" / "parquet"
TARGETS = ["si_etch_mean", "si_nu_halfrange_pct", "si_nu_p90_pct", "si_edge_center", "ox_etch_mean", "selectivity"]
NN_TARGETS = ["si_etch_mean", "si_nu_p90_pct", "ox_etch_mean"]  # robust (P5-P95) NU
LOTS_A, LOTS_B = {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}


def fmt(v, nd=3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.{nd}f}"


def md_table(df: pd.DataFrame, nd: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [fmt(v, nd) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2)))


# ------------------------------------------------------------------------------------- data
def data_section(res: dict) -> list[str]:
    out = ["## Data actually used", ""]
    st = json.loads((OUT / "campaign_state.json").read_text()) if (OUT / "campaign_state.json").exists() else {}
    recs = [json.loads(p.read_text()) for p in sorted((STORE / "_manifest").glob("oes__*.json"))]
    if recs:
        df = pd.DataFrame(
            [{"day": r["source"]["file"][4:14].replace("_", "-"), **r["stats"]} for r in recs if "stats" in r]
        )
        g = df.groupby("day").agg(
            wafers=("group", "count"), rows=("rows", "sum"), seconds=("seconds", "sum"), mb_out=("bytes_out", "sum")
        )
        g["mb_out"] = g["mb_out"] / 1e6
        g["rows_per_s"] = g["rows"] / g["seconds"]
        g = g.reset_index()
        res["data"] = {
            "days": len(g),
            "wafers": int(g["wafers"].sum()),
            "oes_rows": int(g["rows"].sum()),
            "ingest_seconds": float(g["seconds"].sum()),
            "parquet_oes_mb": float(g["mb_out"].sum()),
        }
        out += [
            f"OES ingested: **{len(g)} day files, {int(g['wafers'].sum())} wafers, {int(g['rows'].sum()):,} spectra** "
            f"(× 3648 channels = {g['rows'].sum() * 3648 / 1e9:.2f} G channel-values); ingestion CPU time "
            f"{g['seconds'].sum() / 60:.1f} min; Parquet OES {g['mb_out'].sum() / 1e3:.2f} GB (counts_i16).",
            "",
            md_table(g.rename(columns={"mb_out": "parquet MB", "rows_per_s": "rows/s"}), 1),
            "",
        ]
    # download timings from the campaign log
    log = (OUT / "campaign.log").read_text(encoding="utf-8") if (OUT / "campaign.log").exists() else ""
    dl = re.findall(r"(\S+ \S+) DATA  (Day_\S+): downloaded \+ MD5 verified \((\d+) MB, (\d+)s elapsed\)", log)
    if dl:
        mb = sum(int(x[2]) for x in dl)
        secs = max(int(x[3]) for x in dl)
        res["download"] = {"files": len(dl), "mb": mb, "seconds": secs, "mb_per_s": mb / max(secs, 1)}
        out += [
            f"Download: {len(dl)} day files, {mb / 1e3:.2f} GB in {secs / 60:.1f} min with 3 parallel "
            f"streams → **{mb / max(secs, 1):.2f} MB/s** aggregate; every file MD5-verified, ingested, its Parquet "
            "row counts verified against the netCDF, then the raw file deleted (bounded disk use).",
            "",
        ]
    deleted = [k for k, v in st.items() if k.startswith("data:") and v.get("done")]
    if deleted:
        out += [f"Raw day files verified + deleted: {len(deleted)}.", ""]
    return out


# ---------------------------------------------------------------------------------- tabular
def tabular_section(res: dict) -> list[str]:
    f = OUT / "benchmark" / "benchmark_all.csv"
    if not f.exists():
        return []
    b = pd.read_csv(f)
    out = [
        "## Tabular models (leave-one-lot-out, 10 folds)",
        "",
        "Out-of-fold RMSE / R² / empirical coverage of the 90 % split-conformal interval (calibration lots held "
        "out inside each training fold). `n` = wafers with that feature set and target.",
        "",
    ]
    res["tabular"] = {}
    for t in TARGETS:
        sub = b[b["target"] == t]
        if sub.empty:
            continue
        base = sub[sub["model"] == "mean"]["rmse"].iloc[0]
        rows = []
        for (fs, m), g in sub.groupby(["feature_set", "model"], sort=False):
            r = g.iloc[0]
            rows.append(
                {
                    "feature set": fs,
                    "model": m,
                    "n": int(r["n_wafers"]),
                    "RMSE": r["rmse"],
                    "R²": r["r2"],
                    "RMSE / baseline": r["rmse"] / base if base else np.nan,
                    "coverage": r["coverage"],
                    "interval width": r["interval_width"],
                }
            )
        df = pd.DataFrame(rows)
        df = df[df["model"] != "mean"]
        best = df.sort_values("RMSE").iloc[0]
        res["tabular"][t] = {
            "baseline_rmse": float(base),
            "best": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in best.items()},
        }
        out += [
            f"### {t} (train-mean baseline RMSE {base:.4g})",
            "",
            md_table(df.sort_values(["feature set", "RMSE"]), 3),
            "",
        ]
    pb = OUT / "benchmark" / "point_benchmark.csv"
    if pb.exists():
        p = pd.read_csv(pb)
        res["points"] = p.set_index("model")[["rmse", "mae", "r2"]].to_dict(orient="index")
        out += [
            "### Per-site Si etch depth (89 points)",
            "",
            md_table(p[["model", "n_wafers", "n_points", "rmse", "mae", "r2", "rmse_vs_mean_baseline"]], 3),
            "",
        ]
    return out


def coverage_by_lot(res: dict) -> list[str]:
    f = OUT / "benchmark" / "tool_context" / "oof_predictions.parquet"
    if not f.exists():
        return []
    o = pd.read_parquet(f)
    rows = []
    for m in ("ridge", "gbm"):
        d = o[(o["model"] == m) & (o["target"] == "si_etch_mean")]
        cov = d.assign(c=(d["y"] >= d["lo"]) & (d["y"] <= d["hi"])).groupby("lot")["c"].mean()
        rows.append({"model": m, **{f"lot {k}": v for k, v in cov.items()}, "overall": float(cov.mean())})
    df = pd.DataFrame(rows)
    res["coverage_by_lot"] = df.set_index("model").to_dict(orient="index")
    return [
        "### Conformal coverage per held-out lot (Si etch depth, tool + context, nominal 90 %)",
        "",
        md_table(df, 2),
        "",
    ]


# --------------------------------------------------------------------------------------- nn
def nn_section(res: dict) -> list[str]:
    nn = OUT / "nn"
    groups = {
        "R3 1D-CNN (OES spectra), Colab L4": "R3_cnn1d_s",
        "R4 StatFE transformer (tool + OES), Colab L4": "R4_statfe_s",
        "R5 StatFE transformer (tool only), Colab L4": "R5_statfe_tool_s",
    }
    out, rows, comp = ["## Neural models (leave-one-lot-out, 3 seeds, Colab L4)", ""], [], []
    res["nn"] = {}
    for label, prefix in groups.items():
        runs = sorted(p for p in nn.glob(f"{prefix}*") if (p / "nn_results.csv").exists())
        if not runs:
            continue
        r = pd.concat([pd.read_csv(p / "nn_results.csv").assign(run=p.name) for p in runs])
        stats = [json.loads((p / "nn_stats.json").read_text()) for p in runs]
        train_s = [sum(f["train_seconds"] for f in s["folds"]) for s in stats]
        for t in NN_TARGETS:
            g = r[r["target"] == t]
            if g.empty:
                continue
            row = {
                "model": label,
                "target": t,
                "seeds": len(g),
                "RMSE mean": g["rmse"].mean(),
                "RMSE std": g["rmse"].std(ddof=1) if len(g) > 1 else np.nan,
                "R² mean": g["r2"].mean(),
                "R² std": g["r2"].std(ddof=1) if len(g) > 1 else np.nan,
                "coverage mean": g["coverage"].mean(),
            }
            rows.append(row)
        res["nn"][label] = {
            "n_params": stats[0].get("n_params"),
            "peak_vram_mb": max(s.get("peak_vram_mb", 0) for s in stats),
            "train_seconds_per_seed": float(np.mean(train_s)),
            "input_shape": stats[0].get("input_shape"),
            "n_wafers": stats[0].get("n_wafers"),
        }
        comp.append(
            {
                "model": label,
                "params": stats[0].get("n_params"),
                "input": "×".join(map(str, stats[0].get("input_shape", []))),
                "wafers": stats[0].get("n_wafers"),
                "train s / seed (10 folds)": float(np.mean(train_s)),
                "peak VRAM MB": max(s.get("peak_vram_mb", 0) for s in stats),
            }
        )
    if rows:
        out += [md_table(pd.DataFrame(rows), 3), "", md_table(pd.DataFrame(comp), 1), ""]
    return out


NN_PAIRS = [  # (neural run prefix, comparable tabular config on the same information)
    ("R3_cnn1d_s", "oes_spectrum·pls"),
    ("R4_statfe_s", "oes_tool_context·ridge"),
    ("R5_statfe_tool_s", "tool·ridge"),
    ("R5_statfe_tool_s", "tool_context·ridge"),
]


def nn_pairs_section(res: dict, n_boot: int = 2000) -> list[str]:
    """Lot-bootstrap paired ΔRMSE: neural 3-seed ensemble vs a tabular model given comparable inputs."""
    rows = []
    for t in NN_TARGETS:
        tab = {}
        for p in sorted((OUT / "benchmark").glob("*/oof_predictions.parquet")):
            o = pd.read_parquet(p)
            o = o[o["target"] == t]
            for (fs, m), g in o.groupby(["feature_set", "model"]):
                tab[f"{fs}·{m}"] = g.set_index("wafer_id")
        for prefix, comp in NN_PAIRS:
            runs = [
                pd.read_parquet(q / "nn_oof_predictions.parquet")
                for q in (OUT / "nn").glob(f"{prefix}*")
                if (q / "nn_oof_predictions.parquet").exists()
            ]
            if not runs or comp not in tab:
                continue
            a = pd.concat(runs)
            a = (
                a[a["target"] == t]
                .groupby("wafer_id")
                .agg(y=("y", "first"), pred=("pred", "mean"), lot=("lot", "first"))
            )
            b = tab[comp].reindex(a.index)
            y, pa, pb, lots = a["y"].to_numpy(), a["pred"].to_numpy(), b["pred"].to_numpy(), a["lot"].to_numpy()
            ul = np.unique(lots)
            rng = np.random.default_rng(0)
            ds = []
            for _ in range(n_boot):
                ii = np.concatenate([np.flatnonzero(lots == q) for q in rng.choice(ul, len(ul))])
                ds.append(rmse(y[ii], pa[ii]) - rmse(y[ii], pb[ii]))
            won = sum(rmse(y[lots == q], pa[lots == q]) < rmse(y[lots == q], pb[lots == q]) for q in ul)
            rows.append(
                {
                    "target": t,
                    "neural (3-seed ensemble)": prefix.rstrip("_s"),
                    "vs tabular": comp,
                    "RMSE neural": rmse(y, pa),
                    "RMSE tabular": rmse(y, pb),
                    "Δ [95 % CI]": f"{rmse(y, pa) - rmse(y, pb):+.3f} [{np.percentile(ds, 2.5):+.3f}, {np.percentile(ds, 97.5):+.3f}]",
                    "lots neural wins": f"{won}/{len(ul)}",
                }
            )
    if not rows:
        return []
    res["nn_pairs"] = rows
    return [
        "### Neural vs tabular on comparable inputs (lot bootstrap, Colab L4 runs)",
        "",
        md_table(pd.DataFrame(rows), 3),
        "",
    ]


def sweep_section(res: dict) -> list[str]:
    nn = OUT / "nn"
    out = []
    res["sweep"] = {}
    for fam, prefix, default in (
        ("CNN", "R6_cnn_", "R6_cnn_w32_d4_lr0.001"),
        ("StatFE", "R6_statfe_", "R6_statfe_dmpaper_l1_do0.1"),
    ):
        runs = sorted(p for p in nn.glob(f"{prefix}*") if (p / "nn_oof_predictions.parquet").exists())
        if not runs:
            continue
        rows, oof = [], {}
        for p in runs:
            o = pd.read_parquet(p / "nn_oof_predictions.parquet")
            o = o[o["target"] == "si_etch_mean"]
            oof[p.name] = o
            rows.append(
                {
                    "config": p.name.replace(prefix, ""),
                    "RMSE all lots": rmse(o["y"], o["pred"]),
                    "RMSE lots 1-5": rmse(o[o["lot"].isin(LOTS_A)]["y"], o[o["lot"].isin(LOTS_A)]["pred"]),
                    "RMSE lots 6-10": rmse(o[o["lot"].isin(LOTS_B)]["y"], o[o["lot"].isin(LOTS_B)]["pred"]),
                }
            )
        df = pd.DataFrame(rows).sort_values("RMSE all lots")
        # nested-by-lot selection: choose on one half of the lots, evaluate on the other half
        sel_a = df.sort_values("RMSE lots 1-5").iloc[0]
        sel_b = df.sort_values("RMSE lots 6-10").iloc[0]
        dflt = df[df["config"] == default.replace(prefix, "")]
        nested = {
            "select_on_1-5": sel_a["config"],
            "its_rmse_on_6-10": float(sel_a["RMSE lots 6-10"]),
            "select_on_6-10": sel_b["config"],
            "its_rmse_on_1-5": float(sel_b["RMSE lots 1-5"]),
            "default_rmse_6-10": float(dflt["RMSE lots 6-10"].iloc[0]) if len(dflt) else None,
            "default_rmse_1-5": float(dflt["RMSE lots 1-5"].iloc[0]) if len(dflt) else None,
            "best_all_lots_optimistic": float(df["RMSE all lots"].min()),
        }
        res["sweep"][fam] = {"table": df.to_dict(orient="records"), "nested": nested}
        out += [
            f"### R6 sweep — {fam} (Si etch depth, seed 0, {len(df)} configs)",
            "",
            md_table(df, 3),
            "",
            "Nested-by-lot selection (pick the config on one half of the lots, score it on the other half): "
            f"selected on lots 1–5 → `{nested['select_on_1-5']}` scores {fmt(nested['its_rmse_on_6-10'])} µm on lots "
            f"6–10 (default config: {fmt(nested['default_rmse_6-10'])}); selected on lots 6–10 → "
            f"`{nested['select_on_6-10']}` scores {fmt(nested['its_rmse_on_1-5'])} µm on lots 1–5 (default: "
            f"{fmt(nested['default_rmse_1-5'])}). The best all-lot RMSE ({fmt(nested['best_all_lots_optimistic'])}) "
            "is selection-biased and not used as a headline.",
            "",
        ]
    return ["## R6 hyper-parameter sweep (ablation)", "", *out] if out else []


# ---------------------------------------------------------------------------------- physics
def physics_section(res: dict) -> list[str]:
    fp, tp = FEAT / "wafer_features.parquet", FEAT / "wafer_targets.parquet"
    if not (fp.exists() and tp.exists()):
        return []
    f = pd.read_parquet(fp)
    t = pd.read_parquet(tp)
    d = f.join(t[["si_etch_mean", "lot"]], how="left")
    d["lot"] = d.index.str.slice(0, 10)
    cols = [
        "C_slot",
        "O_etch_F_685_mean",
        "O_dep_F_685_mean",
        "O_dep_C2_516_mean",
        "R_dep_C2_over_F_dep_mean",
        "R_dep_etch_F_dep_over_etch_mean",
        "R_etch_F703_F685_mean",
        "X_F_fall_s_mean",
        "X_F_rise_s_mean",
        "X_C2_rise_s_mean",
        "T_etch_PlatenRFLoadCapacitor_withinsd",
        "TM_tool_etch_n",
    ]
    cols = [c for c in cols if c in d]
    have_oes = d["O_etch_F_685_mean"].notna() if "O_etch_F_685_mean" in d else pd.Series(False, index=d.index)
    dd = d[have_oes].copy()
    rows = []
    for c in cols:
        x = dd[list(dict.fromkeys([c, "C_slot", "si_etch_mean", "lot"]))].dropna(subset=[c])
        dm = x.groupby("lot")[list(dict.fromkeys([c, "C_slot", "si_etch_mean"]))].transform(lambda s: s - s.mean())
        xs = dm.dropna()
        rows.append(
            {
                "feature": c,
                "n wafers": len(x),
                "r(slot), raw": np.corrcoef(x[c], x["C_slot"])[0, 1],
                "r(slot), within-lot": np.corrcoef(dm[c], dm["C_slot"])[0, 1],
                "r(Si depth), within-lot": np.corrcoef(xs[c], xs["si_etch_mean"])[0, 1] if len(xs) > 3 else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    res["physics"] = df.to_dict(orient="records")
    out = [
        "## Physics checks on all wafers",
        "",
        "Pearson correlations; *within-lot* = after subtracting each lot's mean (removes lot-to-lot offsets, keeps the "
        "drift through the lot). Si depth correlations use the 88 wafers with 89-point metrology.",
        "",
        md_table(df, 2),
        "",
    ]
    cals = [json.loads(p.read_text()) for p in sorted((STORE / "meta" / "wavelength_calibration").glob("*.json"))]
    if cals:
        c = pd.DataFrame(cals)
        res["wavelength_calibration"] = {
            "wafers": len(c),
            "shift_mean": float(c["shift_nm"].mean()),
            "shift_std": float(c["shift_nm"].std()),
            "shift_min": float(c["shift_nm"].min()),
            "shift_max": float(c["shift_nm"].max()),
            "n_matched_min": int(c["n_matched"].min()),
            "rms_max": float(c["rms_nm"].max()),
            "fallback_used": int((c["method"] == "default").sum()),
        }
        w = res["wavelength_calibration"]
        out += [
            f"Wavelength calibration over {w['wafers']} wafers: shift {w['shift_mean']:+.3f} ± {w['shift_std']:.3f} nm "
            f"(range {w['shift_min']:+.3f} … {w['shift_max']:+.3f}), ≥ {w['n_matched_min']} F I lines matched, "
            f"max residual RMS {w['rms_max']:.3f} nm, default fallback used {w['fallback_used']}×.",
            "",
        ]
    return out


def compute_section(res: dict) -> list[str]:
    st = json.loads((OUT / "campaign_state.json").read_text()) if (OUT / "campaign_state.json").exists() else {}
    rows = []
    for k, v in st.items():
        if k.startswith(("cpu:", "gpu:")):
            rows.append(
                {
                    "step": k,
                    "status": "done" if v.get("done") else f"failed ({v.get('exit')})",
                    "wall s": v.get("wall_seconds"),
                    "peak VRAM MB": v.get("peak_vram_mb"),
                }
            )
    if not rows:
        return []
    df = pd.DataFrame(rows)
    cpu = df[df["step"].str.startswith("cpu:")]["wall s"].sum()
    gpu = df[df["step"].str.startswith("gpu:")]["wall s"].sum()
    res["compute"] = {"cpu_wall_s": float(cpu), "gpu_wall_s": float(gpu), "steps": len(df)}
    return [
        "## Compute actually used",
        "",
        f"CPU stages {cpu / 60:.1f} min; local GPU jobs {gpu / 60:.1f} min wall (0 if the neural runs were done on "
        "Colab; see REPORT.md, compute used).",
        "",
        md_table(df, 1),
        "",
    ]


# ----------------------------------------------------------------------------------- figures
KEY_CONFIGS = [
    ("slot_cycles_cond·ols", "OLS slot + cycles + conditioning (baseline)"),
    ("tool_context·ridge", "ridge tool + context (pre-declared, served)"),
    ("tool_context·gbm", "GBM tool + context"),
    ("oes_tool_context·gbm", "GBM OES + tool + context"),
    ("oes_physics·gbm", "GBM OES physics features only"),
    ("oes_spectrum·pls", "PLS OES binned spectra only"),
    ("hybrid_rate_tool_context·gbm", "hybrid: cycles × GBM rate(tool + context)"),
    ("nested_selection", "nested-by-lot selection over 22 configs"),
]
NN_KEYS = [
    ("R3_cnn1d_s", "1D-CNN on OES spectra (3 seeds, Colab L4)"),
    ("R4_statfe_s", "StatFE tool + OES (3 seeds, Colab L4)"),
    ("R5_statfe_tool_s", "StatFE tool only (3 seeds, Colab L4)"),
]


def key_config_bootstrap(n_boot: int = 2000, seed: int = 0) -> list[dict]:
    """RMSE and lot-bootstrap 95 % CI for the key Si-depth configs (same lot resamples for all)."""
    preds = {}
    for p in sorted((OUT / "benchmark").glob("*/oof_predictions.parquet")):
        o = pd.read_parquet(p)
        o = o[o["target"] == "si_etch_mean"]
        for (fs, m), g in o.groupby(["feature_set", "model"]):
            preds[f"{fs}·{m}"] = g.set_index("wafer_id")[["y", "pred", "lot"]]
    nf = OUT / "review" / "nested_cvplus_oof.parquet"
    if nf.exists():
        n = pd.read_parquet(nf)
        for c, g in n.groupby("config"):
            preds.setdefault(c, g.set_index("wafer_id")[["y", "pred", "lot"]])
    for prefix, _ in NN_KEYS:
        runs = [
            pd.read_parquet(p / "nn_oof_predictions.parquet")
            for p in (OUT / "nn").glob(f"{prefix}*")
            if (p / "nn_oof_predictions.parquet").exists()
        ]
        if runs:
            a = pd.concat(runs)
            a = (
                a[a["target"] == "si_etch_mean"]
                .groupby("wafer_id")
                .agg(y=("y", "first"), pred=("pred", "mean"), lot=("lot", "first"))
            )
            preds[prefix] = a
    labels = dict(KEY_CONFIGS) | dict(NN_KEYS)
    keys = [k for k in labels if k in preds]
    if not keys:
        return []
    ref = preds[keys[0]]
    wafers = ref.index
    lots = ref["lot"].reindex(wafers).to_numpy()
    ul = np.unique(lots)
    rng = np.random.default_rng(seed)
    picks = [np.concatenate([np.flatnonzero(lots == q) for q in rng.choice(ul, len(ul))]) for _ in range(n_boot)]
    rows = []
    for k in keys:
        d = preds[k].reindex(wafers)
        y, pr = d["y"].to_numpy(float), d["pred"].to_numpy(float)
        bs = [rmse(y[ii], pr[ii]) for ii in picks]
        rows.append(
            {
                "config": k,
                "label": labels[k],
                "rmse": rmse(y, pr),
                "lo": float(np.percentile(bs, 2.5)),
                "hi": float(np.percentile(bs, 97.5)),
                "n": int(np.isfinite(pr).sum()),
            }
        )
    return rows


def figures(res: dict) -> None:
    import render_figures as rf

    from plasma_etch_vm.dashboard import figures as F

    b = OUT / "benchmark" / "benchmark_all.csv"
    if not b.exists():
        return
    import plotly.graph_objects as go

    bench = pd.read_csv(b)
    base = float(bench[(bench["target"] == "si_etch_mean") & (bench["model"] == "mean")]["rmse"].iloc[0])
    cmp_rows = key_config_bootstrap()
    res["key_configs_si_etch"] = cmp_rows
    if cmp_rows:
        d = pd.DataFrame(cmp_rows).sort_values("rmse", ascending=False)
        fig = go.Figure(
            go.Bar(
                x=d["rmse"],
                y=d["label"],
                orientation="h",
                marker_color=F.BLUE,
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=d["hi"] - d["rmse"],
                    arrayminus=d["rmse"] - d["lo"],
                    color=F.MUTED,
                    thickness=1,
                ),
                hovertemplate="%{y}<br>RMSE %{x:.3f} µm<extra></extra>",
            )
        )
        fig.add_vline(x=base, line=dict(color=F.MUTED, dash="dot", width=1))
        fig.add_annotation(
            x=base,
            y=1,
            yref="y domain",
            text="train-mean baseline",
            showarrow=False,
            xanchor="right",
            yanchor="top",
            textangle=-90,
            font=dict(size=10, color=F.MUTED),
        )
        F.style(
            fig,
            "Si etch depth — leave-one-lot-out RMSE with lot-bootstrap 95 % CI (88 wafers, 10 lots)",
            "RMSE [µm]",
            "",
            max(420, 30 * len(d) + 90),
            static=True,
        )
        rf.save(fig, "model_comparison_si_etch", True)
    ph = res.get("physics")
    fp = FEAT / "wafer_features.parquet"
    if ph and fp.exists():
        f = pd.read_parquet(fp)
        col = "R_dep_etch_F_dep_over_etch_mean"  # log[F I (C4F8 step) / F I (SF6 step)], same line -> window-free
        if col in f:
            d = f[[col, "C_slot"]].dropna()
            d["lot"] = d.index.str.slice(0, 10)
            d["F_within"] = d[col] - d.groupby("lot")[col].transform("mean")
            fig = F.feature_scatter(
                d["C_slot"],
                d["F_within"],
                "slot in lot (wafers since conditioning)",
                "log F I(C₄F₈ step)/F I(SF₆ step), lot-demeaned",
                static=True,
                color_by_slot=False,
            )
            F.style(
                fig,
                "Window-independent fluorine ratio falls through every lot (96 wafers, 10 lots)",
                "slot in lot (wafers since conditioning)",
                "log F I(C₄F₈)/F I(SF₆), lot-demeaned",
                420,
                static=True,
            )
            rf.save(fig, "f_line_vs_slot", True, 800)


def main() -> None:
    SUM.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    parts = ["# Results summary (generated by scripts/summarize_results.py)", ""]
    for fn in (
        data_section,
        tabular_section,
        coverage_by_lot,
        nn_section,
        nn_pairs_section,
        sweep_section,
        physics_section,
        compute_section,
    ):
        try:
            parts += fn(res)
        except Exception as exc:  # keep going; report what failed
            parts += [f"> section {fn.__name__} failed: {exc}", ""]
    try:
        figures(res)
    except Exception as exc:
        parts += [f"> figures failed: {exc}", ""]
    (SUM / "summary.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    (SUM / "summary.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"wrote {SUM / 'summary.md'}")


if __name__ == "__main__":
    main()
