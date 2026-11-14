"""Plasma-etch virtual-metrology dashboard.

    streamlit run src/plasma_etch_vm/dashboard/app.py

Tabs: predicted vs. measured (with conformal intervals), top features, OES line viewer
(calibrated spectra, 24 Hz traces, per-cycle drift), 89-point wafer maps, pipeline status.
Reads only local artefacts (outputs/, data/features/, data/parquet/oes_trace); every tab degrades
gracefully if its inputs have not been produced yet. CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from plasma_etch_vm.dashboard import figures as F
from plasma_etch_vm.lines import LINES

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs"
FEAT = ROOT / "data" / "features"
STORE = ROOT / "data" / "parquet"

st.set_page_config(page_title="Plasma-etch VM", layout="wide")


@st.cache_data
def read_parquet(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def read_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def oof_table() -> pd.DataFrame | None:
    frames = [read_parquet(str(f)) for f in sorted((OUT / "benchmark").glob("*/oof_predictions.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else None


def pick(label: str, options: list, preferred=None, fmt=None, container=st):
    idx = options.index(preferred) if preferred in options else 0
    return container.selectbox(label, options, index=idx, format_func=fmt or str)


st.title("Plasma-etch virtual metrology")
st.caption(
    "BOSCH Si etch on an SPTS Rapier (Zenodo 17122442) — tool traces + OES → wafer metrology, "
    "leave-one-lot-out CV, split-conformal intervals."
)

oof = oof_table()
feats = read_parquet(str(FEAT / "wafer_features.parquet"))
tabs = st.tabs(["Predicted vs measured", "Top features", "OES line viewer", "Wafer map", "Pipeline"])

# ---------------------------------------------------------------------------- predicted vs measured
with tabs[0]:
    if oof is None:
        st.info("Run `pevm benchmark` to produce out-of-fold predictions.")
    else:
        c1, c2, c3 = st.columns(3)
        fs = pick("Feature set", sorted(oof["feature_set"].unique()), "tool_context", container=c1)
        sub = oof[oof["feature_set"] == fs]
        model = pick("Model", sorted(sub["model"].unique()), "ridge", container=c2)
        tgt = pick("Target", sorted(sub["target"].unique()), "si_etch_mean", lambda t: F.LABELS.get(t, t), c3)
        d = sub[(sub["model"] == model) & (sub["target"] == tgt)].dropna(subset=["pred"])
        u = F.UNITS.get(tgt, "")
        resid = d["y"] - d["pred"]
        ss = np.sum((d["y"] - d["y"].mean()) ** 2)
        k = st.columns(5)
        k[0].metric("RMSE", f"{np.sqrt(np.mean(resid**2)):.3g} {u}")
        k[1].metric("MAE", f"{np.mean(np.abs(resid)):.3g} {u}")
        k[2].metric("R² (out-of-fold)", f"{1 - np.sum(resid**2) / ss:.2f}" if ss > 0 else "n/a")
        k[3].metric("90 % interval coverage", f"{((d['y'] >= d['lo']) & (d['y'] <= d['hi'])).mean():.0%}")
        k[4].metric("Wafers / lots", f"{len(d)} / {d['lot'].nunique()}")
        st.plotly_chart(F.pred_vs_measured(d, tgt), width="stretch")
        if d["lot"].nunique() < 3:
            st.warning("Fewer than 3 lots: CV fell back to leave-one-wafer-out (smoke test, not leakage-safe).")
        bench = read_csv(str(OUT / "benchmark" / "benchmark_all.csv"))
        if bench is not None:
            with st.expander("Full benchmark table (all feature sets / models)"):
                st.dataframe(bench[bench["target"] == tgt].round(4), width="stretch", hide_index=True)

# ------------------------------------------------------------------------------------ top features
with tabs[1]:
    imp = read_csv(str(OUT / "interpret" / "feature_importance.csv"))
    if imp is None:
        st.info("Run `pevm interpret` to compute permutation importance / SHAP.")
    else:
        c1, c2, c3 = st.columns(3)
        tgt = pick("Target ", sorted(imp["target"].unique()), "si_etch_mean", lambda t: F.LABELS.get(t, t), c1)
        mdl = pick("Model ", sorted(imp["model"].unique()), "gbm", container=c2)
        methods = sorted(imp[(imp["target"] == tgt) & (imp["model"] == mdl)]["method"].unique())
        meth = pick("Method", methods, "perm_lolo", container=c3)
        d = imp[(imp["target"] == tgt) & (imp["model"] == mdl) & (imp["method"] == meth)].head(15)
        xlab = "Δ RMSE when permuted (held-out lots)" if meth == "perm_lolo" else "mean |SHAP| (full fit)"
        st.plotly_chart(F.importance_bar(d, f"Top features — {F.LABELS.get(tgt, tgt)}", xlab), width="stretch")
        tg = read_parquet(str(FEAT / "wafer_targets.parquet"))
        if feats is not None and tg is not None and len(d):
            fsel = st.selectbox("Inspect feature vs target", d["feature"].tolist())
            j = feats[[fsel]].join(tg[[tgt]], how="inner").dropna()
            ylab = f"{F.LABELS.get(tgt, tgt)} [{F.UNITS.get(tgt, '')}]"
            st.plotly_chart(F.feature_scatter(j[fsel], j[tgt], fsel, ylab), width="stretch")
        st.markdown(
            "**Physics reading.** Depth falls −0.11 … −0.16 µm per wafer through 9 of 10 lots (no clean between "
            "wafers). The window-independent ratio F I (C₄F₈ step) / F I (SF₆ step) falls with it (within-lot "
            "r = −0.89 with slot, +0.79 with depth), all absolute emission falls, and the platen RF match "
            "capacitors re-tune harder within each step — a drifting chamber state consistent with fluorocarbon "
            "build-up on the walls (viewport coating cannot be excluded for absolute intensities; C₂ rises only as "
            "a share of total emission). Lot 8 had long idle times before wafers 8–9 and breaks the slot pattern."
        )

# ---------------------------------------------------------------------------------- OES viewer
with tabs[2]:
    spec = read_parquet(str(FEAT / "wafer_spectra.parquet"))
    wl = read_parquet(str(FEAT / "wavelengths.parquet"))
    cyc = read_parquet(str(FEAT / "cycles_oes.parquet"))
    if spec is None or wl is None:
        st.info("Run `pevm ingest` and `pevm cycles` with at least one day file to populate the OES viewer.")
    else:
        spec = spec.assign(wafer_id=spec["day"] + "_" + spec["wafer"].astype(int).map("{:02d}".format))
        wid = st.selectbox("Wafer", sorted(spec["wafer_id"].unique()))
        ch = [c for c in spec.columns if c.startswith("ch")]
        x = wl["wavelength_nm"].to_numpy()
        c1, c2 = st.columns([3, 1])
        rng = c2.slider("Wavelength range [nm]", 186, 884, (500, 790))
        show = c2.multiselect(
            "Mark lines",
            [ln.name for ln in LINES],
            default=["F_624", "F_685", "F_703", "F_739", "F_775", "C2_516", "C2_563"],
        )
        rows = spec[spec["wafer_id"] == wid]
        sp = {ph: rows[rows["phase"] == ph][ch].to_numpy()[0] for ph in ("etch", "dep") if (rows["phase"] == ph).any()}
        marks = [(ln.name, ln.wl_nm) for ln in LINES if ln.name in show]
        c1.plotly_chart(F.spectrum(x, sp, marks, rng), width="stretch")
        day, w = wid[:10], int(wid[11:])
        tr_files = list((STORE / "oes_trace").glob(f"day={day}/lot=*/wafer={w}/part-0.parquet"))
        if tr_files:
            tr = read_parquet(str(tr_files[0]))
            t = (tr["t_s"] - tr["t_s"].iloc[0]).to_numpy()
            t0 = st.slider("Trace window start [s]", 0, int(t.max()) - 30, 30)
            mm = (t >= t0) & (t <= t0 + 30)
            st.plotly_chart(
                F.line_trace(t[mm], tr["L_F_685"].to_numpy()[mm], tr["L_C2_516"].to_numpy()[mm]), width="stretch"
            )
        if cyc is not None:
            avail = [c[2:] for c in cyc.columns if c.startswith("L_")]
            sel = pick("Per-cycle drift of line", avail, "F_685")
            g = cyc[(cyc["day"] == day) & (cyc["wafer"] == w) & (cyc["cycle"] >= 1)]
            st.plotly_chart(F.cycle_drift(g, sel), width="stretch")
        det = ROOT / "docs" / "line_detection.md"
        if det.exists():
            with st.expander("Line detection & wavelength calibration report"):
                st.markdown(det.read_text(encoding="utf-8"))

# ------------------------------------------------------------------------------------ wafer map
with tabs[3]:
    pts = read_parquet(str(OUT / "benchmark" / "point_oof.parquet"))
    if pts is None:
        st.info("Run `pevm points` for the 89-point per-site benchmark.")
    else:
        wid = st.selectbox("Wafer ", sorted(pts["wafer_id"].unique()))
        pred_cols = [c for c in pts.columns if c.startswith("pred_")]
        model_col = pick("Point model", pred_cols, "pred_ridge_x_shape", lambda c: c[5:])
        p = pts[pts["wafer_id"] == wid]
        vmin, vmax = float(pts["si_etch"].quantile(0.01)), float(pts["si_etch"].quantile(0.99))
        cols = st.columns(3)
        cols[0].plotly_chart(
            F.wafer_map(p, p["si_etch"], "Measured Si etch depth [µm]", F.SEQ_BLUE, (vmin, vmax)), width="stretch"
        )
        cols[1].plotly_chart(
            F.wafer_map(p, p[model_col], f"Predicted ({model_col[5:]}) [µm]", F.SEQ_BLUE, (vmin, vmax)),
            width="stretch",
        )
        cols[2].plotly_chart(
            F.wafer_map(p, p["si_etch"] - p[model_col], "Residual measured − predicted [µm]", F.DIVERGING, (-1.5, 1.5)),
            width="stretch",
        )
        pb = read_csv(str(OUT / "benchmark" / "point_benchmark.csv"))
        if pb is not None:
            st.dataframe(pb.round(4), hide_index=True, width="stretch")

# ------------------------------------------------------------------------------------- pipeline
with tabs[4]:
    s = OUT / "ingest" / "ingest_summary.json"
    if s.exists():
        st.subheader("Ingestion (netCDF → partitioned Parquet)")
        st.json(json.loads(s.read_text()))
    q = OUT / "ingest" / "query_benchmark.json"
    if q.exists():
        st.subheader("Query layer (DuckDB / Polars)")
        st.json(json.loads(q.read_text()))
    stats = read_csv(str(OUT / "ingest" / "ingest_stats.csv"))
    if stats is not None:
        st.dataframe(stats, hide_index=True, width="stretch")
    card = OUT / "model_card.json"
    if card.exists():
        st.subheader("Served model")
        c = json.loads(card.read_text(encoding="utf-8"))
        st.json({k: v for k, v in c.items() if k != "feature_columns"})
