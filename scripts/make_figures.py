"""Static figures for README.md / REPORT.md, drawn with matplotlib from saved outputs (no training).

    python scripts/make_figures.py        -> docs/figures/*.png

Inputs (all produced by the pipeline): outputs/review/nested_cvplus_oof.parquet (out-of-fold predictions and
nested CV+ intervals), outputs/summary/summary.json (lot-bootstrap CIs of the key configs),
outputs/benchmark/benchmark_all.csv (train-mean baseline), outputs/benchmark/point_oof.parquet (89-point maps),
outputs/interpret/feature_importance.csv, data/features/*.parquet, data/parquet/oes_trace/.

Style: white background, light grid, colour-blind-safe Okabe-Ito colours per model family, lots coloured by
their chronological order on one sequential (viridis) scale — the same colour per lot in every panel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plasma_etch_vm.lines import LINES  # noqa: E402
from plasma_etch_vm.metrology import DATE_TO_LOT  # noqa: E402

OUT, FEAT, FIG = ROOT / "outputs", ROOT / "data" / "features", ROOT / "docs" / "figures"
DPI = 170
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e5e5e5"
# Okabe-Ito (colour-blind safe); one colour per model family everywhere
FAMILY = {
    "baseline": "#7a7a7a",
    "linear": "#0072B2",
    "tree": "#E69F00",
    "hybrid": "#009E73",
    "neural": "#CC79A7",
}
PHASE = {"etch": "#0072B2", "dep": "#D55E00"}  # SF6 etch step / C4F8 passivation step
HYBRID = "hybrid_rate_tool_context·gbm"
F_RATIO = "R_dep_etch_F_dep_over_etch_mean"  # log[F I(C4F8 step) / F I(SF6 step)], same line -> window-free

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#9a9a9a",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.labelcolor": INK,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
    }
)


def lot_colour(lot: int) -> tuple:
    return plt.get_cmap("viridis")(0.02 + 0.86 * (int(lot) - 1) / 9)


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{name}.png {(FIG / f'{name}.png').stat().st_size / 1024:.0f} KB")


# ------------------------------------------------------------------------------------------- data
def load() -> dict:
    oof = pd.read_parquet(OUT / "review" / "nested_cvplus_oof.parquet")
    hyb = oof[oof["config"] == HYBRID].set_index("wafer_id")
    feats = pd.read_parquet(FEAT / "wafer_features.parquet", columns=["C_slot", F_RATIO])
    hyb = hyb.join(feats["C_slot"])
    summ = json.loads((OUT / "summary" / "summary.json").read_text(encoding="utf-8"))
    bench = pd.read_csv(OUT / "benchmark" / "benchmark_all.csv")
    train_mean = float(bench[(bench["target"] == "si_etch_mean") & (bench["model"] == "mean")]["rmse"].iloc[0])
    f = feats.dropna().copy()
    f["lot"] = [DATE_TO_LOT[i[:10]] for i in f.index]
    f["F_within"] = f[F_RATIO] - f.groupby("lot")[F_RATIO].transform("mean")
    return {"hyb": hyb, "keys": summ["key_configs_si_etch"], "train_mean": train_mean, "fratio": f}


# key configs -> (short label, family); order of the summary's lot-bootstrap table
KEY_LABELS = {
    "slot_cycles_cond·ols": ("OLS slot+cycles+cond. (baseline)", "baseline"),
    "tool_context·ridge": ("ridge, tool+context (served)", "linear"),
    "oes_spectrum·pls": ("PLS, OES spectra only", "linear"),
    "tool_context·gbm": ("GBM, tool+context", "tree"),
    "oes_tool_context·gbm": ("GBM, OES+tool+context", "tree"),
    "oes_physics·gbm": ("GBM, OES lines only", "tree"),
    HYBRID: ("hybrid n_cycles × GBM rate", "hybrid"),
    "nested_selection": ("nested selection (22 configs)", "hybrid"),
    "R3_cnn1d_s": ("1D-CNN, OES spectra (3 seeds)", "neural"),
    "R4_statfe_s": ("StatFE, tool+OES (3 seeds)", "neural"),
    "R5_statfe_tool_s": ("StatFE, tool only (3 seeds)", "neural"),
}


def key_table(keys: list[dict], with_nested: bool) -> pd.DataFrame:
    rows = []
    for k in keys:
        if k["config"] not in KEY_LABELS or (k["config"] == "nested_selection" and not with_nested):
            continue
        label, fam = KEY_LABELS[k["config"]]
        rows.append({"label": label, "family": fam, "rmse": k["rmse"], "lo": k["lo"], "hi": k["hi"]})
    return pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------------------- panels
def panel_drift(ax, d: dict) -> None:
    h = d["hyb"]
    for lot, g in h.groupby("lot"):
        g = g.sort_values("C_slot")
        c = lot_colour(lot)
        ax.plot(g["C_slot"], g["y"], "-", color=c, lw=0.9, alpha=0.85)
        ax.plot(g["C_slot"], g["y"], "o", color=c, ms=3.6, mec="white", mew=0.5)
        ax.plot(g["C_slot"], g["pred"], "x", color=c, ms=3.6, mew=0.9)
    ax.set_xlabel("slot in lot (wafers since chamber conditioning)")
    ax.set_ylabel("Si etch depth [µm]")
    ax.set_xticks(range(1, 11))
    ax.set_title("(a) Depth falls wafer by wafer after every conditioning")
    handles = [
        Line2D([], [], color=INK, marker="o", ls="-", lw=0.9, ms=3.6, label="measured (line per lot)"),
        Line2D([], [], color=INK, marker="x", ls="", ms=4, label="hybrid, out-of-fold prediction"),
    ]
    ax.legend(handles=handles, loc="lower left")
    lot_key(ax)


def lot_key(ax, loc=(0.985, 0.97)) -> None:
    """Compact colour key 'lot 1 ... 10' (chronological) in the upper-right corner."""
    x0, y0 = loc
    ax.text(x0, y0, "lot", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=MUTED)
    for i in range(10):
        ax.annotate(
            str(i + 1),
            xy=(x0, y0),
            xycoords="axes fraction",
            xytext=(-13 * (9 - i), -11),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=7,
            color=lot_colour(i + 1),
            fontweight="bold",
        )


def panel_models(ax, d: dict, with_nested: bool = False) -> None:
    t = key_table(d["keys"], with_nested)
    y = np.arange(len(t))
    ax.barh(y, t["rmse"], color=[FAMILY[f] for f in t["family"]], height=0.66, zorder=2)
    ax.errorbar(
        t["rmse"],
        y,
        xerr=[t["rmse"] - t["lo"], t["hi"] - t["rmse"]],
        fmt="none",
        ecolor=INK,
        elinewidth=0.8,
        capsize=2,
        zorder=3,
    )
    for yi, r in zip(y, t.itertuples()):
        ax.text(r.hi + 0.006, yi, f"{r.rmse:.3f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, t["label"], fontsize=7.8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    tm = d["train_mean"]
    ax.axvline(tm, color=MUTED, ls="--", lw=0.9)
    ax.text(
        tm + 0.006,
        len(t) / 2,
        f"predict train mean ({tm:.3f})",
        rotation=90,
        ha="left",
        va="center",
        fontsize=7,
        color=MUTED,
    )
    ax.set_xlim(0, tm * 1.12)
    ax.set_xlabel("Si depth RMSE [µm], each lot held out (bar: 95 % lot-bootstrap CI)")
    ax.set_title("(b) Physics-structured hybrid has the lowest error")
    fams = [f for f in FAMILY if f in set(t["family"])]
    ax.legend(
        handles=[Patch(color=FAMILY[f], label=f) for f in fams],
        loc="upper right",
        bbox_to_anchor=(0.86, 1.0),
        title="model family",
        title_fontsize=7.5,
    )


def panel_pred(ax, d: dict) -> None:
    h = d["hyb"]
    y, p = h["y"].to_numpy(), h["pred"].to_numpy()
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    cov = float(np.mean((h["cvplus_lo"] <= y) & (y <= h["cvplus_hi"])))
    ax.vlines(y, h["cvplus_lo"], h["cvplus_hi"], color="#b9b9b9", lw=0.8, zorder=1)
    ax.scatter(y, p, c=[lot_colour(v) for v in h["lot"]], s=16, edgecolor="white", linewidth=0.4, zorder=3)
    lo, hi = min(y.min(), h["cvplus_lo"].min()) - 0.05, max(y.max(), h["cvplus_hi"].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, ls="--", zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("measured Si etch depth [µm]")
    ax.set_ylabel("predicted [µm] (grey: 90 % CV+ interval)")
    ax.set_title("(c) Hybrid predictions on held-out lots")
    ax.text(
        0.03,
        0.97,
        f"RMSE {rmse:.3f} µm\nR² {r2:.2f}\n90 % interval coverage {cov:.2f}\n{len(y)} wafers, 10 held-out lots",
        transform=ax.transAxes,
        va="top",
        fontsize=7.8,
        color=INK,
    )
    ax.text(lo + 0.12, lo + 0.06, "y = x", ha="left", va="bottom", fontsize=7, color=MUTED, rotation=45)
    lot_key(ax, (0.985, 0.12))


def panel_fratio(ax, d: dict) -> None:
    f = d["fratio"]
    for lot, g in f.groupby("lot"):
        g = g.sort_values("C_slot")
        ax.plot(g["C_slot"], g["F_within"], "-o", color=lot_colour(lot), lw=0.8, ms=2.6, alpha=0.9)
    m = f.groupby("C_slot")["F_within"].mean()
    ax.plot(m.index, m.values, color=INK, lw=2.2, label="mean over lots")
    slot_within = f["C_slot"] - f.groupby("lot")["C_slot"].transform("mean")
    r = float(np.corrcoef(slot_within, f["F_within"])[0, 1])
    ax.text(
        0.97,
        0.97,
        f"within-lot r = {r:.2f}\n{len(f)} wafers, 10 lots",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        color=INK,
    )
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("slot in lot (wafers since chamber conditioning)")
    ax.set_ylabel("log F I(C₄F₈ step) / F I(SF₆ step)\nminus lot mean")
    ax.set_title("(d) Window-independent F-line ratio falls in every lot")
    ax.legend(loc="lower left")


def importance(d_unused=None) -> None:
    imp = pd.read_csv(OUT / "interpret" / "feature_importance.csv")
    di = imp[(imp["target"] == "si_etch_mean") & (imp["model"] == "gbm") & (imp["method"] == "perm_lolo")]
    di = di.sort_values("importance_mean", ascending=False).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(di))
    ax.barh(y, di["importance_mean"], color=FAMILY["tree"], height=0.62, zorder=2)
    ax.errorbar(di["importance_mean"], y, xerr=di["importance_std"], fmt="none", ecolor=INK, elinewidth=0.7, capsize=2)
    ax.set_yticks(y, [pretty(c) for c in di["feature"]], fontsize=7.8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(left=min(0, float((di["importance_mean"] - di["importance_std"]).min()) * 1.1))
    ax.set_xlabel("increase in Si-depth RMSE when the feature is permuted [µm] (mean ± SD over 10 held-out lots)")
    ax.set_title("Wafer slot and platen RF re-tuning drive the GBM depth model")
    save(fig, "importance_si_etch")


def pretty(c: str) -> str:
    names = {
        "C_slot": "slot in lot",
        "C_log_slot": "log(slot)",
        "TM_tool_dep_n": "number of C₄F₈ steps logged",
        "TM_tool_etch_n": "number of SF₆ steps logged",
    }
    if c in names:
        return names[c]
    step = {"T_etch_": "SF₆ step", "T_dep_": "C₄F₈ step"}
    stat = {"_withinsd": "within-step SD", "_std": "SD", "_mean": "mean", "_d": "first difference"}
    for pre, s in step.items():
        if c.startswith(pre):
            body = c[len(pre) :]
            for suf, st in stat.items():
                if body.endswith(suf):
                    return f"{body[: -len(suf)]} ({s}, {st})"
            return f"{body} ({s})"
    return c


def oes_figures() -> None:
    spec = pd.read_parquet(FEAT / "wafer_spectra.parquet")
    wl = pd.read_parquet(FEAT / "wavelengths.parquet")["wavelength_nm"].to_numpy()
    ch = [c for c in spec.columns if c.startswith("ch")]
    first = spec.iloc[0]
    sel = (spec["day"] == first["day"]) & (spec["wafer"] == first["wafer"])
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    win = (wl >= 500) & (wl <= 790)
    top = 0.0
    for ph, lab in (("etch", "SF₆ etch step"), ("dep", "C₄F₈ passivation step")):
        s = spec[sel & (spec["phase"] == ph)][ch].to_numpy()[0]
        ax.plot(wl[win], s[win], color=PHASE[ph], lw=0.8, label=lab)
        top = max(top, float(np.nanmax(s[win])))
    for ln in LINES:
        if ln.name in ("F_624", "F_685", "F_703", "F_739", "F_775", "C2_516", "C2_563"):
            ax.axvline(ln.wl_nm, color=MUTED, lw=0.5, ls=":", zorder=0)
            short = "C₂" if ln.name.startswith("C2") else "F I"
            ax.text(ln.wl_nm, top * 1.03, short, ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_ylim(top=top * 1.12)
    ax.set_xlim(500, 790)
    ax.set_xlabel("wavelength [nm] (recalibrated axis)")
    ax.set_ylabel("mean intensity [counts]")
    ax.set_title(f"Only F I and C₂ Swan emission stand out (wafer {first['day']} #{int(first['wafer'])})")
    ax.legend(loc="center right")
    save(fig, "oes_spectrum")

    tr_file = next((ROOT / "data" / "parquet" / "oes_trace").glob("day=*/lot=*/wafer=1/part-0.parquet"))
    tr = pd.read_parquet(tr_file)
    t = (tr["t_s"] - tr["t_s"].iloc[0]).to_numpy()
    m = (t >= 30) & (t <= 50)
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    for col, lab, c in (("L_F_685", "F I 685.6 nm", PHASE["etch"]), ("L_C2_516", "C₂ Swan 516.5 nm", PHASE["dep"])):
        v = tr[col].to_numpy()[m]
        ax.plot(t[m], v / np.nanmax(v), color=c, lw=0.9, label=lab)
    ax.set_xlabel("time since plasma start [s]")
    ax.set_ylabel("line intensity / max in window")
    ax.set_title("BOSCH cycling in the OES: 4.5 s SF₆ etch and 1.5 s C₄F₈ passivation alternate")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.22))
    save(fig, "oes_trace")


def wafer_map() -> None:
    pts = pd.read_parquet(OUT / "benchmark" / "point_oof.parquet")
    wid = pts["wafer_id"].iloc[0]
    p = pts[pts["wafer_id"] == wid]
    lo, hi = float(pts["si_etch"].quantile(0.01)), float(pts["si_etch"].quantile(0.99))
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.add_patch(Circle((0, 0), 100, fill=False, color=MUTED, lw=0.8))
    sc = ax.scatter(p["x_mm"], p["y_mm"], c=p["si_etch"], cmap="viridis", vmin=lo, vmax=hi, s=46, edgecolor="none")
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Si etch depth [µm]", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    ax.set_xlim(-105, 105)
    ax.set_ylim(-105, 105)
    ax.grid(False)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Edge-fast etch: 89-point map of wafer {wid}", fontsize=9.5)
    save(fig, "wafer_map")


def main() -> None:
    d = load()
    fig, axs = plt.subplots(2, 2, figsize=(13.0, 9.6), gridspec_kw={"hspace": 0.34, "wspace": 0.5})
    panel_drift(axs[0, 0], d)
    panel_models(axs[0, 1], d)
    panel_pred(axs[1, 0], d)
    panel_fratio(axs[1, 1], d)
    fig.suptitle(
        "Virtual metrology of Si etch depth — BOSCH plasma etch, 88 labelled wafers in 10 lots, each lot held out",
        x=0.07,
        ha="left",
        fontsize=12,
        fontweight="bold",
        y=0.965,
    )
    save(fig, "hero")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    panel_models(ax, d, with_nested=True)
    ax.set_title("Si etch depth: leave-one-lot-out RMSE of every key model (88 wafers, 10 lots)")
    save(fig, "model_comparison_si_etch")

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    panel_fratio(ax, d)
    ax.set_title("Window-independent F-line ratio falls through every lot")
    lot_key(ax, (0.985, 0.82))
    save(fig, "f_line_vs_slot")

    importance()
    oes_figures()
    wafer_map()


if __name__ == "__main__":
    main()
