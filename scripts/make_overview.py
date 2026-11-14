"""16:9 overview visual for the README / project card: sensors in -> predicted depth out -> checked on every wafer.

    python scripts/make_overview.py       -> docs/figures/overview.png (1600 x 900 px)

Drawn from saved outputs only (no training): one wafer's OES spectra (data/parquet/oes) and tool log
(data/parquet/process), its 89-point depth map (outputs/benchmark/point_oof.parquet), and the hybrid model's
out-of-fold predictions with nested CV+ intervals (outputs/review/nested_cvplus_oof.parquet).
Built to stay legible when shown ~400 px wide: >= 18 pt text, three panels, few ticks.
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
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image
from scipy.interpolate import RBFInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_figures import FAMILY, HYBRID, INK, MUTED, PHASE  # noqa: E402

OUT, STORE, FIG = ROOT / "outputs", ROOT / "data" / "parquet", ROOT / "docs" / "figures"
WAFER = "2024-07-09_05"  # lot 3, slot 5: error close to the median, interval covers the measurement
WINDOW_S = 18.0  # three BOSCH cycles
FS = 18  # minimum font size (pt at 1600 px width)


def wafer_paths(wid: str) -> tuple[str, int]:
    day, w = wid[:10], int(wid[-2:])
    return day, w


def oes_window(wid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spectra of one wafer in an 18 s window that starts at an SF6 etch step ~30 s after plasma start."""
    day, w = wafer_paths(wid)
    tr = pd.read_parquet(next((STORE / "oes_trace").glob(f"day={day}/lot=*/wafer={w}/part-0.parquet")))
    on = tr[tr["phase"] != "off"]
    t0 = float(on["t_s"].iloc[0])
    is_etch = (tr["phase"] == "etch").to_numpy()
    starts = tr["t_s"].to_numpy()[1:][is_etch[1:] & ~is_etch[:-1]]
    ws = float(starts[starts >= t0 + 30][0])
    cal = json.loads((STORE / "meta" / "wavelength_calibration" / f"{wid}.json").read_text())
    stored = pd.read_parquet(STORE / "meta" / "wavelengths.parquet")["wavelength_nm"].to_numpy()  # as stored
    wl = cal["a"] + cal["b"] * stored
    keep = np.flatnonzero((wl >= 480) & (wl <= 800))
    cols = ["t_s"] + [f"ch{i:04d}" for i in keep]
    parts = [
        pd.read_parquet(p, columns=cols) for p in (STORE / "oes").glob(f"day={day}/lot=*/wafer={w}/step=*/*.parquet")
    ]
    s = pd.concat(parts).sort_values("t_s")
    s = s[(s["t_s"] >= ws) & (s["t_s"] < ws + WINDOW_S)]
    spec = s[cols[1:]].to_numpy(float).copy()
    spec -= np.percentile(spec, 20, axis=1, keepdims=True)  # per-spectrum baseline (detector jitter)
    spec = pd.DataFrame(spec).rolling(3, center=True, min_periods=1).mean().to_numpy()  # light time smoothing
    return (s["t_s"].to_numpy() - ws), wl[keep], spec


def tool_window(wid: str) -> pd.DataFrame:
    """Gas flows from the tool log (its own clock), 18 s starting at an SF6 step ~30 s after the log starts."""
    day, w = wafer_paths(wid)
    parts = [
        pd.read_parquet(p, columns=["t_s", "Gas4Flow", "Gas5Flow"])
        for p in (STORE / "process").glob(f"day={day}/lot=*/wafer={w}/step=*/*.parquet")
    ]
    p = pd.concat(parts).sort_values("t_s").reset_index(drop=True)
    on = (p["Gas5Flow"] > 300).to_numpy()
    starts = p["t_s"].to_numpy()[1:][on[1:] & ~on[:-1]]
    ws = float(starts[starts >= p["t_s"].iloc[0] + 30][0])
    p = p[(p["t_s"] >= ws - 0.1) & (p["t_s"] <= ws + WINDOW_S)].copy()
    p["t"] = p["t_s"] - ws
    return p


def disc_map(wid: str) -> tuple[np.ndarray, tuple, float, float]:
    pts = pd.read_parquet(OUT / "benchmark" / "point_oof.parquet")
    p = pts[pts["wafer_id"] == wid]
    g = np.linspace(-100, 100, 301)
    gx, gy = np.meshgrid(g, g)
    xy = p[["x_mm", "y_mm"]].to_numpy()
    rbf = RBFInterpolator(xy, p["si_etch"].to_numpy(), kernel="thin_plate_spline", smoothing=1.0)
    z = rbf(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    z[np.hypot(gx, gy) > 100] = np.nan
    lo, hi = float(p["si_etch"].quantile(0.02)), float(p["si_etch"].quantile(0.98))
    return z, (-100, 100, -100, 100), lo, hi


def arrow(fig, x0: float, x1: float, y: float, label: str) -> None:
    fig.add_artist(
        FancyArrowPatch(
            (x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=34, lw=3.2, color=MUTED
        )
    )
    fig.text((x0 + x1) / 2, y + 0.025, label, ha="center", va="bottom", fontsize=FS, color=MUTED, fontweight="bold")


def main() -> None:
    oof = pd.read_parquet(OUT / "review" / "nested_cvplus_oof.parquet")
    h = oof[oof["config"] == HYBRID].copy()
    h["slot"] = h["wafer_id"].str[-2:].astype(int)
    one = h.set_index("wafer_id").loc[WAFER]

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": FS, "axes.edgecolor": "#9a9a9a", "axes.grid": False}
    )
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor="white")
    fig.text(
        0.5,
        0.955,
        "Predicting plasma-etch depth from sensor data, before the wafer is measured",
        ha="center",
        va="top",
        fontsize=24,
        fontweight="bold",
        color=INK,
    )
    heads = [
        (0.035, "1  Sensors in", "optical emission + tool log, one wafer"),
        (0.385, "2  Depth out", "measured wafer vs prediction"),
        (0.705, "3  Every wafer", "88 wafers, each lot held out"),
    ]
    for x, title, sub in heads:
        fig.text(x, 0.855, title, fontsize=25, fontweight="bold", color=INK, va="bottom")
        fig.text(x, 0.848, sub, fontsize=FS, color=MUTED, va="top")

    # ---- panel 1: OES spectrogram + gas flows -------------------------------------------------------
    _t, wl, spec = oes_window(WAFER)
    ax = fig.add_axes((0.075, 0.36, 0.25, 0.405))
    img = np.log10(np.clip(spec, 3, None)).T
    ax.imshow(
        img,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=(0, WINDOW_S, wl[0], wl[-1]),
        vmin=np.percentile(img, 65),
        vmax=np.percentile(img, 99.7),
        interpolation="nearest",
    )
    ax.set_yticks([500, 600, 700, 800])
    ax.set_xticks([])
    ax.set_ylabel("wavelength [nm]", fontsize=FS)
    ax.tick_params(labelsize=FS)
    for lam, lab in ((685.6, "F I"), (516.5, "C₂")):
        ax.annotate(
            lab,
            xy=(WINDOW_S, lam),
            xytext=(WINDOW_S + 1.1, lam),
            textcoords="data",
            annotation_clip=False,
            fontsize=FS + 2,
            fontweight="bold",
            color=INK,
            va="center",
            arrowprops={"arrowstyle": "-", "color": INK, "lw": 1.5},
        )
    ax.set_xlim(0, WINDOW_S)
    gx = fig.add_axes((0.075, 0.12, 0.25, 0.19))
    p = tool_window(WAFER)
    gx.plot(p["t"], p["Gas5Flow"], color=PHASE["etch"], lw=3)
    gx.plot(p["t"], p["Gas4Flow"], color=PHASE["dep"], lw=3)
    gx.text(0.4, 640, "SF₆ etch", color=PHASE["etch"], fontsize=FS, fontweight="bold", va="bottom")
    gx.text(4.6, 340, "C₄F₈", color=PHASE["dep"], fontsize=FS, fontweight="bold", va="bottom")
    gx.set_xlim(0, WINDOW_S)
    gx.set_ylim(-30, 900)
    gx.set_yticks([0, 600])
    gx.set_xticks([0, 6, 12, 18])
    gx.set_xlabel("time [s]", fontsize=FS)
    gx.set_ylabel("sccm", fontsize=FS)
    gx.tick_params(labelsize=FS)
    for s in ("top", "right"):
        gx.spines[s].set_visible(False)

    # ---- panel 2: measured depth map + predicted vs measured mean -----------------------------------
    z, ext, lo, hi = disc_map(WAFER)
    mx = fig.add_axes((0.385, 0.36, 0.2, 0.405))
    im = mx.imshow(z, origin="lower", extent=ext, cmap="cividis", vmin=lo, vmax=hi, interpolation="bilinear")
    mx.add_patch(Circle((0, 0), 100, fill=False, color=INK, lw=1.5))
    mx.set_xlim(-104, 104)
    mx.set_ylim(-104, 104)
    mx.set_aspect("equal")
    mx.axis("off")
    cax = fig.add_axes((0.585, 0.40, 0.011, 0.33))
    cb = fig.colorbar(im, cax=cax, ticks=[int(np.ceil(lo)), int(np.floor(hi))])
    cb.ax.tick_params(labelsize=FS)
    cb.ax.set_title("µm", fontsize=FS, color=INK, pad=10)
    mx.text(0, -118, "measured, 89 sites", ha="center", va="top", fontsize=FS, color=MUTED)

    gg = fig.add_axes((0.395, 0.13, 0.215, 0.1))
    gg.hlines(0, one["cvplus_lo"], one["cvplus_hi"], color="#c4c4c4", lw=16, zorder=1)
    gg.plot(one["pred"], 0, "o", color=FAMILY["hybrid"], ms=20, mec="white", mew=2, zorder=3)
    gg.plot(one["y"], 0, "D", color=INK, ms=13, zorder=4)
    gg.text(
        one["pred"],
        0.95,
        f"predicted {one['pred']:.2f}",
        ha="center",
        va="bottom",
        fontsize=FS,
        color=FAMILY["hybrid"],
        fontweight="bold",
    )
    gg.text(
        one["y"], -0.95, f"measured {one['y']:.2f}", ha="center", va="top", fontsize=FS, color=INK, fontweight="bold"
    )
    gg.set_ylim(-1, 1)
    gg.set_xlim(one["cvplus_lo"] - 0.12, one["cvplus_hi"] + 0.12)
    gg.axis("off")
    fig.text(0.5025, 0.045, "wafer-mean depth [µm], grey = 90 % interval", ha="center", fontsize=FS, color=MUTED)

    # ---- panel 3: predicted vs measured, all held-out lots -----------------------------------------
    y, pr = h["y"].to_numpy(), h["pred"].to_numpy()
    rmse = float(np.sqrt(np.mean((y - pr) ** 2)))
    cov = float(np.mean((h["cvplus_lo"] <= y) & (y <= h["cvplus_hi"])))
    sx = fig.add_axes((0.725, 0.12, 0.245, 0.56))
    lo3, hi3 = 42.8, 44.5
    sx.plot([lo3, hi3], [lo3, hi3], color=MUTED, lw=2, ls="--", zorder=1)
    sc = sx.scatter(y, pr, c=h["slot"], cmap="viridis", vmin=1, vmax=10, s=70, edgecolor="white", lw=0.6, zorder=3)
    sx.scatter([one["y"]], [one["pred"]], s=320, facecolor="none", edgecolor=INK, lw=2.5, zorder=4)
    sx.set_xlim(lo3, hi3)
    sx.set_ylim(lo3, hi3)
    sx.set_aspect("equal")
    sx.set_xticks([43, 44])
    sx.set_yticks([43, 44])
    sx.tick_params(labelsize=FS)
    sx.set_xlabel("measured depth [µm]", fontsize=FS)
    sx.set_ylabel("predicted [µm]", fontsize=FS)
    for s in ("top", "right"):
        sx.spines[s].set_visible(False)
    fig.text(0.725, 0.775, f"RMSE {rmse:.2f} µm", fontsize=25, fontweight="bold", color=FAMILY["hybrid"], va="top")
    fig.text(0.725, 0.725, f"90 % intervals cover {cov:.0%}", fontsize=21, fontweight="bold", color=INK, va="top")
    cx = fig.add_axes((0.905, 0.17, 0.009, 0.2))
    cb3 = fig.colorbar(sc, cax=cx, ticks=[1, 10])
    cb3.ax.tick_params(labelsize=FS - 2)
    cb3.set_label("slot", fontsize=FS - 2)

    arrow(fig, 0.338, 0.383, 0.50, "model")
    arrow(fig, 0.625, 0.668, 0.50, "all wafers")

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "overview.png"
    fig.savefig(out, dpi=100, facecolor="white")
    plt.close(fig)
    Image.open(out).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(out, optimize=True)
    print(f"overview.png {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
