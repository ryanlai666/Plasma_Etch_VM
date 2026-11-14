"""Render the key dashboard figures to standalone HTML (and PNG via headless Chrome/Edge if found).

    python scripts/render_figures.py [--png]

Outputs: outputs/figures/*.html (interactive, inline plotly.js) and docs/figures/*.png (README images).
Uses the same builders as the Streamlit dashboard.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plasma_etch_vm.dashboard import figures as F  # noqa: E402
from plasma_etch_vm.lines import LINES  # noqa: E402

OUT = ROOT / "outputs"
FEAT = ROOT / "data" / "features"
FIGDIR = ROOT / "docs" / "figures"
HTMLDIR = OUT / "figures"

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome",
    "chromium",
    "chromium-browser",
]


def browser() -> str | None:
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    return None


def save(fig, name: str, png: bool, width: int = 1100) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    HTMLDIR.mkdir(parents=True, exist_ok=True)
    html = HTMLDIR / f"{name}.html"
    fig.update_layout(width=width)
    fig.write_html(html, include_plotlyjs="inline", full_html=True, config={"displayModeBar": False})
    b = browser() if png else None
    if b:
        h = int(fig.layout.height or 450) + 20
        subprocess.run(
            [
                b,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--window-size={width + 20},{h}",
                "--virtual-time-budget=8000",
                f"--screenshot={FIGDIR / (name + '.png')}",
                html.as_uri(),
            ],
            check=False,
            capture_output=True,
            timeout=120,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    oof = pd.read_parquet(OUT / "benchmark" / "tool_context" / "oof_predictions.parquet")
    d = oof[(oof["model"] == "ridge") & (oof["target"] == "si_etch_mean")]
    save(F.pred_vs_measured(d, "si_etch_mean", static=True), "pred_vs_measured_si_etch", a.png, 900)
    imp = pd.read_csv(OUT / "interpret" / "feature_importance.csv")
    di = imp[(imp["target"] == "si_etch_mean") & (imp["model"] == "gbm") & (imp["method"] == "perm_lolo")].head(12)
    save(
        F.importance_bar(
            di,
            "Top features — Si etch depth (GBM, permutation on held-out lots)",
            "Δ RMSE when permuted [µm]",
            static=True,
        ),
        "importance_si_etch",
        a.png,
    )
    feats = pd.read_parquet(FEAT / "wafer_features.parquet")
    tg = pd.read_parquet(FEAT / "wafer_targets.parquet")
    j = feats[["C_slot"]].join(tg[["si_etch_mean"]], how="inner")
    save(
        F.feature_scatter(
            j["C_slot"], j["si_etch_mean"], "slot in lot (wafers since conditioning)", "Si etch depth [µm]", static=True
        ),
        "slot_vs_depth",
        a.png,
        800,
    )
    spec = pd.read_parquet(FEAT / "wafer_spectra.parquet")
    wl = pd.read_parquet(FEAT / "wavelengths.parquet")["wavelength_nm"].to_numpy()
    ch = [c for c in spec.columns if c.startswith("ch")]
    first = spec.iloc[0]
    sp = {
        ph: spec[(spec["day"] == first["day"]) & (spec["wafer"] == first["wafer"]) & (spec["phase"] == ph)][
            ch
        ].to_numpy()[0]
        for ph in ("etch", "dep")
    }
    marks = [
        (ln.name, ln.wl_nm)
        for ln in LINES
        if ln.name in ("F_624", "F_685", "F_703", "F_739", "F_775", "C2_516", "C2_563")
    ]
    save(F.spectrum(wl, sp, marks, (500, 790), static=True), "oes_spectrum", a.png)
    tr_file = next((ROOT / "data" / "parquet" / "oes_trace").glob("day=*/lot=*/wafer=1/part-0.parquet"))
    tr = pd.read_parquet(tr_file)
    t = (tr["t_s"] - tr["t_s"].iloc[0]).to_numpy()
    m = (t >= 30) & (t <= 50)
    save(F.line_trace(t[m], tr["L_F_685"].to_numpy()[m], tr["L_C2_516"].to_numpy()[m], static=True), "oes_trace", a.png)
    pts = pd.read_parquet(OUT / "benchmark" / "point_oof.parquet")
    p = pts[pts["wafer_id"] == pts["wafer_id"].iloc[0]]
    save(
        F.wafer_map(
            p,
            p["si_etch"],
            f"Measured Si etch depth [µm] — {p['wafer_id'].iloc[0]}",
            F.SEQ_BLUE,
            (float(pts["si_etch"].quantile(0.01)), float(pts["si_etch"].quantile(0.99))),
            static=True,
        ),
        "wafer_map",
        a.png,
        520,
    )
    print(f"wrote figures to {FIGDIR}")


if __name__ == "__main__":
    main()
