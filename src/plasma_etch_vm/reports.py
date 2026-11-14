"""Markdown reports generated from data (line detection, wavelength calibration)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .lines import detect_lines, robust_noise


def line_detection_report(feature_dir: Path, out_md: Path, snr_min: float = 5.0, store: Path | None = None) -> str:
    """Detect catalogue lines on the phase-resolved, all-wafer mean spectra (calibrated axis)."""
    ws = pd.read_parquet(feature_dir / "wafer_spectra.parquet")
    wl = pd.read_parquet(feature_dir / "wavelengths.parquet")
    wl_cal = wl["wavelength_nm"].to_numpy()
    ch = [c for c in ws.columns if c.startswith("ch")]
    tables = {}
    for phase in ("etch", "dep"):
        spec = ws[ws["phase"] == phase][ch].to_numpy(np.float64).mean(axis=0)
        tables[phase] = pd.DataFrame(detect_lines(spec, wl_cal, robust_noise(spec), snr_min))
    merged = tables["etch"][["line", "species", "role", "wl_nm", "snr"]].rename(columns={"snr": "snr_etch"})
    merged["snr_dep"] = tables["dep"]["snr"].to_numpy()
    merged["detected"] = merged[["snr_etch", "snr_dep"]].max(axis=1) >= snr_min
    cal_rows = []
    store = store or feature_dir.parent / "parquet"
    for f in sorted((store / "meta" / "wavelength_calibration").glob("*.json")):
        c = json.loads(f.read_text())
        keys = ("shift_nm", "a", "b", "n_matched", "rms_nm", "method")
        cal_rows.append({"wafer": f.stem, **{k: c[k] for k in keys}})
    n_wafers = ws[["day", "wafer"]].drop_duplicates().shape[0]
    md = [
        "# Emission-line detection and wavelength calibration",
        "",
        f"Generated from `{feature_dir.name}/wafer_spectra.parquet` ({n_wafers} wafers).",
        "SNR = (peak − q20 local background) / MAD noise on the phase-resolved mean spectrum;",
        f"detected if SNR ≥ {snr_min:g} in either phase. Wavelength axis corrected per wafer (below).",
        "",
        "## Wavelength calibration (λ_true = a + b·λ_stored, anchored on 17 NIST F I lines)",
        "",
    ]
    if cal_rows:
        cdf = pd.DataFrame(cal_rows)
        md += ["| wafer | shift_nm | a | b | n_matched | rms_nm | method |", "|---|---|---|---|---|---|---|"]
        md += [
            f"| {r.wafer} | {r.shift_nm:+.3f} | {r.a:.4f} | {r.b:.6f} | {r.n_matched} | {r.rms_nm:.4f} | {r.method} |"
            for r in cdf.itertuples()
        ]
    md += [
        "",
        "## Line detection",
        "",
        "| line | species | role | λ (nm) | SNR etch | SNR dep | detected |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in merged.itertuples():
        se = "" if pd.isna(r.snr_etch) else f"{r.snr_etch:.1f}"
        sd = "" if pd.isna(r.snr_dep) else f"{r.snr_dep:.1f}"
        md.append(
            f"| {r.line} | {r.species} | {r.role} | {r.wl_nm:.2f} | {se} | {sd} | {'yes' if r.detected else 'no'} |"
        )
    md += [
        "",
        "Notes:",
        "* `S_469` coincides with the C2 Swan Δv=+1 band heads (468.5–473.7 nm); its signal in the C4F8 step",
        "  is attributed to C2 and the line is not used as a sulfur feature.",
        "* No Ar/He/O lines are detected, so classical actinometry (F/Ar) is not possible on this tool;",
        "  ratios are formed between F I lines, C2 bands and across the etch/passivation steps instead.",
    ]
    text = "\n".join(md) + "\n"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")
    return text
