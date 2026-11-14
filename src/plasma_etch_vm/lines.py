"""Emission-line catalogue for SF6 / C4F8 BOSCH plasmas, wavelength calibration and line extraction.

Wavelengths are air reference positions commonly used in plasma-etch OES (NIST ASD atomic lines;
C2 Swan band heads from Pearse & Gaydon). Whether a line is actually resolved in this dataset is
checked on data by :func:`detect_lines` (SNR test on phase-resolved mean spectra) and written to
``docs/line_detection.md``; only detected lines enter the feature set.

Spectrometer: OceanOptics HR4000-class, 3648 px, 185.9–884.0 nm (≈0.19 nm / px).

Wavelength calibration: the stored axis of this dataset is offset by ≈ −1.2 nm (all 17 strong F I
lines of the 3p→3s multiplet appear 1.14–1.29 nm to the blue of their NIST positions).
:func:`calibrate_wavelengths` matches spectral peaks to the F I reference list and fits
``λ_true = a + b·λ_stored``; every line window is placed on the corrected axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Line:
    name: str  # feature-safe identifier
    species: str
    wl_nm: float
    role: str  # etchant | product | passivation | oxygen | actinometer | impurity
    half_width_nm: float = 0.6
    note: str = ""


LINES: tuple[Line, ...] = (
    # --- etchant: atomic fluorine (SF6 dissociation), F I 3p -> 3s multiplet ---
    Line("F_624", "F I", 623.96, "etchant", 0.5),
    Line("F_634", "F I", 634.85, "etchant", 0.5),
    Line("F_641", "F I", 641.37, "etchant", 0.5),
    Line("F_677", "F I", 677.40, "etchant", 0.5),
    Line("F_685", "F I", 685.60, "etchant", 0.5, "strongest F I line in this dataset"),
    Line("F_690", "F I", 690.25, "etchant", 0.5),
    Line("F_703", "F I", 703.75, "etchant", 0.5, "classic actinometry partner of Ar 750.4"),
    Line("F_712", "F I", 712.79, "etchant", 0.5),
    Line("F_739", "F I", 739.87, "etchant", 0.5),
    Line("F_775", "F I", 775.47, "etchant", 0.5),
    # --- etch products ---
    Line("Si_251", "Si I", 251.61, "product"),
    Line("Si_288", "Si I", 288.16, "product", note="Si etch product; loading / endpoint indicator"),
    Line("SiF_440", "SiF (B-X)", 440.05, "product", 1.2),
    Line("SiF_336", "SiF (A-X)", 336.5, "product", 1.2),
    # --- passivation chemistry (C4F8) ---
    Line("C2_516", "C2 Swan (0,0)", 516.52, "passivation", 1.0, "band head, degraded to violet"),
    Line("C2_513", "C2 Swan (1,1)", 512.93, "passivation", 1.0),
    Line("C2_563", "C2 Swan (0,1)", 563.55, "passivation", 1.0),
    Line("CF2_252", "CF2 (A-X)", 251.9, "passivation", 1.0),
    Line("CF2_259", "CF2 (A-X)", 259.5, "passivation", 1.0),
    Line("CF2_263", "CF2 (A-X)", 262.8, "passivation", 1.0),
    Line("CF2_271", "CF2 (A-X)", 271.0, "passivation", 1.0),
    Line("CF_202", "CF (B-X)", 202.4, "passivation", 1.0),
    Line("CS_258", "CS (A-X)", 257.6, "passivation", 1.0),
    # --- oxygen (cleaning residues, SiO2 mask erosion, wall release) ---
    Line("O_777", "O I", 777.19, "oxygen", 0.8),
    Line("O_844", "O I", 844.64, "oxygen"),
    Line("CO_483", "CO Angstrom (B-A)", 483.5, "oxygen", 1.0),
    Line("CO_519", "CO Angstrom (B-A)", 519.8, "oxygen", 1.0),
    # --- sulfur fragments (SF6) ---
    Line("S_469", "S I / S2", 469.4, "etchant", 1.0),
    # --- actinometers / inert ---
    Line("Ar_750", "Ar I", 750.39, "actinometer"),
    Line("Ar_811", "Ar I", 811.53, "actinometer"),
    Line("He_587", "He I", 587.56, "actinometer"),
    Line("He_667", "He I", 667.82, "actinometer"),
    Line("He_706", "He I", 706.52, "actinometer"),
    # --- impurities / leaks ---
    Line("H_656", "H I (Balmer-alpha)", 656.28, "impurity"),
    Line("N2_337", "N2 (C-B) 2nd pos.", 337.13, "impurity", 1.0),
    Line("OH_309", "OH (A-X)", 308.9, "impurity", 1.0),
)

LINE_BY_NAME = {ln.name: ln for ln in LINES}

# NIST ASD F I lines (air, nm) used as wavelength-calibration anchors. All 17 are observed in the
# SF6 step of this dataset.
F_I_REF: tuple[float, ...] = (
    623.96,
    634.85,
    641.37,
    677.40,
    683.43,
    685.60,
    687.02,
    690.25,
    691.02,
    703.75,
    712.79,
    720.24,
    733.20,
    739.87,
    755.22,
    775.47,
    780.02,
)

BG_QUANTILE = 0.2


# ------------------------------------------------------------------------------- calibration
def robust_noise(spectrum: np.ndarray) -> float:
    """MAD estimate of white noise from the first difference of a spectrum."""
    d = np.diff(np.asarray(spectrum, dtype=np.float64))
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def subpixel_peaks(spec: np.ndarray, wl: np.ndarray, min_prominence: float) -> tuple[np.ndarray, np.ndarray]:
    """Peak positions (parabolic sub-pixel refinement, in nm) and prominences."""
    from scipy.signal import find_peaks

    spec = np.asarray(spec, dtype=np.float64)
    pk, props = find_peaks(spec, prominence=min_prominence, distance=3)
    ok = (pk > 0) & (pk < spec.size - 1)
    pk, prom = pk[ok], props["prominences"][ok]
    y0, y1, y2 = spec[pk - 1], spec[pk], spec[pk + 1]
    den = y0 - 2 * y1 + y2
    off = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
    pos = np.interp(pk + np.clip(off, -0.5, 0.5), np.arange(wl.size), np.asarray(wl, dtype=np.float64))
    return pos, prom


def calibrate_wavelengths(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    refs: tuple[float, ...] = F_I_REF,
    max_shift_nm: float = 3.0,
    tol_nm: float = 0.35,
    snr_min: float = 8.0,
) -> dict:
    """Fit ``λ_true = a + b·λ_stored`` by matching spectral peaks to reference lines.

    1. coarse global shift maximising a Gaussian match score over ``±max_shift_nm``;
    2. match each reference to the nearest sub-pixel peak within ``tol_nm`` of the shifted axis;
    3. linear least squares if ≥5 matches spanning ≥80 nm, otherwise a pure shift.

    Returns ``{"a", "b", "shift_nm", "n_matched", "rms_nm", "method"}``.
    """
    spec = np.asarray(spectrum, dtype=np.float64)
    wl = np.asarray(wavelengths, dtype=np.float64)
    refs_arr = np.asarray(refs, dtype=np.float64)
    noise = robust_noise(spec)
    pos, prom = subpixel_peaks(spec, wl, snr_min * noise)
    if pos.size == 0:
        return {"a": 0.0, "b": 1.0, "shift_nm": 0.0, "n_matched": 0, "rms_nm": float("nan"), "method": "none"}
    weight = np.log1p(prom / max(noise, 1e-9))
    shifts = np.arange(-max_shift_nm, max_shift_nm + 1e-9, 0.01)
    dist = np.abs((pos[None, :, None] + shifts[:, None, None]) - refs_arr[None, None, :])  # (S, P, R)
    score = (np.exp(-0.5 * (dist / 0.1) ** 2) * weight[None, :, None]).max(axis=1).sum(axis=1)
    s = float(shifts[int(np.argmax(score))])
    mo_l, mr_l = [], []
    for r in refs_arr:
        j = int(np.argmin(np.abs(pos + s - r)))
        if abs(pos[j] + s - r) <= tol_nm:
            mo_l.append(pos[j])
            mr_l.append(r)
    mo, mr = np.asarray(mo_l), np.asarray(mr_l)
    if mo.size >= 5 and np.ptp(mo) >= 80:
        b, a = np.polyfit(mo, mr, 1)
        method = "linear"
    elif mo.size >= 1:
        a, b, method = float(np.mean(mr - mo)), 1.0, "shift"
    else:
        a, b, method = s, 1.0, "coarse"
    resid = mr - (a + b * mo) if mo.size else np.array([np.nan])
    return {
        "a": float(a),
        "b": float(b),
        "shift_nm": float(np.mean(mr - mo)) if mo.size else s,
        "n_matched": int(mo.size),
        "rms_nm": float(np.sqrt(np.mean(resid**2))),
        "method": method,
    }


def apply_calibration(wavelengths: np.ndarray, cal: dict | None) -> np.ndarray:
    wl = np.asarray(wavelengths, dtype=np.float64)
    return wl if not cal else cal["a"] + cal["b"] * wl


# ----------------------------------------------------------------------------- line windows
@dataclass(frozen=True)
class LineWindow:
    line: Line
    peak: np.ndarray  # channel indices of the peak window
    bg: np.ndarray  # channel indices of the local background


def line_windows(
    wavelengths: np.ndarray, lines: tuple[Line, ...] = LINES, bg_span_nm: float = 3.0
) -> dict[str, LineWindow]:
    """Map each line to peak / background channel indices on a (calibrated) wavelength axis.

    Background = channels within ``±bg_span_nm`` outside the peak window; the intensity uses their
    20th percentile, robust to neighbouring lines of the dense F I multiplet.
    """
    wl = np.asarray(wavelengths, dtype=np.float64)
    out = {}
    for ln in lines:
        if ln.wl_nm < wl[0] or ln.wl_nm > wl[-1]:
            continue
        dist = np.abs(wl - ln.wl_nm)
        peak = np.flatnonzero(dist <= ln.half_width_nm)
        if peak.size == 0:
            peak = np.array([int(np.argmin(dist))])
        bg = np.flatnonzero((dist <= ln.half_width_nm + bg_span_nm) & (dist > ln.half_width_nm))
        out[ln.name] = LineWindow(ln, peak, bg)
    return out


def line_intensities(spectra: np.ndarray, windows: dict[str, LineWindow]) -> dict[str, np.ndarray]:
    """Background-subtracted peak intensity per line: ``max(peak window) − q20(local background)``.

    ``spectra``: (n_time, n_channels) -> name -> (n_time,).
    """
    s = np.asarray(spectra, dtype=np.float32)
    out = {}
    for name, w in windows.items():
        peak = s[:, w.peak].max(axis=1)
        if w.bg.size:
            bg = np.quantile(s[:, w.bg], BG_QUANTILE, axis=1).astype(np.float32)
        else:
            bg = np.zeros(len(s), dtype=np.float32)
        out[name] = peak - bg
    return out


def detect_lines(
    mean_spectrum: np.ndarray,
    wavelengths: np.ndarray,
    noise_sigma: float | None = None,
    snr_min: float = 5.0,
    lines: tuple[Line, ...] = LINES,
) -> list[dict]:
    """SNR test: is each catalogue line a resolved peak in ``mean_spectrum`` (calibrated axis)?

    SNR = (peak − local background) / noise, noise = MAD estimate (:func:`robust_noise`).
    """
    ms = np.asarray(mean_spectrum, dtype=np.float64)[None, :]
    wl = np.asarray(wavelengths, dtype=np.float64)
    wins = line_windows(wl, lines)
    vals = line_intensities(ms, wins)
    if noise_sigma is None:
        noise_sigma = robust_noise(ms[0])
    rows = []
    for ln in lines:
        if ln.name not in wins:
            rows.append(
                {
                    "line": ln.name,
                    "species": ln.species,
                    "role": ln.role,
                    "wl_nm": ln.wl_nm,
                    "in_range": False,
                    "detected": False,
                }
            )
            continue
        w = wins[ln.name]
        prom = float(vals[ln.name][0])
        pk = int(w.peak[np.argmax(ms[0, w.peak])])
        snr = prom / max(noise_sigma, 1e-9)
        rows.append(
            {
                "line": ln.name,
                "species": ln.species,
                "role": ln.role,
                "wl_nm": ln.wl_nm,
                "in_range": True,
                "peak_wl_nm": float(wl[pk]),
                "prominence": prom,
                "snr": float(snr),
                "detected": bool(snr >= snr_min),
            }
        )
    return rows
