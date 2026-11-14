"""Unit tests: config, lines / calibration, segmentation, metrology."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plasma_etch_vm.config import apply_overrides, deep_merge, load_config
from plasma_etch_vm.lines import (
    F_I_REF,
    LINE_BY_NAME,
    apply_calibration,
    calibrate_wavelengths,
    detect_lines,
    line_intensities,
    line_windows,
)
from plasma_etch_vm.metrology import load_lot_status, parse_group_name, rotate_points, wafer_id, wafer_targets
from plasma_etch_vm.segmentation import (
    assign_cycles,
    cycle_summary,
    label_from_indicator,
    label_process,
    otsu_threshold,
    segments,
)

from .conftest import ETCH_SPEC, WL


def test_config_merge_and_overrides():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
    cfg = apply_overrides({"train": {"epochs": 5}}, ["train.epochs=7", "model.d_model=null"])
    assert cfg["train"]["epochs"] == 7 and cfg["model"]["d_model"] is None
    tab = load_config("models/tabular.yaml")
    assert tab["paths"]["raw"] == "data/raw" and "feature_sets" in tab and tab["metrology"]["points"] == "points_89"


def test_calibration_recovers_shift():
    rng = np.random.default_rng(1)
    spec = ETCH_SPEC + rng.normal(0, 5, WL.size)
    cal = calibrate_wavelengths(spec, WL)
    assert cal["n_matched"] >= 15 and abs(cal["shift_nm"] - 1.22) < 0.05 and cal["rms_nm"] < 0.05
    wl = apply_calibration(WL, cal)
    det = {r["line"]: r for r in detect_lines(spec, wl)}
    assert det["F_685"]["detected"] and not det["Ar_750"]["detected"]


def test_line_intensity_background_subtraction():
    wins = line_windows(WL, (LINE_BY_NAME["F_685"],))
    spec = np.full((2, WL.size), 50.0)
    spec[1, wins["F_685"].peak] += 100.0
    li = line_intensities(spec, wins)["F_685"]
    assert li[0] == pytest.approx(0.0, abs=1e-4) and li[1] == pytest.approx(100.0, abs=1e-3)


def test_f_reference_lines_sorted_and_in_range():
    assert list(F_I_REF) == sorted(F_I_REF) and min(F_I_REF) > 600 and max(F_I_REF) < 800


def test_otsu_and_indicator_segmentation():
    x = np.r_[np.ones(50), 10 * np.ones(50)]
    assert 1 < otsu_threshold(x) < 10
    ind = np.tile(np.r_[np.full(22, 30.0), np.full(8, 0.5)], 5)
    ph = label_from_indicator(ind)
    assert (assign_cycles(ph).max()) == 5


def test_label_process_cycles():
    t = np.arange(0, 62, 0.2)
    tt = t - 2
    etch = (tt % 6) < 4.5
    on = (tt >= 0) & (tt < 60)
    df = pd.DataFrame(
        {
            "t_s": t,
            "Gas5Flow": np.where(etch & on, 600.0, 0.0),
            "Gas4Flow": np.where(~etch & on, 300.0, 0.0),
            "SourceRFLoadPower": np.where(on, 2800.0, 0.0),
        }
    )
    out = label_process(df)
    seg = segments(out["t_s"].to_numpy(), out["phase"].to_numpy(), out["cycle"].to_numpy())
    summ = cycle_summary(seg)
    assert summ["n_cycles"] == 10
    assert summ["etch_dur_mean_s"] == pytest.approx(4.6, abs=0.25)


def test_metrology_ids_and_targets(tmp_path):
    assert wafer_id("2024-08-05", 3) == "2024-08-05_03"
    assert parse_group_name("Day_2024_08_05_Wafer_03") == ("2024-08-05", 3)
    pts = pd.DataFrame(
        {
            "wafer_id": ["2024-08-05_01"] * 4,
            "day": "2024-08-05",
            "lot": 7,
            "wafer": 1,
            "r_norm": [0.0, 0.2, 0.8, 0.9],
            "si_etch": [40.0, 41.0, 50.0, 52.0],
            "oxide_etch": [0.6, 0.6, 0.7, 0.7],
        }
    )
    t = wafer_targets(pts).iloc[0]
    assert t["si_etch_mean"] == pytest.approx(45.75)
    assert t["si_nu_halfrange_pct"] == pytest.approx(100 * 12 / (2 * 45.75))
    assert t["si_edge_center"] == pytest.approx(51.0 - 40.5)
    raw = pd.DataFrame(
        [
            ["Lot No.", "Date", "Wafers", "Measurements", "Type"],
            ["Lot 1", "2th July 2024", 10, "89 & 9 points", "3C "],
            ["Lot 7", "5th Aug 2024", 6, "89 & 9 points", "3C - SiO2"],
        ]
    )
    raw.to_excel(tmp_path / "lot.xlsx", header=False, index=False)
    ls = load_lot_status(tmp_path / "lot.xlsx")
    assert ls.set_index("lot").loc[7, "cond_substrate"] == "SiO2" and ls.set_index("lot").loc[1, "n_cond"] == 3


def test_rotate_points_180():
    df = pd.DataFrame({"x_mm": [0.0, 76.0], "y_mm": [76.0, 0.0]})
    r = rotate_points(df, 180)
    assert r["x_mm"].tolist() == [0.0, -76.0] and r["y_mm"].tolist() == [-76.0, 0.0]
    assert rotate_points(rotate_points(df, 90), 270)[["x_mm", "y_mm"]].equals(df[["x_mm", "y_mm"]])
