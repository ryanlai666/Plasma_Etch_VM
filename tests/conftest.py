"""Synthetic miniature of the Zenodo 17122442 file layout (CPU-only, ~seconds).

* ``Dictionary_OES.nc`` / ``Dictionary_process.nc``: sorted float32 lookup tables.
* ``Day_2024_08_05.nc``: 2 wafer groups, 3648 channels on the real (mis-calibrated, −1.22 nm) axis,
  BOSCH-like switching between an F I spectrum (etch, 4.5 s) and a C2 Swan spectrum (dep, 1.5 s).
* ``Process_data.nc``: matching 5 Hz tool traces (SF6 600 sccm / C4F8 300 sccm duty cycle).
"""

from __future__ import annotations

from pathlib import Path

import h5netcdf
import numpy as np
import pytest

from plasma_etch_vm.lines import F_I_REF

N_CH = 3648
WL = np.linspace(185.89, 883.97, N_CH)
SHIFT = -1.22  # stored axis is blue-shifted w.r.t. true wavelengths
OES_DT = 1 / 24.0
PROC_DT = 0.2
N_CYCLES = 4
CYCLE_S = 6.0
ETCH_S = 4.5
LEAD_S = 2.0


def oes_dictionary() -> np.ndarray:
    return np.arange(-200.0, 6000.0, 0.5, dtype=np.float32)


def encode(values: np.ndarray, table: np.ndarray) -> np.ndarray:
    idx = np.clip(np.rint((values - table[0]) / (table[1] - table[0])), 0, table.size - 1)
    return idx.astype(np.uint16)


def peak(center: float, height: float, width: float = 0.25) -> np.ndarray:
    return height * np.exp(-0.5 * ((WL - (center + SHIFT)) / width) ** 2)


ETCH_SPEC = sum(peak(c, 3000.0 if c == 685.60 else 400.0) for c in F_I_REF)
DEP_SPEC = 0.35 * ETCH_SPEC + peak(516.52, 1500.0, 0.4) + peak(563.55, 300.0, 0.4)


def phase_at(t: np.ndarray) -> np.ndarray:
    """'off' before LEAD_S and after the cycles, else etch/dep by the 6 s period."""
    tt = t - LEAD_S
    ph = np.where((tt % CYCLE_S) < ETCH_S, "etch", "dep").astype(object)
    ph[(tt < 0) | (tt >= N_CYCLES * CYCLE_S)] = "off"
    return ph


def write_day(path: Path, n_wafers: int = 2, seed: int = 0, bad_code: bool = False) -> None:
    rng = np.random.default_rng(seed)
    table = oes_dictionary()
    total_s = LEAD_S + N_CYCLES * CYCLE_S + 1.0
    with h5netcdf.File(path, "w") as f:
        for w in range(1, n_wafers + 1):
            g = f.create_group(f"Wafer_{w:02d}")
            t = np.arange(0, total_s, OES_DT)
            ph = phase_at(t)
            spec = np.where((ph == "etch")[:, None], ETCH_SPEC, np.where((ph == "dep")[:, None], DEP_SPEC, 0.0))
            spec = spec * (1 - 0.01 * w) + rng.normal(0, 10, size=spec.shape) + rng.normal(0, 3, size=(len(t), 1))
            codes = encode(spec.astype(np.float32), table)
            if bad_code:
                codes[0, 0] = table.size + 5
            g.dimensions = {"time": len(t), "wavelength": N_CH}
            v = g.create_variable("times", ("time",), data=1.7228e9 + w * 700 + t)
            v.attrs["units"] = "seconds since 1970-01-01 00:00:00 UTC"
            v.attrs["calendar"] = "standard"
            g.create_variable("wavelengths", ("wavelength",), data=WL)
            g.create_variable("data", ("time", "wavelength"), data=codes, chunks=(200, 1216))


PROC_FEATURES = [
    "Gas4Flow",
    "Gas5Flow",
    "SourceRFLoadPower",
    "Pressure",
    "PlatenRFLoadPower",
    "PlatenRFLoadCapacitor",
    "HeliumBPFlow",
    "Gas8Flow",
]


def write_process(path: Path, day: str = "2024_08_05", n_wafers: int = 2, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    table = np.linspace(-50.0, 3000.0, 60000, dtype=np.float32)  # must fit uint16 codes
    with h5netcdf.File(path, "w") as f:
        for w in range(1, n_wafers + 1):
            g = f.create_group(f"Day_{day}_Wafer_{w:02d}")
            t = np.arange(0, LEAD_S + N_CYCLES * CYCLE_S + 1.0, PROC_DT)
            ph = phase_at(t)
            vals = np.zeros((len(t), len(PROC_FEATURES)), dtype=np.float32)
            vals[:, 0] = np.where(ph == "dep", 300.0, 0.1)
            vals[:, 1] = np.where(ph == "etch", 600.0, 0.0)
            vals[:, 2] = np.where(ph == "off", 0.0, 2800.0)
            vals[:, 3] = 0.04 + 0.001 * w + rng.normal(0, 1e-4, len(t))
            vals[:, 4] = np.where(ph == "etch", 29.0, 39.0) + rng.normal(0, 0.5, len(t))
            vals[:, 5] = 85 + w + rng.normal(0, 0.3, len(t))
            vals[:, 6] = 1.3 + rng.normal(0, 0.01, len(t))
            vals[:, 7] = 0.0
            g.dimensions = {"feature": len(PROC_FEATURES), "time": len(t)}
            import h5py

            names = np.array([("Stat3_Etch_MV_" + c).encode() for c in PROC_FEATURES], dtype=object)
            g.create_variable("feature", ("feature",), data=names, dtype=h5py.string_dtype())
            v = g.create_variable("times", ("time",), data=100.0 + t)
            v.attrs["units"] = "seconds since 1970-01-01 00:00:00 UTC"
            g.create_variable("data", ("time", "feature"), data=encode(vals, table))
    with h5netcdf.File(path.parent / "Dictionary_process.nc", "w") as f:
        f.dimensions = {"x": table.size}
        f.create_variable("data", ("x",), data=table)


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("raw")
    with h5netcdf.File(d / "Dictionary_OES.nc", "w") as f:
        f.dimensions = {"x": oes_dictionary().size}
        f.create_variable("data", ("x",), data=oes_dictionary())
    write_day(d / "Day_2024_08_05.nc")
    write_process(d / "Process_data.nc")
    return d


@pytest.fixture(scope="session")
def ingest_cfg(raw_dir, tmp_path_factory) -> dict:
    from plasma_etch_vm.config import load_config

    cfg = load_config("data.yaml")
    out = tmp_path_factory.mktemp("store")
    cfg["paths"] = {
        "raw": str(raw_dir),
        "parquet": str(out / "parquet"),
        "features": str(out / "features"),
        "outputs": str(out / "outputs"),
    }
    cfg["ingest"]["row_group_rows"] = 300
    return cfg


@pytest.fixture(scope="session")
def ingested(ingest_cfg):
    from plasma_etch_vm.ingest import run_ingest

    stats = run_ingest(ingest_cfg)
    return ingest_cfg, stats
