"""Chunked, dictionary-decoded readers for the BOSCH netCDF files (xarray + h5netcdf).

Layout (see docs/data_schema.md):

* ``Dictionary_OES.nc`` / ``Dictionary_process.nc``: 1-D float32 ``data`` lookup table; the stored
  uint16 code ``c`` decodes to ``table[c]`` (lossless 16-bit dictionary encoding).
* ``Day_YYYY_MM_DD.nc``: one group per wafer (``Wafer_01`` ...), variables ``data`` (time × 3648,
  uint16 codes), ``times`` (float seconds, CF units string), ``wavelengths`` (nm).
* ``Process_data.nc``: one group per wafer (``Day_YYYY_MM_DD_Wafer_WW``), variables ``data``
  (time × n_feature uint16 codes), ``times``, ``feature`` (parameter names). The 2024-07-02 lot has
  44 parameters, all other lots 31; readers return names so callers can align columns.

All readers slice the on-disk arrays in row blocks so peak memory is bounded by ``chunk_rows``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import h5netcdf
import numpy as np
import xarray as xr

PROCESS_PREFIX = "Stat3_Etch_MV_"


def load_decoder(path: Path) -> np.ndarray:
    """Load a dictionary file -> float32 lookup table indexed by the uint16 code."""
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        return np.asarray(ds["data"].values, dtype=np.float32)


def list_groups(path: Path) -> list[str]:
    with h5netcdf.File(path, "r") as f:
        return sorted(f.groups)


@dataclass
class GroupInfo:
    group: str
    n_time: int
    n_cols: int
    columns: list[str]
    time_units: str


def _as_str_list(values: np.ndarray) -> list[str]:
    out = []
    for v in np.asarray(values).ravel():
        out.append(v.decode() if isinstance(v, bytes) else str(v))
    return out


def group_info(path: Path, group: str, kind: str) -> GroupInfo:
    """Metadata of one wafer group without loading the data block."""
    with h5netcdf.File(path, "r") as f:
        g = f.groups[group]
        shape = g.variables["data"].shape
        if kind == "oes":
            cols = [f"{w:.4f}" for w in np.asarray(g.variables["wavelengths"][:], dtype=np.float64)]
        else:
            cols = [c.replace(PROCESS_PREFIX, "") for c in _as_str_list(g.variables["feature"][:])]
        units = g.variables["times"].attrs.get("units", "")
        units = units.decode() if isinstance(units, bytes) else str(units)
        return GroupInfo(group, int(shape[0]), int(shape[1]), cols, units)


def read_axis(path: Path, group: str, name: str) -> np.ndarray:
    with h5netcdf.File(path, "r") as f:
        return np.asarray(f.groups[group].variables[name][:])


def hdf5_chunk_rows(path: Path, group: str) -> int:
    """Row height of the HDF5 chunks of ``group/data`` (reading in multiples avoids re-inflating chunks)."""
    with h5netcdf.File(path, "r") as f:
        v = f.groups[group].variables["data"]
        return int(v.chunks[0]) if v.chunks else int(v.shape[0])


def iter_blocks(path: Path, group: str, chunk_rows: int | str = "auto") -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(start_row, times[start:stop], codes[start:stop, :])`` blocks of the raw uint16 codes.

    Uses xarray (h5netcdf engine) lazily so only ``chunk_rows`` rows are in memory at once;
    ``chunk_rows="auto"`` aligns blocks with the HDF5 chunk height.
    """
    if chunk_rows in (None, "auto"):
        chunk_rows = hdf5_chunk_rows(path, group)
    chunk_rows = int(chunk_rows)
    with xr.open_dataset(path, group=group, engine="h5netcdf", decode_times=False, cache=False) as ds:
        data = ds["data"]
        times = ds["times"]
        n = data.shape[0]
        for start in range(0, n, chunk_rows):
            stop = min(n, start + chunk_rows)
            yield (
                start,
                np.asarray(times[start:stop].values, dtype=np.float64),
                np.asarray(data[start:stop].values, dtype=np.uint16),
            )


def read_group_decoded(
    path: Path, group: str, decoder: np.ndarray, kind: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read a full group (small groups only, e.g. process traces): ``times, values(float32), columns``."""
    info = group_info(path, group, kind)
    t_parts, v_parts = [], []
    for _, t, codes in iter_blocks(path, group, chunk_rows=10_000):
        t_parts.append(t)
        v_parts.append(decoder[codes])
    return np.concatenate(t_parts), np.concatenate(v_parts), info.columns
