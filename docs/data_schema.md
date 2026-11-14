# Data schema — BOSCH plasma-etch dataset (Zenodo 17122442)

Everything below was read from the files themselves (Readme.pdf, the netCDF groups, the CSV/XLSX
tables). The survey started on `Day_2024_08_05.nc` (lot 7) and was then extended to all ten day files during
ingestion (one shared, validated Parquet schema); a value checked on that first day only is marked *(1 day)*.

## 1. Experiment

* Tool: SPTS Omega i2L DSi **Rapier** (ICP), BOSCH process, SF6 etch / C4F8 passivation.
* Recipe (Readme.pdf): 1 s ignition, then **100 cycles × (4.5 s SF6 + 1.5 s C4F8)** ≈ 10 min per wafer.
* Wafers: 200 mm ⟨100⟩ Si, >99.5 % exposed Si, 1 µm SiO2 mask with 89 1 mm² squares.
* DOE: before each lot of 10 wafers the chamber is cleaned (O2, then O2/SF6), a 30 min dummy BOSCH run,
  an end-pointed clean, a 30 min break, then **conditioning** repeated 1×, 3× or 9× on the chuck (C), a
  blank Si wafer (Si) or a blank SiO2 wafer (SiO2). The 10 wafers then run back-to-back (1 min breaks)
  **without intermediate cleaning** → chamber-state drift through the lot.
* Lots (`Lot_status.xlsx`):

| lot | day | wafers | conditioning | metrology |
|---|---|---|---|---|
| 1 | 2024-07-02 | 10 | 3C | 89 & 9 |
| 2 | 2024-07-05 | 10 | 1C | 89 & 9 |
| 3 | 2024-07-09 | 10 | 9C | 89 & 9 |
| 4 | 2024-07-11 | 10 | 3C - Si | 89 & 9 |
| 5 | 2024-07-19 | 10 | 1C - Si | 89 & 9 |
| 6 | 2024-08-01 | 10 | 9C - Si | 89 & 9 |
| 7 | 2024-08-05 | 6 (wafers 7–10 damaged) | 3C - SiO2 | 89 & 9 |
| 8 | 2024-08-07 | 10 | 3C - SiO2 (repeat after tool failure) | 89 only |
| 9 | 2024-08-21 | 10 | 3C | 89 & 9 |
| 10 | 2024-08-22 | 10 | 3C - SiO2 | 89 only |

96 processed wafers in total; **one lot per day**, so "lot" and "day" are the same grouping key.

## 2. Keys linking the sources

| source | wafer key | example |
|---|---|---|
| OES day file `Day_YYYY_MM_DD.nc` | file day + group `Wafer_WW` | `Day_2024_08_05.nc` / `Wafer_03` |
| `Process_data.nc` | group `Day_YYYY_MM_DD_Wafer_WW` | `Day_2024_08_05_Wafer_03` |
| 9 / 89-point CSV | `experiment_key` = `YYYY-MM-DD_WW` | `2024-08-05_03` |
| `Lot_status.xlsx` | lot ↔ date | lot 7 ↔ 2024-08-05 |

The package normalises everything to `wafer_id = "YYYY-MM-DD_WW"` (`metrology.wafer_id`) and derives
`lot` from the date (`metrology.DATE_TO_LOT`). `WW` is the **slot in the lot** (order of processing
after conditioning) — the strongest single predictor of etch depth (section 7).

## 3. Optical emission spectroscopy (`Day_*.nc`)

| item | value |
|---|---|
| spectrometer | OceanOptics HR4-4UVV250-5 (HR4000 class), 3648 px |
| stored wavelength axis | 185.891 – 883.967 nm (float64, identical for all wafers *(1 day)*) |
| **wavelength offset** | stored axis is **−1.22 nm** off: the 17 strongest F I lines all appear 1.14–1.29 nm blue of NIST. Corrected per wafer by `lines.calibrate_wavelengths` (λ_true = 1.335 + 0.99984·λ_stored, 14–17 lines matched, RMS 0.02 nm ≈ 0.1 px) |
| variables per wafer group | `data` (time × 3648, **uint16 codes**), `times` (float64), `wavelengths` (float64) |
| decoding | `value = Dictionary_OES.nc:data[code]`; 61 443 float32 entries, −323.6 … 7782.9 counts (dark-subtracted, so negative values occur) |
| time base | `times` = **absolute** seconds since 1970-01-01 UTC |
| sampling | nominal 25 Hz; measured median Δt = 0.0419 s (**≈23.9 Hz**); 18–23 gaps > 0.1 s per wafer; wafer 2024-08-05_06 is degraded (1241 gaps, max 0.99 s, 10 598 rows instead of ~15 000) |
| rows per wafer | 10 598 – 15 179 *(1 day)*; ~648 s recorded, plasma on for 598 s |
| HDF5 layout | chunks ≈ 4974 × 1216, gzip level 9; read in chunk-aligned blocks (`chunk_rows: auto`) |
| detector baseline | common-mode jitter of ±~20 counts/channel per spectrum → the raw channel sum swings by ±70 000 and is useless as a plasma-on signal. 186–250 nm carries no detectable emission (etch/dep within 0.6 count of dark) and is used as a per-spectrum baseline |

**Detected emission** (`docs/line_detection.md`, SNR ≥ 5 on phase-resolved means): ten F I lines
(624–775 nm, SNR 13–232) in the SF6 step and the C2 Swan Δv = 0, −1 bands (513, 516, 563 nm, SNR
21–113) in the C4F8 step. **Not detected**: Si I, SiF, CF2, CF, CS, O I, CO, Ar I, He I, H I, N2, OH →
no actinometer gas is present, so F/Ar actinometry is impossible; ratios use F-line pairs, C2/F and
cross-phase F(dep)/F(etch) instead.

## 4. Tool traces (`Process_data.nc`)

| item | value |
|---|---|
| groups | 96 (`Day_YYYY_MM_DD_Wafer_WW`) — all wafers, all days, in one 8.8 MB file |
| variables | `data` (time × n_feature, uint16 codes), `times`, `feature` (names, prefix `Stat3_Etch_MV_`) |
| decoding | `Dictionary_process.nc`, 49 290 float32 entries |
| sampling | 5 Hz (Δt = 0.2 s), 3193 – 3835 rows/wafer, no gaps (as stated in the Readme) |
| time base | **relative** seconds (first sample at 91–142 s; 0 s for 2024-08-07_08/09) — *not* alignable to the OES epoch clock, so each stream is segmented independently |
| parameters | **31** on every day; 2024-07-02 has **44** (extra: Gas6Flow, Heater5–8Temp, SourceRF2LoadCapacitor, SourceRFLoadCapacitor, ThermoCouple1–4Temp, attenuatorRatio, moriOuterCurrent). Features use the 31 common ones; 9 are constant and are dropped automatically |

Common parameters: EpdIntensity, ForeLinePressure, Gas1/2/3/4/5/7/8Flow, Heater1–4Temp, HeliumBPFlow,
HeliumBPPressure, PlatenDcBias, PlatenRFLoadCapacitor, PlatenRFLoadPower, PlatenRFPeakToPeak,
PlatenRFReflectedPower, PlatenRFTuningCapacitor, Pressure, SourceRF2LoadPower/PeakToPeak/
ReflectedPower/TuningCapacitor, SourceRFLoadPower/PeakToPeak/ReflectedPower/TuningCapacitor,
moriInnerCurrent.

**Gas channels are anonymous.** From the duty cycle: `Gas5Flow` switches 0 ↔ 600 sccm for 4.5 s of
every 6 s → **SF6**; `Gas4Flow` 0 ↔ 300 sccm for 1.5 s → **C4F8**; `Gas1Flow` (150 sccm) appears only in
the pre-plasma stabilisation; `Gas2Flow` (~4 sccm) and `Gas7Flow` (~2 sccm) are constant small flows.
Source RF ≈ 2.8 kW; platen (bias) ≈ 29 W in the SF6 step with a ~79 W sub-step at its start, 39 W in
the C4F8 step. (Mapping configurable in `configs/data.yaml: gas_map`.)

## 5. Phase / cycle segmentation (both streams)

| stream | method | result *(1 day)* |
|---|---|---|
| tool log (5 Hz) | plasma on = SourceRFLoadPower > 500 W; etch vs dep from normalised SF6 / C4F8 flow | 99–100 cycles, etch 22 / dep 8 samples per cycle |
| OES (~24 Hz) | plasma on from background-subtracted F I 685.6 + C2 516.5 signal; etch vs dep by Otsu split of log(F/C2); 1 s ignition tagged `ign` | 100 SF6 steps (median 4.55 s), 99 C4F8 steps (1.46 s), plasma span 598 s |

`cycle` increments at every SF6-step onset (cycle k = SF6 step k + the C4F8 step that follows it).
In the tool log 21 of 88 wafers show 99 instead of 100 cycles; these wafers etch 0.28 µm less on
average (whether this is real or a logging truncation cannot be decided from the data).

## 6. Metrology (targets)

All values in **µm**. Coordinates `X`, `Y` in µm (converted to mm).

| table | wafers | sites | instruments | notes |
|---|---|---|---|---|
| `Si_Oxide_etch_89_points.csv` | **88** | 89 (19 mm grid) | FRT MicroProf 300 interferometer (post-ox), KLA Tencor P-17 profilometer (step height), reflectometer pre-ox at 15 sites + IDW | measured Feb 2025; 2 % of post-ox values IDW-interpolated (`postox_thickness_nan` = NaN) |
| `Si_Oxide_etch_9_points.csv` | **75** | 9 (B6 … J6) | MPROBE 40-MSP reflectometer, Dektak 8 profilometer | measured on the day, manual; lots 8 & 10 absent; 9 rows with empty `experiment_key` dropped |

Missing from the 89-point table: 2024-07-02_07, 2024-08-21_09, 2024-08-22_05 … _10.
Columns: `preox_thickness`, `postox_thickness`, `stepheight`, `oxide_etch` = pre − post oxide,
`si_etch` = stepheight − post oxide (Si trench depth).

**Cross-campaign checks (new findings):**

* The 9-point coordinates are **rotated by 180°** relative to the 89-point frame: site-level Si depth
  correlates at r = −0.06 as stored and at **r = 0.995** after (x, y) → (−x, −y) (657 matched rows).
  `metrology.rotate_points` applies the correction; wafer-level targets are unaffected.
* After alignment the 89-point depths are systematically **+1.34 µm** deeper (different profilometer,
  measured months later); wafer means still agree at r = 0.989 (73 wafers).
* SiO2 mask loss agrees poorly between campaigns (wafer means r = 0.21) → `ox_etch_mean` and
  `selectivity` labels are noisy, which bounds how well any model can predict them.

**Wafer-level targets** (`metrology.wafer_targets`, 89-point table):

| target | definition | mean ± sd (88 wafers) |
|---|---|---|
| `si_etch_mean` | mean Si depth | 43.67 ± 0.40 µm |
| `si_nu_halfrange_pct` | 100·(max − min)/(2·mean) | 13.4 ± 1.4 % (one outlier, 25.9 %, wafer 2024-08-01_07 with a 28.7 µm site) |
| `si_nu_std_pct` | 100·sd/mean | — |
| `si_edge_center` | mean(r ≥ 0.75 R) − mean(r ≤ 0.25 R) | 4.53 µm (4.10 – 4.99): edge-fast profile, 46.2 µm edge vs 41.7 µm centre on average |
| `ox_etch_mean` | mean SiO2 loss | 0.658 ± 0.044 µm |
| `selectivity` | si_etch_mean / ox_etch_mean | 66.7 ± 4.7 |
| `si_rate_um_per_cycle` | si_etch_mean / n_cycles | ≈0.44 µm/cycle |

## 7. Derived store (this repo)

```
data/parquet/
  process/day=…/lot=…/wafer=…/step={off,ign,etch,dep}/part-0.parquet   5 Hz, 31 float32 params + t_s, cycle, sample
  oes/day=…/lot=…/wafer=…/step=…/part-0.parquet                          ~24 Hz, ch0000…ch3647 (int16 counts) + t_s, cycle, sample
  oes_trace/day=…/lot=…/wafer=…/part-0.parquet                           per-sample F I / C2 line intensities, total, phase, cycle
  meta/{oes_dictionary,process_dictionary,wavelengths}.parquet, wavelength_calibration/*.json, oes_storage.json
  _manifest/*.json                                                         per-wafer source signature, config hash, rows, MB, s
data/features/
  cycles_process.parquet  cycles_oes.parquet  wafer_spectra.parquet  wavelengths.parquet
  wafer_features.parquet  wafer_targets.parquet  feature_groups.json
```

OES storage modes (`configs/data.yaml: ingest.store_oes_as`): `codes` (lossless uint16, ~92 MB/wafer),
`counts_i16` (default: intensities rounded to integer counts, |error| ≤ 0.5 count vs ≥ 14 count noise,
~64 MB/wafer), `float32`. The raw netCDF files remain the lossless archive.
