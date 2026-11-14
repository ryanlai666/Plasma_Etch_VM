"""Virtual metrology for BOSCH plasma etching (Zenodo 17122442).

Sub-modules
-----------
config       YAML config loading
zenodo       download + checksum verification against the Zenodo API
netcdf_io    dictionary-decoded, chunked readers for the OES / process netCDF files
segmentation BOSCH cycle / phase labelling from tool traces
ingest       netCDF -> partitioned Parquet (memory-bounded, idempotent, validated)
metrology    9 / 89 point thickness tables -> wafer- and point-level targets
lines        emission-line catalogue (F, O, Si, SiF, CF2, C2, CO, S, He, H, N2 ...)
features     per-wafer / per-phase feature builders (tool params, OES lines, PCA/NMF)
models       baselines, gradient boosting, 1D-CNN, transformer, group CV, conformal
serve        FastAPI prediction service
dashboard    Streamlit dashboard
"""

__version__ = "0.1.0"
