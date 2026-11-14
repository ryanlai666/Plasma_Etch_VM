# Data

## Source

| | |
|---|---|
| Title | A Multi-Model Dataset for BOSCH Plasma-Etching: Optical Emission Spectra, Process Parameters, and Wafer Measurements for Data-Driven Plasma Modeling |
| Authors | Mudassir Ali Sayyed, Tom Seifert, Stephan Zieger, Simeon Schwarzenberg, Aditya Deshmukh, Micha Haase, Jan Langer (TU Chemnitz / Fraunhofer ENAS / TU Bergakademie Freiberg) |
| Repository | Zenodo record 17122442 — https://zenodo.org/records/17122442 (API: https://zenodo.org/api/records/17122442) |
| DOI | 10.5281/zenodo.17122442 |
| Published | 2025-09-15 |
| License | **CC BY 4.0** (attribution required; commercial use and derivatives allowed) |
| Total size | 7.913 GB (18 files) |

Metadata above was read from the Zenodo API on 2026-11-12 (cached in `data/raw/zenodo_record_17122442.json`).

### Citation

```bibtex
@dataset{sayyed_2025_17122442,
  author    = {Sayyed, Mudassir Ali and Seifert, Tom and Zieger, Stephan and Schwarzenberg, Simeon and
               Deshmukh, Aditya and Haase, Micha and Langer, Jan},
  title     = {A Multi-Model Dataset for BOSCH Plasma-Etching: Optical Emission Spectra, Process
               Parameters, and Wafer Measurements for Data-Driven Plasma Modeling},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17122442},
  url       = {https://doi.org/10.5281/zenodo.17122442}
}
```

## Files, sizes, checksums

Download URL pattern: `https://zenodo.org/api/records/17122442/files/<file>/content`.
MD5 values are from the Zenodo API; "verified" = local size and MD5 match (`pevm download` writes
`data/raw/MANIFEST.json`). All 18 files were downloaded and verified for the reported results.

| file | MB | md5 | MD5 verified |
|---|---|---|---|
| Day_2024_07_02.nc | 829.42 | 28f96ab71714e070af1b2ccddb952b60 | yes |
| Day_2024_07_05.nc | 833.56 | fe2a0432b433daf0bdf9745b877b879b | yes |
| Day_2024_07_09.nc | 817.50 | 1632b5b67f52cc8e8a53191777a330b7 | yes |
| Day_2024_07_11.nc | 824.22 | a32f4406f243d267a430fc9a3624ee28 | yes |
| Day_2024_07_19.nc | 832.14 | dd631872ca55ca28965e464fe81f13f2 | yes |
| Day_2024_08_01.nc | 832.65 | 3a2f4efc1ee23fb09dbe1b920c1913ae | yes |
| Day_2024_08_05.nc | 476.31 | 4d13022c1817334dd02ffdfa936c4469 | yes |
| Day_2024_08_07.nc | 823.27 | a9e989c84ba2c29aecc6de7f3410f60c | yes |
| Day_2024_08_21.nc | 816.26 | aeee1ee145575633b599a71c2622bb3a | yes |
| Day_2024_08_22.nc | 817.95 | e63c41f3560914e9a1fa9c388314b3a5 | yes |
| Dictionary_OES.nc | 0.09 | e7979670371604ff6f46d75dc3221cb0 | yes |
| Dictionary_process.nc | 0.09 | 0dde5a3a913eb1fa8512ef2f8748fb34 | yes |
| Lot_status.xlsx | 0.01 | 339beca13f321dc3af244bf2d2ce284c | yes |
| Process_data.nc | 8.82 | 4567d24ec2125102a2e5129203ba31fa | yes |
| Readme.pdf | 0.36 | f9a9bd323bd9e227a486249a631e8468 | yes |
| Si_Oxide_etch_89_points.csv | 0.65 | 446e75b040eea37b634eeb8f763a62fc | yes |
| Si_Oxide_etch_9_points.csv | 0.05 | 78515caf25e29e558e1859b92f8a4827 | yes |
| Wafer_layout.pdf | 0.03 | 9bb916f316fadb02caad704f7cb2f09d | yes |

The ten OES day files (7.90 GB) were downloaded with 3 parallel streams, MD5-verified, ingested into Parquet,
row-count-verified against the netCDF and then deleted, so peak disk use stays near one day file
(`scripts/run_campaign.py --stages data`). Download rates from Zenodo varied between ~0.24 MB/s (single stream)
and ~4.3 MB/s (three streams).

## Local layout

* `data/raw/` — the small files above (gitignored; day files are deleted after ingestion). Re-create with `pevm download`
  (set `zenodo.day_files: all` in `configs/data.yaml` for all ten day files).
* `data/parquet/` — partitioned analytics copy produced by `pevm ingest` (gitignored).
* `data/features/` — cycle tables and wafer feature/target tables (gitignored).

Schema, keys, sampling and data-quality findings: [`docs/data_schema.md`](docs/data_schema.md).

## Optional sanity dataset (not downloaded)

SECOM (UCI ML Repository, dataset 179): 1567 instances × 591 features, semiconductor manufacturing
pass/fail classification, CC BY 4.0, DOI 10.24432/C54305, https://archive.ics.uci.edu/dataset/179/secom.
Verified on the UCI page 2026-11-12. Not used yet (the BOSCH data already provides real targets).
