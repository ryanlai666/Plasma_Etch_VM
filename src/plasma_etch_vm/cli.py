"""Command line entry point: ``pevm <command> [--config ...] [key=value overrides]``.

Commands (in pipeline order)::

    download      fetch allowed files from Zenodo, verify size + MD5, write data/raw/MANIFEST.json
    ingest        netCDF -> partitioned Parquet (process + OES), throughput report
    query-bench   DuckDB / Polars queries over the store (projection + partition pushdown timing)
    lines         wavelength calibration + line-detection report (docs/line_detection.md)
    cycles        per-cycle tables (tool params, OES lines, binned spectra)
    features      wafer feature table + targets -> data/features/
    benchmark     tabular CV benchmark (PLS / ridge / GBM / PCA / NMF) with conformal intervals
    points        per-site (89-point) Si etch-depth benchmark
    train-nn      1D-CNN or StatFE transformer, group CV, checkpoint/resume
    interpret     permutation importance + SHAP for the best tabular models
    export        fit final models on all wafers -> outputs/model_bundle.joblib (used by the API)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import pandas as pd

from .config import load_config, project_path

log = logging.getLogger("pevm")


def _paths(cfg: dict) -> dict:
    return {k: project_path(v) for k, v in cfg["paths"].items()}


def _peak_rss_mb() -> float:
    import psutil

    mi = psutil.Process().memory_info()
    return float(getattr(mi, "peak_wset", getattr(mi, "rss", 0)) / 2**20)


# --------------------------------------------------------------------------------- commands
def cmd_download(cfg: dict, args) -> None:
    from . import zenodo

    p = _paths(cfg)
    zc = cfg["zenodo"]
    record = zenodo.fetch_record(zc["api_url"], cache=p["raw"] / "zenodo_record_17122442.json")
    files = zenodo.list_files(record)
    wanted = list(zc["small_files"])
    day_files = sorted(k for k in files if k.startswith("Day_"))
    wanted += day_files if zc["day_files"] == "all" else list(zc["day_files"])
    total = sum(files[k].size for k in wanted)
    log.info("allowance: %d files, %.2f GB", len(wanted), total / 1e9)
    for key in wanted:
        zenodo.download(files[key], p["raw"], min_free_gb=zc.get("min_free_gb", 15.0))
    rows = zenodo.write_manifest(files, p["raw"], p["raw"] / "MANIFEST.json")
    print(pd.DataFrame(rows)[["file", "size_bytes", "downloaded", "verified"]].to_string(index=False))


def cmd_ingest(cfg: dict, args) -> None:
    from .ingest import run_ingest

    p = _paths(cfg)
    cfg = {**cfg, "paths": {k: str(v) for k, v in p.items()}}
    t0 = time.perf_counter()
    stats = run_ingest(cfg, force=args.force, days=args.days)
    wall = time.perf_counter() - t0
    done = stats[~stats["skipped"]] if len(stats) else stats
    out = p["outputs"] / "ingest"
    out.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out / "ingest_stats.csv", index=False)
    summary = {"wall_seconds": wall, "peak_rss_mb": _peak_rss_mb()}
    for kind, g in done.groupby("kind"):
        summary[kind] = {
            "wafers": len(g),
            "rows": int(g["rows"].sum()),
            "seconds": float(g["seconds"].sum()),
            "rows_per_s": float(g["rows"].sum() / max(g["seconds"].sum(), 1e-9)),
            "mb_in": float(g["bytes_in"].sum() / 1e6),
            "mb_out": float(g["bytes_out"].sum() / 1e6),
            "mb_per_s_in": float(g["bytes_in"].sum() / 1e6 / max(g["seconds"].sum(), 1e-9)),
        }
    (out / "ingest_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_query_bench(cfg: dict, args) -> None:
    from .store import query_benchmark

    p = _paths(cfg)
    res = query_benchmark(p["parquet"])
    out = p["outputs"] / "ingest"
    out.mkdir(parents=True, exist_ok=True)
    (out / "query_benchmark.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def cmd_lines(cfg: dict, args) -> None:
    from .reports import line_detection_report

    p = _paths(cfg)
    md = line_detection_report(
        p["features"], project_path("docs/line_detection.md"), snr_min=cfg.get("lines", {}).get("snr_min", 5.0)
    )
    print(md)


def cmd_cycles(cfg: dict, args) -> None:
    from .features.cycles import build_cycle_tables

    p = _paths(cfg)
    t = build_cycle_tables(p["parquet"], p["features"], bin_width=cfg["features"]["bin_width"], overwrite=args.force)
    print(json.dumps(t, indent=2))


def cmd_features(cfg: dict, args) -> None:
    from .pipeline import build_feature_table

    p = _paths(cfg)
    feats, tg, groups = build_feature_table(cfg, p)
    print(f"features: {feats.shape}, targets: {tg.shape}")
    print({k: len(v) for k, v in groups.items()})


def cmd_benchmark(cfg: dict, args) -> None:
    from .pipeline import run_tabular_benchmark

    res = run_tabular_benchmark(cfg, _paths(cfg), mlflow_run=not args.no_mlflow)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(res.round(4).to_string(index=False))


def cmd_points(cfg: dict, args) -> None:
    from .pipeline import run_point_benchmark

    res = run_point_benchmark(cfg, _paths(cfg))
    print(res.round(4).to_string(index=False))


def cmd_train_nn(cfg: dict, args) -> None:
    from .pipeline import run_nn

    out = run_nn(cfg, _paths(cfg), mlflow_run=not args.no_mlflow)
    print(out["results"].round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in out["stats"].items() if k != "folds"}, indent=2, default=str))


def cmd_interpret(cfg: dict, args) -> None:
    from .pipeline import run_interpret

    run_interpret(cfg, _paths(cfg))


def cmd_export(cfg: dict, args) -> None:
    from .pipeline import export_bundle

    path = export_bundle(cfg, _paths(cfg))
    print(f"wrote {path}")


def cmd_export_v2(cfg: dict, args) -> None:
    from .pipeline import export_bundle_v2

    print(f"wrote {export_bundle_v2(cfg, _paths(cfg))}")


COMMANDS = {
    "download": (cmd_download, "data.yaml"),
    "ingest": (cmd_ingest, "data.yaml"),
    "query-bench": (cmd_query_bench, "data.yaml"),
    "lines": (cmd_lines, "features.yaml"),
    "cycles": (cmd_cycles, "features.yaml"),
    "features": (cmd_features, "features.yaml"),
    "benchmark": (cmd_benchmark, "models/tabular.yaml"),
    "points": (cmd_points, "models/tabular.yaml"),
    "train-nn": (cmd_train_nn, "models/cnn1d.yaml"),
    "interpret": (cmd_interpret, "models/tabular.yaml"),
    "export": (cmd_export, "models/tabular.yaml"),
    "export-v2": (cmd_export_v2, "models/tabular.yaml"),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pevm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--config", default=None, help="YAML under configs/ (default depends on command)")
    ap.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    ap.add_argument("--days", nargs="*", default=None, help="restrict ingestion to these days (YYYY-MM-DD)")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("overrides", nargs="*", help="dotted key=value overrides, e.g. train.epochs=5")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    fn, default_cfg = COMMANDS[args.command]
    cfg = load_config(args.config or default_cfg, args.overrides)
    fn(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
