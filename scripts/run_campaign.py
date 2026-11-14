"""Resumable end-to-end campaign: download -> ingest -> CPU analyses -> neural-model runs.

    python scripts/run_campaign.py [--stages data,cpu,gpu] [--dl-workers 3] [--dry-run]

Stages (each step is skipped if ``outputs/campaign_state.json`` marks it done):

* ``data`` — per remaining day file: download (3 parallel streams, resumable ``.part`` files, MD5),
  ingest that day (netCDF -> Parquet), verify the Parquet row counts against the netCDF, then delete
  the raw day file (keeps peak disk use at about one day file). Small files are kept. The pilot day
  2024-08-05 (ingested first) is verified and deleted last.
* ``cpu`` — cycles, lines, features, benchmark, points, interpret, export, query-bench, figures.
* ``gpu`` — R3 CNN ×3 seeds, R4 StatFE (tool+OES) ×3, R5 StatFE (tool only) ×3 (``--with-sweep`` adds an
  optional, non-nested sweep R6: 12 CNN + 12 StatFE configs, seed 0). If the env var ``GPU_GUARD`` points at a
  GPU-scheduler script (``python $GPU_GUARD run --gb X --agent NAME -- cmd``), every run is wrapped by it with a
  1.0 GB memory reservation; otherwise runs start directly. Writes ``outputs/gpu_runs.md`` at the end.

Everything is logged to ``outputs/campaign.log`` (start/end, exit code, wall time, peak VRAM).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plasma_etch_vm import netcdf_io as nio  # noqa: E402
from plasma_etch_vm import zenodo  # noqa: E402
from plasma_etch_vm.config import load_config, project_path  # noqa: E402
from plasma_etch_vm.ingest import Manifest, ingest_oes_file  # noqa: E402
from plasma_etch_vm.metrology import DATE_TO_LOT, day_from_filename  # noqa: E402

OUT = ROOT / "outputs"
LOG = OUT / "campaign.log"
STATE = OUT / "campaign_state.json"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe") if os.name == "nt" else str(ROOT / ".venv" / "bin" / "python")
GUARD = os.environ.get("GPU_GUARD", "")  # optional GPU-scheduler wrapper script; unset -> run without it
GUARD_NAME = os.environ.get("GPU_GUARD_NAME", "plasma-etch-vm")
GPU_GB = 1.0


# ------------------------------------------------------------------------------------ helpers
def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, STATE)


def mark(state: dict, key: str, **info) -> None:
    state[key] = {"done": True, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}
    save_state(state)


def done(state: dict, key: str) -> bool:
    return bool(state.get(key, {}).get("done"))


def run_cmd(args: list[str], name: str, env: dict | None = None) -> tuple[int, float]:
    t0 = time.time()
    log(f"START {name}: {' '.join(args)}")
    with open(OUT / "logs" / f"{name}.log", "a", encoding="utf-8") as fh:
        rc = subprocess.call(args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    wall = time.time() - t0
    log(f"END   {name}: exit={rc} wall={wall:.1f}s")
    return rc, wall


# --------------------------------------------------------------------------------------- data
def verify_day(cfg: dict, raw_file: Path) -> dict:
    """Parquet rows per wafer (OES partitions + trace) must equal the netCDF time length."""
    import pyarrow.parquet as pq

    day = day_from_filename(raw_file)
    lot = DATE_TO_LOT[day]
    root = Path(cfg["paths"]["parquet"])
    res = {"day": day, "wafers": 0, "rows": 0}
    for g in nio.list_groups(raw_file):
        w = int(g.split("_")[-1])
        n = nio.group_info(raw_file, g, "oes").n_time
        d = root / "oes" / f"day={day}" / f"lot={lot}" / f"wafer={w}"
        got = sum(pq.ParquetFile(f).metadata.num_rows for f in d.rglob("*.parquet"))
        tr = root / "oes_trace" / f"day={day}" / f"lot={lot}" / f"wafer={w}" / "part-0.parquet"
        ntr = pq.ParquetFile(tr).metadata.num_rows if tr.exists() else -1
        cal = root / "meta" / "wavelength_calibration" / f"{day}_{w:02d}.json"
        if got != n or ntr != n or not cal.exists():
            raise RuntimeError(f"verification failed for {day} wafer {w}: parquet {got}, trace {ntr}, netCDF {n}")
        res["wafers"] += 1
        res["rows"] += n
    return res


def ingest_and_clean(cfg: dict, raw_file: Path, state: dict) -> None:
    key = f"data:{raw_file.name}"
    root = Path(cfg["paths"]["parquet"])
    t0 = time.time()
    stats = ingest_oes_file(cfg, raw_file, root, Manifest(root))
    t_ing = time.time() - t0
    rows = sum(s.rows for s in stats if not s.skipped)
    ver = verify_day(cfg, raw_file)
    size = raw_file.stat().st_size
    raw_file.unlink()
    log(
        f"DATA  {raw_file.name}: ingested {rows} rows in {t_ing:.1f}s, verified {ver['wafers']} wafers / "
        f"{ver['rows']} rows, deleted raw ({size / 1e6:.0f} MB)"
    )
    mark(state, key, ingest_seconds=t_ing, rows=ver["rows"], wafers=ver["wafers"], raw_bytes=size)


def stage_data(cfg: dict, state: dict, workers: int) -> None:
    raw = Path(cfg["paths"]["raw"])
    record = zenodo.fetch_record(cfg["zenodo"]["api_url"], cache=raw / "zenodo_record_17122442.json")
    files = zenodo.list_files(record)
    days = sorted(k for k in files if k.startswith("Day_"))
    already = "Day_2024_08_05.nc"
    todo = [d for d in days if d != already and not done(state, f"data:{d}")]
    log(f"DATA  stage: {len(todo)} day files to fetch with {workers} parallel streams")
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(zenodo.download, files[d], raw, cfg["zenodo"].get("min_free_gb", 15.0)): d for d in todo}
        for fut in cf.as_completed(futs):
            d = futs[fut]
            try:
                path = fut.result()
            except Exception as exc:
                log(f"FAIL  download {d}: {exc}")
                continue
            log(f"DATA  {d}: downloaded + MD5 verified ({files[d].size / 1e6:.0f} MB, {time.time() - t0:.0f}s elapsed)")
            try:
                ingest_and_clean(cfg, path, state)
            except Exception as exc:
                log(f"FAIL  ingest/verify {d}: {exc} (raw file kept)")
    # the day ingested first (pilot): verify, then delete
    f = raw / already
    if f.exists() and not done(state, f"data:{already}"):
        ver = verify_day(cfg, f)
        size = f.stat().st_size
        f.unlink()
        log(f"DATA  {already}: verified {ver['wafers']} wafers / {ver['rows']} rows (ingested earlier), deleted raw")
        mark(state, f"data:{already}", rows=ver["rows"], wafers=ver["wafers"], raw_bytes=size)
    missing = [d for d in days if not done(state, f"data:{d}")]
    zenodo.write_manifest(files, raw, raw / "MANIFEST.json")
    if missing:
        raise RuntimeError(f"data stage incomplete: {missing}")
    log(f"DATA  stage complete in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------------------- cpu
CPU_STEPS = [
    ("cycles", ["cycles"]),
    ("lines", ["lines"]),
    ("features", ["features"]),
    ("benchmark", ["benchmark"]),
    ("points", ["points"]),
    ("interpret", ["interpret"]),
    ("export", ["export"]),
    ("query-bench", ["query-bench"]),
]


def stage_cpu(state: dict) -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MLFLOW_DISABLE_AGENT_HINT="1", PYTHONIOENCODING="utf-8")
    for name, args in CPU_STEPS:
        key = f"cpu:{name}"
        if done(state, key):
            continue
        log(f"CPU   {name}: free RAM {wait_for_ram():.1f} GB")
        rc, wall = run_cmd([PY, "-m", "plasma_etch_vm.cli", *args], f"cpu_{name}", env)
        if rc != 0:
            raise RuntimeError(f"cpu step {name} failed (exit {rc}); see outputs/logs/cpu_{name}.log")
        mark(state, key, wall_seconds=wall)
    if not done(state, "cpu:figures"):
        rc, wall = run_cmd([PY, "scripts/render_figures.py", "--png"], "cpu_figures", env)
        if rc == 0:
            mark(state, "cpu:figures", wall_seconds=wall)


# ---------------------------------------------------------------------------------------- gpu
def gpu_runs(with_sweep: bool = False) -> list[dict]:
    """R3-R5 (3 seeds each). ``with_sweep`` adds R6, a hyper-parameter sweep that is NOT nested inside the
    leave-one-lot-out folds (its best score would be optimistic); it is not part of the reported results."""
    runs = []
    for s in (0, 1, 2):
        runs.append({"id": f"R3_cnn1d_s{s}", "config": "models/cnn1d.yaml", "overrides": [f"seed={s}"]})
    for s in (0, 1, 2):
        runs.append({"id": f"R4_statfe_s{s}", "config": "models/statfe_transformer.yaml", "overrides": [f"seed={s}"]})
    for s in (0, 1, 2):
        runs.append(
            {
                "id": f"R5_statfe_tool_s{s}",
                "config": "models/statfe_transformer.yaml",
                "overrides": [f"seed={s}", "model.sources=[process]"],
            }
        )
    if not with_sweep:
        return runs
    for w, d, lr in itertools.product((16, 32, 64), (3, 4), (1e-3, 3e-4)):
        runs.append(
            {
                "id": f"R6_cnn_w{w}_d{d}_lr{lr:g}",
                "config": "models/cnn1d.yaml",
                "overrides": ["seed=0", f"model.width={w}", f"model.depth={d}", f"train.lr={lr}"],
            }
        )
    for (dm, nh), nl, dr in itertools.product(((None, 1), (16, 2), (32, 4)), (1, 2), (0.1, 0.3)):
        runs.append(
            {
                "id": f"R6_statfe_dm{dm or 'paper'}_l{nl}_do{dr}",
                "config": "models/statfe_transformer.yaml",
                "overrides": [
                    "seed=0",
                    f"model.d_model={'null' if dm is None else dm}",
                    f"model.n_heads={nh}",
                    f"model.n_layers={nl}",
                    f"model.dropout={dr}",
                ],
            }
        )
    return runs


def wait_for_ram(min_gb: float = 8.0, poll_s: int = 30) -> float:
    """Block until at least ``min_gb`` RAM is free (keeps headroom on a shared machine)."""
    import psutil

    while True:
        free = psutil.virtual_memory().available / 2**30
        if free >= min_gb:
            return free
        log(f"WAIT  free RAM {free:.1f} GB < {min_gb} GB")
        time.sleep(poll_s)


def stage_gpu(state: dict, dry: bool = False, with_sweep: bool = False) -> None:
    runs = gpu_runs(with_sweep)
    wrapper = f"via {GUARD} ({GPU_GB} GB each)" if GUARD else "directly (GPU_GUARD unset)"
    log(f"GPU   stage: {len(runs)} runs {wrapper}")
    for r in runs:
        key = f"gpu:{r['id']}"
        if done(state, key):
            continue
        out_dir = OUT / "nn" / r["id"]
        prefix = (
            [sys.executable, GUARD, "run", "--agent", GUARD_NAME, "--gb", str(GPU_GB), "--note", r["id"], "--"]
            if GUARD
            else []
        )
        cmd = [
            *prefix,
            PY,
            "-m",
            "plasma_etch_vm.cli",
            "train-nn",
            "--config",
            r["config"],
            f"run_name={r['id']}",
            f"checkpoint.dir=checkpoints/{r['id']}",
            *r["overrides"],
        ]
        if dry:
            log(f"DRY   {' '.join(cmd)}")
            continue
        env = dict(os.environ, MLFLOW_DISABLE_AGENT_HINT="1", PYTHONIOENCODING="utf-8")
        log(f"GPU   {r['id']}: free RAM {wait_for_ram():.1f} GB, starting")
        rc, wall = run_cmd(cmd, f"gpu_{r['id']}", env)
        stats_f = out_dir / "nn_stats.json"
        peak = json.loads(stats_f.read_text()).get("peak_vram_mb") if stats_f.exists() else None
        log(f"GPU   {r['id']}: exit={rc} wall={wall:.1f}s peak_vram_alloc_mb={peak}")
        if rc == 0 and (out_dir / "nn_results.csv").exists():
            reservation = GPU_GB if GUARD else None
            mark(state, key, wall_seconds=wall, peak_vram_mb=peak, gpu_reservation_gb=reservation, exit=rc)
        else:
            state[key] = {"done": False, "exit": rc, "wall_seconds": wall, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            save_state(state)
    if not dry:
        write_gpu_done(state, runs)


def write_gpu_done(state: dict, runs: list[dict]) -> None:
    rows, total = [], 0.0
    for r in runs:
        st = state.get(f"gpu:{r['id']}", {})
        status = "done" if st.get("done") else (f"failed (exit {st.get('exit')})" if st else "not run")
        wall = st.get("wall_seconds")
        total += wall or 0
        peak = st.get("peak_vram_mb")
        rows.append(f"| {r['id']} | {status} | {wall:.0f} s |" if wall else f"| {r['id']} | {status} | - |")
        rows[-1] += f" {peak:.0f} MB |" if peak else " - |"
        rows[-1] += f" {GPU_GB} GB |" if GUARD else " none |"
    n_ok = sum(1 for r in runs if state.get(f"gpu:{r['id']}", {}).get("done"))
    text = [
        "# Neural-model runs",
        "",
        f"{n_ok}/{len(runs)} runs succeeded. Total wall time: {total / 60:.1f} min (incl. any GPU-wrapper wait).",
        "Peak VRAM = torch.cuda.max_memory_allocated of the training process (CUDA context ~0.4 GB extra);",
        "with a reservation, each job capped its PyTorch memory to it.",
        "",
        "| run | status | wall time | peak VRAM (allocated) | GPU reservation |",
        "|---|---|---|---|---|",
        *rows,
        "",
        f"Written {time.strftime('%Y-%m-%d %H:%M:%S')} by scripts/run_campaign.py (log: outputs/campaign.log).",
    ]
    (OUT / "gpu_runs.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    log("GPU   wrote outputs/gpu_runs.md")


# --------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="data,cpu,gpu")
    ap.add_argument("--dl-workers", type=int, default=3)  # <= 4 parallel download streams
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--with-sweep", action="store_true", help="add the non-nested R6 sweep (not reported)")
    a = ap.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    cfg = load_config("data.yaml")
    cfg["paths"] = {k: str(project_path(v)) for k, v in cfg["paths"].items()}
    state = load_state()
    stages = a.stages.split(",")
    log(f"CAMPAIGN start stages={stages} pid={os.getpid()}")
    try:
        if "data" in stages:
            stage_data(cfg, state, a.dl_workers)
        if "cpu" in stages:
            stage_cpu(state)
        if "gpu" in stages:
            stage_gpu(state, a.dry_run, a.with_sweep)
    except Exception as exc:
        log(f"CAMPAIGN error: {exc}")
        return 1
    log("CAMPAIGN finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
