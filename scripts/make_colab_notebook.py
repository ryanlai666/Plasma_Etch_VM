"""Generate notebooks/colab_launcher.ipynb (kept as code so the notebook stays reviewable / diffable).

python scripts/make_colab_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


CELLS = [
    md("""
# Plasma-etch VM — Colab launcher

Runs the pipeline on Google Colab with everything that must survive a disconnect on **Google Drive**:
raw data, Parquet store, features, MLflow runs and training checkpoints. Colab has **no background
execution**: keep the tab open. Training scripts checkpoint every 20 min (`checkpoint.every_minutes`)
and at the end of every CV fold; re-running the training cell resumes where it stopped.

**The download cell** fetches the small files and one day file by default; the other nine day files add ~7.4 GB
(set `FULL_DOWNLOAD = True`). Runtime: CPU is enough for ingestion and tabular models; pick a T4/L4 GPU
only for the neural models (they need < 1 GB VRAM).
"""),
    md("## 1. Mount Drive and choose the persistent workspace"),
    code("""
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE = '/content/drive/MyDrive/plasma_etch_vm'   # persistent: data, features, mlruns, checkpoints, outputs
os.makedirs(DRIVE, exist_ok=True)
for sub in ['data/raw', 'data/parquet', 'data/features', 'mlruns', 'checkpoints', 'outputs']:
    os.makedirs(f'{DRIVE}/{sub}', exist_ok=True)
print(DRIVE)
"""),
    md("""
## 2. Get the repository onto Colab

Option A — the repo is on GitHub (set `REPO_URL`). Option B — upload a zip of the project folder
(without `.venv` and `data/`) to `MyDrive/plasma_etch_vm/plasma_etch_vm_src.zip`.
"""),
    code("""
REPO_URL = ''   # e.g. 'https://github.com/<user>/plasma-etch-vm.git'  (left empty on purpose)
WORK = '/content/plasma_etch_vm'
import os, shutil, subprocess
if not os.path.exists(WORK):
    if REPO_URL:
        subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, WORK], check=True)
    else:
        shutil.unpack_archive(f'{DRIVE}/plasma_etch_vm_src.zip', WORK)
%cd {WORK}
!ls
"""),
    md("## 3. Link persistent folders from Drive into the working copy and install"),
    code("""
import os
for sub in ['data/raw', 'data/parquet', 'data/features', 'mlruns', 'checkpoints', 'outputs']:
    local = os.path.join(WORK, sub)
    if os.path.islink(local):
        continue
    if os.path.isdir(local):
        os.system(f'cp -rn "{local}/." "{DRIVE}/{sub}/" && rm -rf "{local}"')
    os.makedirs(os.path.dirname(local), exist_ok=True)
    os.symlink(f'{DRIVE}/{sub}', local)
# Colab already ships a CUDA build of torch; install the package and its extras on top.
!pip -q install -e ".[serve,interp,dev]"
!pevm --help | head -20
"""),
    md("""
## 4. Data (resumable, verified by MD5)

`zenodo.day_files=all` downloads the nine remaining day files (7.4 GB). Files already on Drive and
verified are skipped, partial files are resumed.
"""),
    code("""
FULL_DOWNLOAD = False   # set True to download all ten OES day files (~7.4 GB)
if FULL_DOWNLOAD:
    !pevm download zenodo.day_files=all zenodo.min_free_gb=5
else:
    !pevm download            # small files + the one default day file
"""),
    md("## 5. Ingest -> cycle tables -> features (idempotent: finished wafers are skipped)"),
    code("""
!pevm ingest
!pevm cycles
!pevm lines
!pevm features
"""),
    md("## 6. Tabular benchmark (CPU, minutes) + per-site model + interpretation + serving bundle"),
    code("""
!pevm benchmark
!pevm points
!pevm interpret
!pevm export
"""),
    md("""
## 7. Neural models (GPU) — checkpoint every 20 min to Drive, resumable

Re-running this cell after a disconnect resumes: finished CV folds are read back from
`outputs/nn/<run_name>/fold_k.json`, the running fold restarts from `checkpoints/<model>/latest.pt`.
"""),
    code("""
import torch; print(torch.__version__, torch.cuda.is_available())
RUNS = [("models/cnn1d.yaml", "cnn1d"), ("models/statfe_transformer.yaml", "statfe")]
for seed in [0, 1, 2]:
    for cfg, name in RUNS:
        !pevm train-nn --config {cfg} seed={seed} run_name={name}_s{seed} checkpoint.dir=checkpoints/{name}_s{seed}
"""),
    md("## 8. Inspect results"),
    code("""
import pandas as pd, glob
print(pd.read_csv('outputs/benchmark/benchmark_all.csv').query("feature_set=='oes_tool_context'").round(4))
for f in sorted(glob.glob('outputs/nn/*/nn_results.csv')):
    print(f); print(pd.read_csv(f).round(4))
# MLflow UI (optional): !mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
"""),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "notebooks" / "colab_launcher.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
