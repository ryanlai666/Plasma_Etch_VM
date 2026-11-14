"""Pack the neural-model inputs + code into one small tar.gz for a remote (Colab) run.

    python scripts/pack_nn_inputs.py

Builds the exact per-wafer tensors that `pevm train-nn` would build locally (same build_inputs code, same cycle
tables), so remote runs see identical inputs; never ships the raw 6 GB Parquet store. Output:
outputs/colab/plasma_nn_pack.tar.gz containing
    pack/inputs/{cnn1d,statfe,statfe_tool}.npz, pack/inputs/wafer_targets.parquet,
    pack/code/{pyproject.toml,README.md,src/,configs/,scripts/colab_run_nn.sh}
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plasma_etch_vm.config import load_config  # noqa: E402
from plasma_etch_vm.models.train_nn import build_inputs  # noqa: E402
from plasma_etch_vm.pipeline import save_nn_inputs  # noqa: E402

OUT = ROOT / "outputs" / "colab"
STAGE = OUT / "pack"
VARIANTS = {
    "cnn1d": ("models/cnn1d.yaml", []),
    "statfe": ("models/statfe_transformer.yaml", []),
    "statfe_tool": ("models/statfe_transformer.yaml", ["model.sources=[process]"]),
}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "inputs").mkdir(parents=True)
    feat = ROOT / "data" / "features"
    cyc_oes = pd.read_parquet(feat / "cycles_oes.parquet")
    cyc_proc = pd.read_parquet(feat / "cycles_process.parquet")
    manifest = {}
    for name, (cfg_path, ov) in VARIANTS.items():
        cfg = load_config(cfg_path, ov)
        X, ids, meta = build_inputs(cfg, cyc_oes, cyc_proc)
        f = STAGE / "inputs" / f"{name}.npz"
        save_nn_inputs(f, X, ids, meta)
        manifest[name] = {"shape": list(X.shape), "n_ids": len(ids), "bytes": f.stat().st_size, "md5": md5(f)}
        print(name, X.shape, f"{f.stat().st_size / 1e6:.2f} MB")
    shutil.copy(feat / "wafer_targets.parquet", STAGE / "inputs" / "wafer_targets.parquet")
    code = STAGE / "code"
    code.mkdir()
    for item in ("pyproject.toml", "README.md"):
        shutil.copy(ROOT / item, code / item)
    shutil.copytree(ROOT / "src", code / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    shutil.copytree(ROOT / "configs", code / "configs")
    (code / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "colab_run_nn.sh", code / "scripts" / "colab_run_nn.sh")
    (STAGE / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    tgz = OUT / "plasma_nn_pack.tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(STAGE, arcname="pack")
    print(f"packed {tgz} : {tgz.stat().st_size / 1e6:.2f} MB (md5 {md5(tgz)})")


if __name__ == "__main__":
    main()
