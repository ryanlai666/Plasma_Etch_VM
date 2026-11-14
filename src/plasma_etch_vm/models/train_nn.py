"""Group-CV training of the neural VM models with checkpoint / resume.

* Checkpoints (model, optimiser, scheduler, RNG states, fold, epoch) are written atomically every
  ``checkpoint.every_minutes`` (default 20, Colab-safe) and at the end of every fold; a finished
  fold leaves ``fold_<k>.json`` with its predictions so a resumed run skips it.
* ``smoke`` settings (``max_folds``, ``epochs``) are used to measure throughput and peak VRAM.
* Intervals: cross-conformal — for test fold k the quantile of |OOF residuals| of all *other* folds.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .benchmark import regression_metrics
from .conformal import coverage, mean_width, split_conformal_quantile
from .cv import make_splits
from .nn import SpectralCNN1D, StatFETransformer, TokenScaler, count_params, spectra_windows, statfe_tokens

log = logging.getLogger(__name__)


def cap_gpu_memory() -> float:
    """Honour an optional GPU-memory reservation on a shared machine.

    When a GPU-scheduler wrapper (see ``GPU_GUARD`` in ``scripts/run_campaign.py``) sets the env var
    ``GPU_GUARD_GB``, the CUDA caching allocator is capped to that many GB. Returns the cap in GB (0 = none).
    """
    gb = float(os.environ.get("GPU_GUARD_GB", "0") or 0)
    if gb and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2**30 / total))
        log.info("GPU memory capped to %.2f GB (GPU_GUARD_GB reservation)", gb)
    elif torch.cuda.is_available():
        log.info("no GPU memory reservation (GPU_GUARD_GB unset); using the whole device")
    return gb


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - global state is what must be checkpointed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])  # noqa: NPY002
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


# ----------------------------------------------------------------------------------- data prep
def build_inputs(cfg: dict, cyc_oes: pd.DataFrame | None, cyc_proc: pd.DataFrame | None):
    """Return ``(X, wafer_ids, meta)``. CNN: X (N, W, C, L); StatFE: X (N, p*w, 6)."""
    mcfg = cfg["model"]
    if mcfg["type"] == "cnn1d":
        X, ids = spectra_windows(cyc_oes, n_windows=mcfg["n_windows"], max_cycle=mcfg.get("max_cycle", 100))
        return X, ids, {"n_phases": X.shape[2], "n_bins": X.shape[3]}
    if mcfg["type"] == "statfe":
        tables = []
        if "process" in mcfg["sources"] and cyc_proc is not None:
            tables.append(cyc_proc[[c for c in cyc_proc.columns if not c.startswith("Psd_")]])
        if "oes" in mcfg["sources"] and cyc_oes is not None:
            keep = [c for c in cyc_oes.columns if not (c.startswith("b") and c[1:].isdigit())]
            t = cyc_oes[keep].copy()
            lcols = [c for c in t.columns if c.startswith("L_")]
            t[lcols] = t[lcols].div(t["total"], axis=0) * 1e3
            if mcfg.get("oes_lines"):
                drop = [c for c in lcols if c[2:] not in mcfg["oes_lines"]]
                t = t.drop(columns=drop)
            tables.append(t)
        X, ids, sensors = statfe_tokens(tables, n_windows=mcfg["n_windows"], max_cycle=mcfg.get("max_cycle", 100))
        return X, ids, {"n_sensors": len(sensors), "sensors": sensors}
    raise ValueError(mcfg["type"])


def build_net(cfg: dict, meta: dict, n_targets: int) -> nn.Module:
    m = cfg["model"]
    if m["type"] == "cnn1d":
        return SpectralCNN1D(
            meta["n_phases"],
            n_targets,
            width=m.get("width", 32),
            depth=m.get("depth", 4),
            dropout=m.get("dropout", 0.2),
        )
    return StatFETransformer(
        meta["n_sensors"],
        m["n_windows"],
        6,
        n_targets,
        d_model=m.get("d_model"),
        n_heads=m.get("n_heads", 1),
        n_layers=m.get("n_layers", 1),
        ff_dim=m.get("ff_dim", 32),
        dropout=m.get("dropout", 0.1),
    )


# ------------------------------------------------------------------------------------ training
def _fold_arrays(cfg, X, Y, tr, te):
    """Instance arrays for one fold (+ per-fold scaling, fitted on training wafers only)."""
    mt = cfg["model"]["type"]
    mu, sd = np.nanmean(Y[tr], axis=0), np.nanstd(Y[tr], axis=0) + 1e-9
    Ys = (Y - mu) / sd
    if mt == "cnn1d":
        W = X.shape[1]
        xs_mu = X[tr].mean(axis=(0, 1), keepdims=True)
        xs_sd = X[tr].std(axis=(0, 1), keepdims=True) + 1e-6
        Xn = ((X - xs_mu) / xs_sd).astype(np.float32)
        xtr = Xn[tr].reshape(-1, *X.shape[2:])
        ytr = np.repeat(Ys[tr], W, axis=0)
        xte = Xn[te].reshape(-1, *X.shape[2:])
        return xtr, ytr, xte, W, mu, sd
    sc = TokenScaler().fit(X[tr])
    return sc.transform(X[tr]), Ys[tr], sc.transform(X[te]), 1, mu, sd


def train_cv(cfg: dict, X: np.ndarray, ids: list[str], meta: dict, targets: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tcfg, ccfg = cfg["train"], cfg.get("checkpoint", {})
    tnames = cfg["targets"]
    tg = targets.reindex(ids)
    keep = tg[tnames].notna().all(axis=1).to_numpy()
    X, ids, tg = X[keep], [i for i, k in zip(ids, keep) if k], tg[keep]
    Y = tg[tnames].to_numpy(np.float32)
    groups = tg["lot"].to_numpy()
    splits, scheme = make_splits(
        groups, cfg["cv"]["scheme"], cfg["cv"].get("n_splits", 5), cfg["cv"].get("min_groups", 3)
    )
    if tcfg.get("max_folds"):
        splits = splits[: tcfg["max_folds"]]
    device = torch.device("cuda" if torch.cuda.is_available() and tcfg.get("device", "auto") != "cpu" else "cpu")
    stats_cap = cap_gpu_memory() if device.type == "cuda" else 0.0
    ckpt_path = Path(ccfg.get("dir", out_dir / "checkpoints")) / "latest.pt"
    every_s = 60 * ccfg.get("every_minutes", 20)
    oof = np.full(Y.shape, np.nan, dtype=np.float64)
    fold_of = np.full(len(ids), -1)
    stats = {
        "device": str(device),
        "n_wafers": len(ids),
        "cv": scheme,
        "n_folds": len(splits),
        "folds": [],
        "gpu_guard_gb": stats_cap,
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for k, (tr, te) in enumerate(splits):
        fold_file = out_dir / f"fold_{k}.json"
        if fold_file.exists():  # resume: finished fold
            rec = json.loads(fold_file.read_text())
            oof[te] = np.array(rec["pred"])
            fold_of[te] = k
            stats["folds"].append(rec["stats"])
            continue
        set_seed(cfg.get("seed", 0) + k)
        xtr, ytr, xte, W, mu, sd = _fold_arrays(cfg, X, Y, tr, te)
        net = build_net(cfg, meta, len(tnames)).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 1e-2))
        epochs = tcfg["epochs"]
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
        start_epoch = 0
        ck = load_checkpoint(ckpt_path)
        if ck is not None and ck.get("fold") == k and ck.get("config_tag") == cfg.get("run_name"):
            net.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            _set_rng_state(ck["rng"])
            start_epoch = ck["epoch"] + 1
            log.info("resumed fold %d at epoch %d", k, start_epoch)
        xt = torch.from_numpy(xtr).to(device)
        yt = torch.from_numpy(ytr.astype(np.float32)).to(device)
        bs = tcfg["batch_size"]
        loss_fn = nn.SmoothL1Loss(beta=tcfg.get("huber_beta", 1.0))
        last_ck = time.perf_counter()
        t0 = time.perf_counter()
        seen = 0
        for epoch in range(start_epoch, epochs):
            net.train()
            perm = torch.randperm(len(xt), device=device)
            for i in range(0, len(xt), bs):
                b = perm[i : i + bs]
                xb = xt[b]
                if cfg["model"]["type"] == "cnn1d" and tcfg.get("noise_std", 0) > 0:
                    xb = xb + tcfg["noise_std"] * torch.randn_like(xb)
                loss = loss_fn(net(xb), yt[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                seen += len(b)
            sched.step()
            if time.perf_counter() - last_ck > every_s or epoch == epochs - 1:
                save_checkpoint(
                    ckpt_path,
                    {
                        "fold": k,
                        "epoch": epoch,
                        "model": net.state_dict(),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "rng": _rng_state(),
                        "config_tag": cfg.get("run_name"),
                    },
                )
                last_ck = time.perf_counter()
        train_s = time.perf_counter() - t0
        net.eval()
        with torch.no_grad():
            p = net(torch.from_numpy(xte).to(device)).cpu().numpy().astype(np.float64)
        p = p.reshape(len(te), W, -1).mean(axis=1) * sd + mu
        oof[te] = p
        fold_of[te] = k
        fs = {
            "fold": k,
            "train_seconds": train_s,
            "instances_per_s": seen / max(train_s, 1e-9),
            "n_train_instances": len(xt),
            "epochs_run": epochs - start_epoch,
        }
        stats["folds"].append(fs)
        fold_file.write_text(json.dumps({"pred": p.tolist(), "stats": fs}))
    stats["n_params"] = count_params(build_net(cfg, meta, len(tnames)))
    stats["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0

    done = fold_of >= 0
    alpha = cfg.get("conformal", {}).get("alpha", 0.1)
    rows, pred_rows = [], []
    for j, t in enumerate(tnames):
        res = np.abs(Y[:, j] - oof[:, j])
        lo, hi = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
        for k in np.unique(fold_of[done]):
            other = done & (fold_of != k)
            q = split_conformal_quantile(res[other], alpha) if other.any() else np.inf
            m = fold_of == k
            lo[m], hi[m] = oof[m, j] - q, oof[m, j] + q
        base = np.array([np.nanmean(Y[np.array(splits[f][0]), j]) if f >= 0 else np.nan for f in fold_of])
        met = regression_metrics(Y[done, j], oof[done, j], base[done])
        met.update(coverage=coverage(Y[done, j], lo[done], hi[done]), interval_width=mean_width(lo[done], hi[done]))
        rows.append({"target": t, "model": cfg["model"]["type"], **met})
        pred_rows.append(
            pd.DataFrame(
                {
                    "wafer_id": ids,
                    "lot": groups,
                    "target": t,
                    "y": Y[:, j],
                    "pred": oof[:, j],
                    "lo": lo,
                    "hi": hi,
                    "fold": fold_of,
                }
            )
        )
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "nn_results.csv", index=False)
    pd.concat(pred_rows).to_parquet(out_dir / "nn_oof_predictions.parquet", index=False)
    (out_dir / "nn_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    return {"results": results, "stats": stats}
