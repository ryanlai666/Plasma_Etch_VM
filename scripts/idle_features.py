"""EXPLORATORY: do idle time before a wafer and time/wafers since conditioning help Si-depth prediction?

    python scripts/idle_features.py      -> outputs/review/idle_features.{md,json}

Only lot 8 (2024-08-07) has unusually long idle gaps (1411 s / 1127 s before wafers 8 / 9; median 691 s), so this is a
one-lot anecdote test, not evidence. Features (OES epoch clock, plasma-on start of each wafer):
  X_idle_log_gap   log(gap to the previous wafer's plasma start) — NaN for the first wafer of a lot (median-imputed)
  X_idle_t_since_lot_start_s   seconds since the lot's first wafer started (differs from slot only when gaps vary)
  X_idle_wafers_since_cond     slot - 1 (collinear with C_slot by construction; included as requested)
Leave-one-lot-out, paired lot-bootstrap 95 % CIs (2000 draws).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plasma_etch_vm.config import load_config  # noqa: E402
from plasma_etch_vm.models.tabular import build_model  # noqa: E402

FEAT, STORE, OUT = ROOT / "data" / "features", ROOT / "data" / "parquet", ROOT / "outputs" / "review"
IDLE = ["X_idle_log_gap", "X_idle_t_since_lot_start_s", "X_idle_wafers_since_cond"]


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def idle_table() -> pd.DataFrame:
    rows = []
    for p in sorted((STORE / "oes_trace").glob("day=*/lot=*/wafer=*/part-0.parquet")):
        day, w = p.parent.parent.parent.name.split("=")[1], int(p.parent.name.split("=")[1])
        tr = pd.read_parquet(p, columns=["t_s", "phase"])
        rows.append(
            {
                "wafer_id": f"{day}_{w:02d}",
                "day": day,
                "wafer": w,
                "start": float(tr.loc[tr["phase"] != "off", "t_s"].iloc[0]),
            }
        )
    d = pd.DataFrame(rows).sort_values(["day", "wafer"])
    d["X_idle_log_gap"] = np.log(d.groupby("day")["start"].diff())
    d["X_idle_t_since_lot_start_s"] = d["start"] - d.groupby("day")["start"].transform("min")
    d["X_idle_wafers_since_cond"] = d["wafer"] - 1.0
    return d.set_index("wafer_id")[IDLE]


def lolo(model, X, y, lots, rate_n=None):
    pred = np.full(len(y), np.nan)
    for lt in np.unique(lots):
        te = lots == lt
        yfit = y / rate_n if rate_n is not None else y
        m = build_model(*model).fit(X[~te], yfit[~te])
        pred[te] = np.asarray(m.predict(X[te])).ravel() * (rate_n[te] if rate_n is not None else 1.0)
    return pred


def main() -> None:
    cfg = load_config("models/tabular.yaml")
    feats = pd.read_parquet(FEAT / "wafer_features.parquet").join(idle_table())
    tg = pd.read_parquet(FEAT / "wafer_targets.parquet")
    groups = json.loads((FEAT / "feature_groups.json").read_text())
    common = feats.index.intersection(tg["si_etch_mean"].dropna().index)
    y, lots = tg.loc[common, "si_etch_mean"].to_numpy(float), tg.loc[common, "lot"].to_numpy()
    n = feats.loc[common, "TM_tool_etch_n"].to_numpy(float)
    gparams = dict(cfg["model_params"]["gbm"])
    base_slot = groups["context"] + groups["cycle_count"]
    tool_ctx = groups["tool"] + groups["timing"] + groups["context"]
    specs = [
        ("OLS slot + cycles + conditioning", ("ols", None), base_slot, None),
        ("ridge tool + context (served default)", ("ridge", None), tool_ctx, None),
        ("GBM tool + context", ("gbm", gparams), tool_ctx, None),
        ("hybrid n_cycles × GBM rate (v2 candidate)", ("gbm", gparams), tool_ctx, n),
    ]
    ul = np.unique(lots)
    rng = np.random.default_rng(0)
    picks = [np.concatenate([np.flatnonzero(lots == q) for q in rng.choice(ul, len(ul))]) for _ in range(2000)]
    rows = []
    for name, model, cols, rn in specs:
        p0 = lolo(model, feats.loc[common, cols].to_numpy(float), y, lots, rn)
        p1 = lolo(model, feats.loc[common, cols + IDLE].to_numpy(float), y, lots, rn)
        ds = [rmse(y[ii], p1[ii]) - rmse(y[ii], p0[ii]) for ii in picks]
        won = sum(rmse(y[lots == q], p1[lots == q]) < rmse(y[lots == q], p0[lots == q]) for q in ul)
        m8 = lots == 8
        rows.append(
            {
                "model": name,
                "RMSE without idle": rmse(y, p0),
                "RMSE with idle features": rmse(y, p1),
                "Δ [95 % CI]": f"{rmse(y, p1) - rmse(y, p0):+.3f} [{np.percentile(ds, 2.5):+.3f}, {np.percentile(ds, 97.5):+.3f}]",
                "lots improved": f"{won}/{len(ul)}",
                "lot 8 RMSE without → with": f"{rmse(y[m8], p0[m8]):.3f} → {rmse(y[m8], p1[m8]):.3f}",
            }
        )
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "idle_features.csv", index=False)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += [
        "| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |"
        for r in df.itertuples(index=False)
    ]
    (OUT / "idle_features.md").write_text(
        "# EXPLORATORY idle-time features (Si depth, leave-one-lot-out)\n\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )
    (OUT / "idle_features.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
