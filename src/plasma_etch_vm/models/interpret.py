"""Feature attribution: grouped-CV permutation importance and SHAP (tree models).

Permutation importance is computed on held-out lots (each fold's model scored on its own test lot),
so it measures what generalises across chamber states rather than what memorises a lot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance


def cv_permutation_importance(
    model, X: pd.DataFrame, y: np.ndarray, splits, n_repeats: int = 10, seed: int = 0, top: int | None = 30
) -> pd.DataFrame:
    imps = []
    for tr, te in splits:
        if len(te) < 2:
            continue
        m = clone(model).fit(X.iloc[tr].to_numpy(), y[tr])
        r = permutation_importance(
            m,
            X.iloc[te].to_numpy(),
            y[te],
            n_repeats=n_repeats,
            random_state=seed,
            scoring="neg_root_mean_squared_error",
        )
        imps.append(r.importances_mean)
    imp = np.vstack(imps)
    df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": imp.mean(axis=0),
            "importance_std": imp.std(axis=0),
            "n_folds": imp.shape[0],
        }
    )
    df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df.head(top) if top else df


def shap_tree_importance(model, X: pd.DataFrame, y: np.ndarray, top: int | None = 30) -> pd.DataFrame:
    """Mean |SHAP| of a tree model fit on all wafers (descriptive, not a CV estimate)."""
    import shap

    m = clone(model).fit(X.to_numpy(), y)
    est = m.steps[-1][1] if hasattr(m, "steps") else m
    Xt = X.to_numpy()
    cols = list(X.columns)
    if hasattr(m, "steps") and len(m.steps) > 1:
        prep = m[:-1]
        Xt = prep.transform(Xt)
        support = prep[-1].get_support() if hasattr(prep[-1], "get_support") else np.ones(len(cols), bool)
        cols = [c for c, s in zip(cols, support) if s]
    sv = shap.TreeExplainer(est).shap_values(Xt)
    df = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df.head(top) if top else df
