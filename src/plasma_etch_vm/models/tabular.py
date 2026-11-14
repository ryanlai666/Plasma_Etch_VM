"""Tabular VM models (one regressor per target).

``build_model(name, params)`` returns an *unfitted* sklearn estimator. Every pre-processing step
(imputation, scaling, PCA/NMF on spectra) lives inside the pipeline so it is fit on training folds
only — no leakage through the decomposition.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import NMF, PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:  # pragma: no cover
    HAS_LGB = False


class PLS1(BaseEstimator, RegressorMixin):
    """PLSRegression with a 1-D ``predict`` and ``n_components`` capped by the training size."""

    def __init__(self, n_components: int = 3, scale: bool = False):
        self.n_components = n_components
        self.scale = scale

    def fit(self, X, y):
        k = int(max(1, min(self.n_components, X.shape[1], X.shape[0] - 1)))
        self.model_ = PLSRegression(n_components=k, scale=self.scale).fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X)).ravel()


class CappedPCA(PCA):
    """PCA whose ``n_components`` is capped at ``min(n_samples-1, n_features)`` at fit time."""

    def fit(self, X, y=None):
        self._cap(X)
        return super().fit(X, y)

    def fit_transform(self, X, y=None):
        self._cap(X)
        return super().fit_transform(X, y)

    def _cap(self, X):
        if isinstance(self.n_components, int):
            self.n_components = int(max(1, min(self.n_components, X.shape[0] - 1, X.shape[1])))


class CappedNMF(NMF):
    """NMF whose ``n_components`` is capped at ``min(n_samples, n_features)`` at fit time."""

    def fit_transform(self, X, y=None, W=None, H=None):
        self.n_components = int(max(1, min(self.n_components, X.shape[0], X.shape[1])))
        return super().fit_transform(X, y, W=W, H=H)

    def fit(self, X, y=None, **params):
        self.fit_transform(X)
        return self


def nonneg(x):
    """Clip to >= 0 (NMF input). Module-level so fitted pipelines stay picklable."""
    return np.clip(x, 0, None)


def _base_prep(variance_threshold: float = 0.0):
    return [
        SimpleImputer(strategy="median", keep_empty_features=True),
        VarianceThreshold(variance_threshold),
        StandardScaler(),
    ]


def _ridge(params: dict):
    alphas = np.logspace(*params.get("alpha_logspace", [-3, 4, 15]))
    return RidgeCV(alphas=alphas)


def build_model(name: str, params: dict | None = None, seed: int = 0):
    p = dict(params or {})
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "ols":  # plain least squares: for the small, physically chosen baseline sets only
        return make_pipeline(*_base_prep(), LinearRegression())
    if name == "ridge":
        return make_pipeline(*_base_prep(), _ridge(p))
    if name == "pls":
        return make_pipeline(*_base_prep(), PLS1(n_components=p.get("n_components", 3)))
    if name == "pca_ridge":
        return make_pipeline(
            *_base_prep(), CappedPCA(n_components=p.get("n_components", 10), random_state=seed), _ridge(p)
        )
    if name == "nmf_ridge":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            FunctionTransformer(nonneg),
            CappedNMF(
                n_components=p.get("n_components", 8),
                init="nndsvda",
                max_iter=p.get("max_iter", 500),
                random_state=seed,
            ),
            StandardScaler(),
            _ridge(p),
        )
    if name == "gbm":
        if HAS_LGB and p.pop("backend", "lightgbm") == "lightgbm":
            lgb_params = {
                "n_estimators": 300,
                "learning_rate": 0.03,
                "num_leaves": 7,
                "max_depth": 3,
                "min_child_samples": 5,
                "subsample": 0.8,
                "subsample_freq": 1,
                "colsample_bytree": 0.5,
                "reg_lambda": 1.0,
                "verbose": -1,
                "random_state": seed,
                "n_jobs": 1,
            }
            lgb_params.update(p)
            return Pipeline([("prep", VarianceThreshold(0.0)), ("gbm", lgb.LGBMRegressor(**lgb_params))])
        return HistGradientBoostingRegressor(
            max_iter=p.get("n_estimators", 300),
            learning_rate=p.get("learning_rate", 0.05),
            max_leaf_nodes=p.get("num_leaves", 7),
            min_samples_leaf=p.get("min_child_samples", 5),
            random_state=seed,
        )
    raise ValueError(f"unknown model {name}")


def fit_predict(model, X_tr, y_tr, X_te):
    m = clone(model).fit(X_tr, y_tr)
    return m, np.asarray(m.predict(X_te)).ravel()
