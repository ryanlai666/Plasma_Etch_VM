"""Thin MLflow helpers. Local tracking under ``./mlruns``.

MLflow ≥ 3.x puts the plain file store in maintenance mode, so the default here is a SQLite backend
*inside* ``./mlruns`` (``mlruns/mlflow.db``) with artifacts in ``mlruns/artifacts``. View with::

    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
"""

from __future__ import annotations

import os
from typing import Any

from .config import project_path

DEFAULT_URI = "sqlite:./mlruns/mlflow.db"


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def resolve_uri(uri: str) -> tuple[str, str]:
    """Return ``(tracking_uri, artifact_location)`` with project-relative paths made absolute."""
    if uri.startswith("sqlite:./"):
        db = project_path(uri[len("sqlite:./") :])
        db.parent.mkdir(parents=True, exist_ok=True)
        art = db.parent / "artifacts"
        return f"sqlite:///{db.as_posix()}", art.as_uri()
    if uri.startswith("file:./"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        p = project_path(uri[len("file:./") :])
        return p.as_uri(), p.as_uri()
    return uri, ""


def start_run(cfg: dict, run_name: str):
    """Start an MLflow run, log the flattened config as params, return the ``mlflow`` module."""
    import mlflow

    tcfg = cfg.get("tracking", {})
    uri, art = resolve_uri(tcfg.get("uri", DEFAULT_URI))
    os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
    mlflow.set_tracking_uri(uri)
    name = tcfg.get("experiment", "plasma_etch_vm")
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=art or None)
    mlflow.set_experiment(name)
    mlflow.start_run(run_name=run_name)
    params = {k: str(v)[:250] for k, v in _flatten(cfg).items()}
    items = list(params.items())
    for i in range(0, len(items), 90):  # MLflow caps params per batch
        mlflow.log_params(dict(items[i : i + 90]))
    return mlflow
