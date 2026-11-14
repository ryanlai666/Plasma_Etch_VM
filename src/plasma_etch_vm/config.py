"""YAML configuration loading.

Configs live in ``configs/`` and are plain YAML. ``load_config`` resolves ``defaults`` includes
(a list of other YAML files merged first, Hydra-style but without the dependency) and applies
dotted ``key=value`` overrides from the command line.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def deep_merge(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``other`` into a copy of ``base`` (``other`` wins)."""
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides (value parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_scalar(raw)
    return cfg


def resolve_path(path: str | Path, base: Path = CONFIG_DIR) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return base / p


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config, resolving ``defaults: [other.yaml, ...]`` relative to ``configs/``."""
    path = resolve_path(path)
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    merged: dict[str, Any] = {}
    for dep in cfg.pop("defaults", []) or []:
        merged = deep_merge(merged, load_config(resolve_path(dep, path.parent)))
    merged = deep_merge(merged, cfg)
    return apply_overrides(merged, overrides)


def project_path(rel: str | Path) -> Path:
    """Resolve a path from a config relative to the project root."""
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p
