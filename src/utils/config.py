"""
src/utils/config.py
-------------------
Load and expose the central config.yaml as a typed namespace.
"""

from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Any

import yaml


_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Load config.yaml from the project root (cached after first call).

    Args:
        path: Optional explicit path to a config file. Defaults to
              ``<project_root>/config.yaml``.

    Returns:
        Parsed YAML as a plain Python dict.
    """
    config_path = pathlib.Path(path) if path else _PROJECT_ROOT / "config.yaml"
    with config_path.open("r") as fh:
        return yaml.safe_load(fh)


def get(key: str, default: Any = None) -> Any:
    """Dot-notation accessor for nested config keys.

    Example::

        get("model.baseline.n_estimators")  # -> 300
    """
    cfg = load_config()
    keys = key.split(".")
    node: Any = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node
