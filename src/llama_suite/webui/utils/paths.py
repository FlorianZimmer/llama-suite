"""WebUI path helpers (allow override via env vars)."""

from __future__ import annotations

import os
from pathlib import Path

from llama_suite.utils.config_utils import find_project_root


def get_project_root() -> Path:
    """Get the project root directory."""
    return find_project_root()


def get_base_config_path() -> Path:
    """
    Get path to the base config.

    Allows overriding for tests or alternative deployments via `LLAMA_SUITE_BASE_CONFIG_PATH`.
    If the provided value is relative, it is resolved relative to the project root.
    """
    raw = os.getenv("LLAMA_SUITE_BASE_CONFIG_PATH")
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = get_project_root() / p
        return p

    return get_project_root() / "configs" / "config.base.yaml"

