from __future__ import annotations

import os
from enum import Enum

from fastapi import HTTPException, status


class LlamaSuiteMode(str, Enum):
    LOCAL = "local"
    GITOPS = "gitops"


def get_mode() -> LlamaSuiteMode:
    raw = (os.getenv("LLAMA_SUITE_MODE") or "local").strip().lower()
    if raw == LlamaSuiteMode.GITOPS.value:
        return LlamaSuiteMode.GITOPS
    return LlamaSuiteMode.LOCAL


def get_capabilities() -> dict[str, bool | str]:
    """
    Capabilities derived from the current mode.

    In `gitops` mode the Web UI is endpoint-only: no subprocess spawning and no writes under
    configs/models/runs via the API.
    """
    mode = get_mode()
    is_local = mode == LlamaSuiteMode.LOCAL
    return {
        "mode": mode.value,
        "can_spawn_subprocesses": is_local,
        "can_write_configs": is_local,
        "can_write_models": is_local,
        "can_delete_runs": is_local,
    }


def require_local_mode() -> None:
    if get_mode() != LlamaSuiteMode.LOCAL:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is disabled in LLAMA_SUITE_MODE=gitops",
        )


def require_not_read_only() -> None:
    if get_mode() != LlamaSuiteMode.LOCAL:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is read-only in LLAMA_SUITE_MODE=gitops",
        )

