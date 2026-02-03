"""Memory scan API routes for checking whether models fit in memory.

The intent of a memory scan is to start the model with the current configuration,
wait for it to become healthy, parse memory usage from logs, and stop. It should
not run the normal benchmark (TPS measurement).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import sys

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from llama_suite.utils.config_utils import find_project_root
from llama_suite.webui.utils.process_manager import process_manager
from llama_suite.webui.utils.task_output import handle_task_output
from llama_suite.webui.utils.ws_manager import manager as ws_manager
from llama_suite.webui.utils.mode import require_local_mode


router = APIRouter(prefix="/api/memory", tags=["memory"])


def get_project_root() -> Path:
    """Get the project root directory."""
    return find_project_root()


class MemoryScanRequest(BaseModel):
    """Request to start a memory scan run."""

    override: Optional[str] = None
    model: Optional[str] = None
    health_timeout: int = 120


@router.post("/run")
async def start_memory_scan(request: MemoryScanRequest, _=Depends(require_local_mode)):
    """Start a new memory scan run."""

    async def run_memory_scan_task(task_id: str, override: Optional[str], model: Optional[str], health_timeout: int):
        root = get_project_root()
        venv_python = root / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / "python"
        if not venv_python.exists():
            venv_python = Path(sys.executable)

        scan_script = root / "src" / "llama_suite" / "bench" / "scan_model_memory.py"
        cmd = [
            str(venv_python),
            "-u",
            str(scan_script),
            "--config",
            str(root / "configs" / "config.base.yaml"),
            "--health-timeout",
            str(health_timeout),
        ]

        if override:
            override_path = root / "configs" / "overrides" / f"{override}.yaml"
            cmd.extend(["--override", str(override_path)])

        if model:
            cmd.extend(["--model", model])

        async def on_output(line: str):
            await handle_task_output(ws_manager, task_id, line, is_stderr=False, progress_style="steps")

        async def on_error(line: str):
            await handle_task_output(ws_manager, task_id, line, is_stderr=True, progress_style="steps")

        await ws_manager.send_progress(task_id, 0, "Starting memory scan...")

        returncode = await process_manager.run_subprocess(
            task_id,
            cmd,
            cwd=root,
            on_stdout=on_output,
            on_stderr=on_error,
        )

        success = returncode == 0
        await ws_manager.send_complete(task_id, success, {"returncode": returncode})

        return {"returncode": returncode, "success": success}

    task_id = await process_manager.start_task(
        task_type="memory_scan",
        description="Memory scan" + (f" for {request.model}" if request.model else ""),
        coro=run_memory_scan_task,
        override=request.override,
        model=request.model,
        health_timeout=request.health_timeout,
    )

    return {"task_id": task_id, "status": "started"}


@router.post("/cancel/{task_id}")
async def cancel_memory_scan(task_id: str):
    """Cancel a running memory scan."""
    cancelled = await process_manager.cancel_task(task_id)
    if cancelled:
        return {"status": "cancelled", "task_id": task_id}
    raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found or already completed")
