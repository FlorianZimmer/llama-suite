"""Watcher API routes for managing the llama-swap watcher."""

from pathlib import Path
from typing import Optional
import sys

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llama_suite.utils.config_utils import find_project_root
from llama_suite.webui.utils.process_manager import process_manager
from llama_suite.webui.utils.ws_manager import manager as ws_manager


router = APIRouter(prefix="/api/watcher", tags=["watcher"])


def get_project_root() -> Path:
    """Get the project root directory."""
    return find_project_root()


class WatcherStartRequest(BaseModel):
    """Request to start the llama-swap watcher."""
    override: Optional[str] = None
    verbose: bool = False
    dry_run: bool = False


@router.get("/status")
async def get_watcher_status():
    """Get status of the llama-swap watcher."""
    tasks = process_manager.get_running_tasks()
    watcher_tasks = {k: v for k, v in tasks.items() if v.task_type == "watcher"}
    
    is_running = len(watcher_tasks) > 0
    
    return {
        "running": is_running,
        "tasks": [
            {
                "task_id": t.task_id,
                "description": t.description,
                "started_at": t.started_at.isoformat()
            }
            for t in watcher_tasks.values()
        ]
    }


@router.post("/start")
async def start_watcher(request: WatcherStartRequest):
    """Start the llama-swap watcher."""
    # Check if already running
    tasks = process_manager.get_running_tasks()
    watcher_tasks = {k: v for k, v in tasks.items() if v.task_type == "watcher"}
    if watcher_tasks:
        raise HTTPException(status_code=409, detail="Watcher is already running")
    
    async def run_watcher_task(task_id: str, **kwargs):
        """Execute watcher as a background task."""
        root = get_project_root()
        venv_python = root / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / "python"
        
        if not venv_python.exists():
            venv_python = Path(sys.executable)
        
        base_config = root / "configs" / "config.base.yaml"
        
        # Build command with correct CLI arguments
        cmd = [
            str(venv_python), "-u", "-m", "llama_suite.watchers.llama_swap_watch",
            str(base_config)  # Positional argument for base config
        ]
        
        if kwargs.get("override"):
            override_path = root / "configs" / "overrides" / f"{kwargs['override']}.yaml"
            cmd.extend(["-o", str(override_path)])  # -o for override, not --override-config
        
        if kwargs.get("verbose"):
            cmd.append("-v")
        
        if kwargs.get("dry_run"):
            cmd.append("--dry-run")
        
        async def on_output(line: str):
            await ws_manager.send_log(task_id, line, "info")
        
        async def on_error(line: str):
            await ws_manager.send_log(task_id, line, "error")
        
        await ws_manager.send_progress(task_id, 0, "Starting llama-swap watcher...")
        
        # The watcher runs indefinitely, so returncode will only come when stopped
        returncode = await process_manager.run_subprocess(
            task_id, cmd, cwd=root,
            on_stdout=on_output, on_stderr=on_error
        )
        
        success = returncode == 0
        await ws_manager.send_complete(task_id, success, {"returncode": returncode})
        
        return {"returncode": returncode, "success": success}
    
    task_id = await process_manager.start_task(
        task_type="watcher",
        description="llama-swap watcher",
        coro=run_watcher_task,
        **request.model_dump()
    )
    
    return {"task_id": task_id, "status": "started"}


@router.post("/stop")
async def stop_watcher():
    """Stop the llama-swap watcher."""
    tasks = process_manager.get_running_tasks()
    watcher_tasks = {k: v for k, v in tasks.items() if v.task_type == "watcher"}
    
    if not watcher_tasks:
        raise HTTPException(status_code=404, detail="No watcher is running")
    
    stopped = []
    for task_id in watcher_tasks:
        cancelled = await process_manager.cancel_task(task_id)
        if cancelled:
            stopped.append(task_id)
    
    return {"status": "stopped", "stopped_tasks": stopped}


@router.get("/logs")
async def get_watcher_logs(limit: int = 100):
    """Get recent logs from the watcher."""
    tasks = process_manager.get_all_tasks()
    watcher_tasks = {k: v for k, v in tasks.items() if v.task_type == "watcher"}
    
    if not watcher_tasks:
        return {"logs": [], "message": "No watcher tasks found"}
    
    # Get most recent watcher task
    sorted_tasks = sorted(watcher_tasks.values(), key=lambda t: t.started_at, reverse=True)
    task = sorted_tasks[0]
    
    return {
        "task_id": task.task_id,
        "status": task.status,
        "logs": task.logs[-limit:]
    }
