"""Benchmark API routes for running model benchmarks."""

from pathlib import Path
from typing import Optional
import sys

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llama_suite.utils.config_utils import find_project_root
from llama_suite.webui.utils.process_manager import process_manager
from llama_suite.webui.utils.ws_manager import manager as ws_manager
from llama_suite.webui.utils.task_output import handle_task_output


router = APIRouter(prefix="/api/bench", tags=["benchmark"])


def get_project_root() -> Path:
    """Get the project root directory."""
    return find_project_root()


class BenchmarkRequest(BaseModel):
    """Request to start a benchmark run."""
    override: Optional[str] = None
    model: Optional[str] = None
    question: str = "What is the capital of France?"
    health_timeout: int = 120


@router.get("/status")
async def get_bench_status():
    """Get status of running benchmark tasks."""
    tasks = process_manager.get_running_tasks()
    bench_tasks = {k: v for k, v in tasks.items() if v.task_type == "benchmark"}
    
    return {
        "running": len(bench_tasks) > 0,
        "tasks": [
            {
                "task_id": t.task_id,
                "description": t.description,
                "progress": t.progress,
                "started_at": t.started_at.isoformat()
            }
            for t in bench_tasks.values()
        ]
    }


@router.post("/run")
async def start_benchmark(request: BenchmarkRequest):
    """Start a new benchmark run."""
    
    async def run_benchmark_task(task_id: str, override: Optional[str], model: Optional[str], 
                                  question: str, health_timeout: int):
        """Execute the benchmark as a background task."""
        root = get_project_root()
        venv_python = root / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / "python"
        
        if not venv_python.exists():
            venv_python = Path(sys.executable)
        
        bench_script = root / "src" / "llama_suite" / "bench" / "benchmark-models.py"
        cmd = [
            str(venv_python), "-u", str(bench_script),
            "--config", str(root / "configs" / "config.base.yaml"),
            "--question", question,
            "--health-timeout", str(health_timeout),
            "--plain",
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
        
        await ws_manager.send_progress(task_id, 0, "Starting benchmark...")
        
        returncode = await process_manager.run_subprocess(
            task_id, cmd, cwd=root,
            on_stdout=on_output, on_stderr=on_error
        )
        
        success = returncode == 0
        await ws_manager.send_complete(task_id, success, {"returncode": returncode})
        
        return {"returncode": returncode, "success": success}
    
    task_id = await process_manager.start_task(
        task_type="benchmark",
        description=f"Benchmark run" + (f" for {request.model}" if request.model else ""),
        coro=run_benchmark_task,
        override=request.override,
        model=request.model,
        question=request.question,
        health_timeout=request.health_timeout
    )
    
    return {"task_id": task_id, "status": "started"}


@router.post("/cancel/{task_id}")
async def cancel_benchmark(task_id: str):
    """Cancel a running benchmark."""
    cancelled = await process_manager.cancel_task(task_id)
    if cancelled:
        return {"status": "cancelled", "task_id": task_id}
    else:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found or already completed")


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a specific task."""
    task = process_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "description": task.description,
        "status": task.status,
        "progress": task.progress,
        "started_at": task.started_at.isoformat(),
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "result": task.result,
        "error": task.error,
        "log_count": len(task.logs)
    }


@router.get("/task/{task_id}/logs")
async def get_task_logs(task_id: str, offset: int = 0, limit: int = 100):
    """Get logs for a specific task."""
    task = process_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    
    logs = task.logs[offset:offset + limit]
    return {
        "task_id": task_id,
        "total": len(task.logs),
        "offset": offset,
        "logs": logs
    }
