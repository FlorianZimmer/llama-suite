"""Evaluation API routes for running model evaluations."""

from pathlib import Path
from typing import Optional, List
import sys
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llama_suite.utils.config_utils import find_project_root
from llama_suite.webui.utils.process_manager import process_manager
from llama_suite.webui.utils.ws_manager import manager as ws_manager
from llama_suite.webui.utils.task_output import handle_task_output



router = APIRouter(prefix="/api/eval", tags=["evaluation"])


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

_TQDM_PCT_RE = re.compile(r"(\d+)%\|")
_TQDM_COUNT_RE = re.compile(r"\|\s*(\d+)/(\d+)")


async def _maybe_update_tqdm_progress(task_id: str, clean_line: str) -> bool:
    m = _TQDM_PCT_RE.search(clean_line)
    if not m:
        return False

    try:
        pct = int(m.group(1))
    except Exception:
        return False

    msg = "Running evaluation..."
    if "REQUESTING API" in clean_line.upper():
        msg = "Requesting API..."

    m3 = _TQDM_COUNT_RE.search(clean_line)
    if m3:
        msg = f"Evaluating samples ({m3.group(1)}/{m3.group(2)})..."

    await ws_manager.send_progress(task_id, pct, msg)
    return True

def get_project_root() -> Path:
    """Get the project root directory."""
    return find_project_root()


class EvalHarnessRequest(BaseModel):
    """Request to start an eval-harness evaluation."""
    override: Optional[str] = None
    model: Optional[str] = None
    tasks: str = "humaneval"
    limit: Optional[float] = None
    num_fewshot: Optional[int] = None
    batch_size: str = "1"
    health_timeout: int = 3600


class CustomEvalRequest(BaseModel):
    """Request to start a custom evaluation."""
    override: Optional[str] = None
    model: Optional[str] = None
    dataset: str = "tasks.jsonl"
    max_tasks: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@router.get("/status")
async def get_eval_status():
    """Get status of running evaluation tasks."""
    tasks = process_manager.get_running_tasks()
    eval_tasks = {k: v for k, v in tasks.items() if v.task_type in ("eval-harness", "custom-eval")}
    
    return {
        "running": len(eval_tasks) > 0,
        "tasks": [
            {
                "task_id": t.task_id,
                "task_type": t.task_type,
                "description": t.description,
                "progress": t.progress,
                "started_at": t.started_at.isoformat()
            }
            for t in eval_tasks.values()
        ]
    }


@router.post("/harness/run")
async def start_eval_harness(request: EvalHarnessRequest):
    """Start an eval-harness evaluation run."""
    
    async def run_eval_task(task_id: str, **kwargs):
        """Execute eval-harness as a background task."""
        root = get_project_root()
        venv_python = root / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / "python"
        
        if not venv_python.exists():
            venv_python = Path(sys.executable)
        
        cmd = [
            str(venv_python), "-u", "-m", "llama_suite.eval.evaluate_models_evalharness",
            # Fix: expect --config instead of --base-config
            "--config", str(root / "configs" / "config.base.yaml"),
            "--tasks", kwargs["tasks"],
            "--batch-size", kwargs["batch_size"],
            "--health-timeout", str(kwargs["health_timeout"]),
            "--plain",
        ]
        
        if kwargs.get("override"):
            override_path = root / "configs" / "overrides" / f"{kwargs['override']}.yaml"
            cmd.extend(["--override", str(override_path)])
        
        if kwargs.get("model"):
            cmd.extend(["--model", kwargs["model"]])
        
        if kwargs.get("limit"):
            cmd.extend(["--limit", str(kwargs["limit"])])
        
        if kwargs.get("num_fewshot") is not None:
            cmd.extend(["--num-fewshot", str(kwargs["num_fewshot"])])
        
        async def on_output(line: str):
            clean_line = strip_ansi(line)
            if await _maybe_update_tqdm_progress(task_id, clean_line):
                return
            await handle_task_output(ws_manager, task_id, clean_line, is_stderr=False, progress_style="steps")
        
        async def on_error(line: str):
            clean_line = strip_ansi(line)
            if await _maybe_update_tqdm_progress(task_id, clean_line):
                return
            await handle_task_output(ws_manager, task_id, clean_line, is_stderr=True, progress_style="steps")
        
        await ws_manager.send_progress(task_id, 0, "Starting eval-harness...")
        
        returncode = await process_manager.run_subprocess(
            task_id, cmd, cwd=root,
            on_stdout=on_output, on_stderr=on_error
        )
        
        success = returncode == 0
        await ws_manager.send_complete(task_id, success, {"returncode": returncode})
        
        return {"returncode": returncode, "success": success}
    
    task_id = await process_manager.start_task(
        task_type="eval-harness",
        description=f"Eval-harness: {request.tasks}" + (f" for {request.model}" if request.model else ""),
        coro=run_eval_task,
        override=request.override,
        model=request.model,
        tasks=request.tasks,
        limit=request.limit,
        num_fewshot=request.num_fewshot,
        batch_size=request.batch_size,
        health_timeout=request.health_timeout
    )
    
    return {"task_id": task_id, "status": "started"}


@router.post("/custom/run")
async def start_custom_eval(request: CustomEvalRequest):
    """Start a custom evaluation run."""
    
    async def run_custom_eval_task(task_id: str, **kwargs):
        """Execute custom eval as a background task."""
        root = get_project_root()
        venv_python = root / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / "python"
        
        if not venv_python.exists():
            venv_python = Path(sys.executable)
        
        # Resolve dataset path
        dataset = kwargs["dataset"]
        if not Path(dataset).is_absolute():
            dataset = str(root / "datasets" / "custom" / dataset)
        
        cmd = [
            str(venv_python), "-u", "-m", "llama_suite.eval.eval",
            "--swap-config", str(root / "configs" / "config.base.yaml"),
            "--data", dataset,
            "--plain",
        ]
        
        if kwargs.get("override"):
            # eval.py does not support --override-config yet. 
            # We log a warning or just ignore it for now to prevent crash.
            pass
            # override_path = root / "configs" / "overrides" / f"{kwargs['override']}.yaml"
            # cmd.extend(["--override-config", str(override_path)])
        
        if kwargs.get("model"):
            cmd.extend(["--model", kwargs["model"]])
        
        if kwargs.get("max_tasks"):
            cmd.extend(["--max-tasks", str(kwargs["max_tasks"])])
        
        if kwargs.get("temperature") is not None:
            cmd.extend(["--temperature", str(kwargs["temperature"])])
        
        if kwargs.get("max_tokens"):
            cmd.extend(["--max-tokens", str(kwargs["max_tokens"])])
        
        async def on_output(line: str):
            clean_line = strip_ansi(line)
            if await _maybe_update_tqdm_progress(task_id, clean_line):
                return
            await handle_task_output(ws_manager, task_id, clean_line, is_stderr=False, progress_style="steps")
        
        async def on_error(line: str):
            clean_line = strip_ansi(line)
            if await _maybe_update_tqdm_progress(task_id, clean_line):
                return
            await handle_task_output(ws_manager, task_id, clean_line, is_stderr=True, progress_style="steps")
        
        await ws_manager.send_progress(task_id, 0, "Starting custom eval...")
        
        returncode = await process_manager.run_subprocess(
            task_id, cmd, cwd=root,
            on_stdout=on_output, on_stderr=on_error
        )
        
        success = returncode == 0
        await ws_manager.send_complete(task_id, success, {"returncode": returncode})
        
        return {"returncode": returncode, "success": success}
    
    task_id = await process_manager.start_task(
        task_type="custom-eval",
        description=f"Custom eval: {request.dataset}" + (f" for {request.model}" if request.model else ""),
        coro=run_custom_eval_task,
        **request.model_dump()
    )
    
    return {"task_id": task_id, "status": "started"}


@router.get("/datasets")
async def list_datasets():
    """List available custom evaluation datasets."""
    datasets_dir = get_project_root() / "datasets" / "custom"
    if not datasets_dir.exists():
        return {"datasets": []}
    
    datasets = []
    for path in sorted(datasets_dir.glob("*.jsonl")):
        # Count lines
        try:
            with open(path, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
        except:
            line_count = None
        
        datasets.append({
            "name": path.name,
            "path": str(path),
            "task_count": line_count
        })
    
    return {"datasets": datasets}


@router.post("/cancel/{task_id}")
async def cancel_eval(task_id: str):
    """Cancel a running evaluation."""
    cancelled = await process_manager.cancel_task(task_id)
    if cancelled:
        return {"status": "cancelled", "task_id": task_id}
    else:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found or already completed")
