"""System API routes for install, update, uninstall, and model download."""

from pathlib import Path
from typing import Optional, List
import sys
import platform

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llama_suite.utils.config_utils import find_project_root
from llama_suite.webui.utils.process_manager import process_manager
from llama_suite.webui.utils.ws_manager import manager as ws_manager
from llama_suite.webui.utils.task_output import handle_task_output


router = APIRouter(prefix="/api/system", tags=["system"])


def get_project_root() -> Path:
    """Get the project root directory."""
    return find_project_root()


@router.get("/info")
async def get_system_info():
    """Get system and project information."""
    root = get_project_root()
    venv_dir = root / ".venv"
    vendor_dir = root / "vendor"
    
    # Check what's installed
    llama_swap_exists = (vendor_dir / "llama-swap").exists() or \
                        (vendor_dir / "llama-swap.exe").exists()
    llama_cpp_exists = (vendor_dir / "llama.cpp").exists() or \
                       (root / "llama.cpp").exists()
    
    return {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "project_root": str(root),
        "venv_exists": venv_dir.exists(),
        "venv_path": str(venv_dir),
        "llama_swap_installed": llama_swap_exists,
        "llama_cpp_installed": llama_cpp_exists,
        "vendor_path": str(vendor_dir)
    }


class UpdateRequest(BaseModel):
    """Request to run update."""
    update_python: bool = True
    update_llama_swap: bool = True
    update_llama_cpp: bool = True
    update_open_webui: bool = True
    gpu_backend: str = "auto"


@router.post("/update")
async def run_update(request: UpdateRequest):
    """Run the update script."""
    
    async def run_update_task(task_id: str, **kwargs):
        """Execute update as a background task."""
        root = get_project_root()
        venv_python = root / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / "python"
        
        if not venv_python.exists():
            venv_python = Path(sys.executable)
        
        cmd = [str(venv_python), str(root / "tools" / "scripts" / "update.py")]
        
        if not kwargs.get("update_python"):
            cmd.extend(["--skip", "venv"])
        if not kwargs.get("update_llama_swap"):
            cmd.extend(["--skip", "swap"])
        if not kwargs.get("update_llama_cpp"):
            cmd.extend(["--skip", "cpp"])
        if not kwargs.get("update_open_webui"):
            cmd.extend(["--skip", "webui"])
        if kwargs.get("gpu_backend") and kwargs["gpu_backend"] != "auto":
            cmd.extend(["--gpu-backend", kwargs["gpu_backend"]])
        
        async def on_output(line: str):
            await handle_task_output(ws_manager, task_id, line, is_stderr=False, progress_style="steps")
        
        async def on_error(line: str):
            await handle_task_output(ws_manager, task_id, line, is_stderr=True, progress_style="steps")
        
        await ws_manager.send_progress(task_id, 0, "Starting update...")
        
        returncode = await process_manager.run_subprocess(
            task_id, cmd, cwd=root,
            on_stdout=on_output, on_stderr=on_error
        )
        
        success = returncode == 0
        await ws_manager.send_complete(task_id, success, {"returncode": returncode})
        
        return {"returncode": returncode, "success": success}
    
    task_id = await process_manager.start_task(
        task_type="update",
        description="System update",
        coro=run_update_task,
        **request.model_dump()
    )
    
    return {"task_id": task_id, "status": "started"}


class InstallRequest(BaseModel):
    """Request to run install."""
    gpu_backend: str = "auto"


@router.post("/install")
async def run_install(request: InstallRequest):
    """Run the install script."""
    
    async def run_install_task(task_id: str, **kwargs):
        """Execute install as a background task."""
        root = get_project_root()
        # Use system python for initial install
        python_exe = sys.executable
        
        cmd = [python_exe, str(root / "tools" / "scripts" / "install.py")]
        
        if kwargs.get("gpu_backend") and kwargs["gpu_backend"] != "auto":
            cmd.extend(["--gpu-backend", kwargs["gpu_backend"]])
        
        async def on_output(line: str):
            await handle_task_output(ws_manager, task_id, line, is_stderr=False, progress_style="steps")
        
        async def on_error(line: str):
            await handle_task_output(ws_manager, task_id, line, is_stderr=True, progress_style="steps")
        
        await ws_manager.send_progress(task_id, 0, "Starting install...")
        
        returncode = await process_manager.run_subprocess(
            task_id, cmd, cwd=root,
            on_stdout=on_output, on_stderr=on_error
        )
        
        success = returncode == 0
        await ws_manager.send_complete(task_id, success, {"returncode": returncode})
        
        return {"returncode": returncode, "success": success}
    
    task_id = await process_manager.start_task(
        task_type="install",
        description="System install",
        coro=run_install_task,
        **request.model_dump()
    )
    
    return {"task_id": task_id, "status": "started"}


class DownloadRequest(BaseModel):
    """Request to download models."""
    override: Optional[str] = None
    include_drafts: bool = False
    include_tokenizers: bool = True
    force: bool = False
    models: Optional[List[str]] = None  # Specific models to download


@router.post("/download")
async def download_models(request: DownloadRequest):
    """Download models using hf_fetch."""
    
    async def run_download_task(task_id: str, **kwargs):
        """Execute hf_fetch as a background task."""
        root = get_project_root()
        venv_python = root / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / "python"
        
        if not venv_python.exists():
            venv_python = Path(sys.executable)
        
        cmd = [
            str(venv_python), str(root / "tools" / "scripts" / "hf_fetch.py"),
            "--base", str(root / "configs" / "config.base.yaml"),
            "--target", str(root / "models")
        ]
        
        if kwargs.get("override"):
            override_path = root / "configs" / "overrides" / f"{kwargs['override']}.yaml"
            cmd.extend(["--override", str(override_path)])
        
        if kwargs.get("include_drafts"):
            cmd.append("--include-drafts")
        
        if kwargs.get("include_tokenizers"):
            cmd.append("--also-tokenizers")
            cmd.extend(["--tokenizers-dir", str(root / "models" / "tokenizers")])
        
        if kwargs.get("force"):
            cmd.append("--force")

        # Prefer plain output in the Web UI (no rich progress bars/markup).
        cmd.append("--plain")

        async def on_output(line: str):
            await handle_task_output(ws_manager, task_id, line, is_stderr=False, progress_style="steps")
        
        async def on_error(line: str):
            await handle_task_output(ws_manager, task_id, line, is_stderr=True, progress_style="steps")
        
        await ws_manager.send_progress(task_id, 0, "Starting model download...")
        
        returncode = await process_manager.run_subprocess(
            task_id, cmd, cwd=root,
            on_stdout=on_output, on_stderr=on_error
        )
        
        success = returncode == 0
        await ws_manager.send_complete(task_id, success, {"returncode": returncode})
        
        return {"returncode": returncode, "success": success}
    
    task_id = await process_manager.start_task(
        task_type="download",
        description="Model download",
        coro=run_download_task,
        **request.model_dump()
    )
    
    return {"task_id": task_id, "status": "started"}


@router.get("/tasks")
async def get_all_tasks():
    """Get all system tasks (running and completed)."""
    tasks = process_manager.get_all_tasks()
    
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "task_type": t.task_type,
                "description": t.description,
                "status": t.status,
                "progress": t.progress,
                "started_at": t.started_at.isoformat(),
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                "error": t.error
            }
            for t in sorted(tasks.values(), key=lambda x: x.started_at, reverse=True)
        ]
    }


@router.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running task."""
    cancelled = await process_manager.cancel_task(task_id)
    if cancelled:
        return {"status": "cancelled", "task_id": task_id}
    else:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found or already completed")
