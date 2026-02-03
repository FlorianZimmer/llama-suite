"""Background process manager for long-running operations."""

from typing import Dict, Optional, Callable, Any
import asyncio
import sys
import os
import signal
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TaskInfo:
    """Information about a running or completed task."""
    task_id: str
    task_type: str
    description: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, cancelled
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: list = field(default_factory=list)


class ProcessManager:
    """Manages background processes for long-running operations."""

    def __init__(self):
        self.tasks: Dict[str, TaskInfo] = {}
        self.processes: Dict[str, asyncio.subprocess.Process] = {}
        self._lock = asyncio.Lock()

    def create_task_id(self) -> str:
        """Generate a unique task ID."""
        return str(uuid.uuid4())[:8]

    async def start_task(
        self,
        task_type: str,
        description: str,
        coro: Callable,
        *args,
        **kwargs
    ) -> str:
        """Start a new background task and return its ID."""
        task_id = self.create_task_id()
        
        task_info = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            description=description,
            started_at=datetime.now()
        )
        
        async with self._lock:
            self.tasks[task_id] = task_info
        
        # Run the coroutine in the background
        asyncio.create_task(self._run_task(task_id, coro, *args, **kwargs))
        
        return task_id

    async def _run_task(self, task_id: str, coro: Callable, *args, **kwargs):
        """Execute a task and update its status."""
        try:
            result = await coro(task_id, *args, **kwargs)
            async with self._lock:
                if task_id in self.tasks:
                    self.tasks[task_id].status = "completed"
                    self.tasks[task_id].completed_at = datetime.now()
                    self.tasks[task_id].progress = 100.0
                    self.tasks[task_id].result = result
        except asyncio.CancelledError:
            async with self._lock:
                if task_id in self.tasks:
                    self.tasks[task_id].status = "cancelled"
                    self.tasks[task_id].completed_at = datetime.now()
        except Exception as e:
            async with self._lock:
                if task_id in self.tasks:
                    self.tasks[task_id].status = "failed"
                    self.tasks[task_id].completed_at = datetime.now()
                    self.tasks[task_id].error = str(e)

    async def run_subprocess(
        self,
        task_id: str,
        cmd: list,
        cwd: Optional[Path] = None,
        on_stdout: Optional[Callable[[str], None]] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
        on_start: Optional[Callable[[asyncio.subprocess.Process], Any]] = None,
    ) -> int:
        """Run a subprocess and stream its output."""
        env = os.environ.copy()
        # Make Python subprocess output UTF-8 so the Web UI can display it reliably on Windows.
        # (Without this, many environments default to legacy codepages that can't encode emojis/symbols.)
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        # Prefer predictable, non-ANSI output for Web UI logs.
        env.setdefault("LLAMA_SUITE_PLAIN", "1")
        # NOTE: do not force-disable tqdm; some tasks rely on it for progress, and this manager
        # already splits output on both \n and \r.

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

        if on_start:
            await on_start(process) if asyncio.iscoroutinefunction(on_start) else on_start(process)
        
        async with self._lock:
            self.processes[task_id] = process

        async def read_stream(stream, callback):
            buffer = ""
            while True:
                # Read chunks to handle \r updates (e.g. from tqdm) immediately
                chunk = await stream.read(1024)
                if not chunk:
                    if buffer:
                        # Flush remaining buffer
                        text = buffer.rstrip()
                        if text:
                            if callback:
                                await callback(text) if asyncio.iscoroutinefunction(callback) else callback(text)
                            async with self._lock:
                                if task_id in self.tasks:
                                    self.tasks[task_id].logs.append(text)
                    break
                
                # Decode and process
                decoded = chunk.decode("utf-8", errors="replace")
                buffer += decoded
                
                while True:
                    # Split on either \n or \r
                    # prioritize \n for standard lines
                    newline_pos = buffer.find('\n')
                    cr_pos = buffer.find('\r')
                    
                    pos = -1
                    if newline_pos != -1 and cr_pos != -1:
                        pos = min(newline_pos, cr_pos)
                    elif newline_pos != -1:
                        pos = newline_pos
                    elif cr_pos != -1:
                        pos = cr_pos
                    
                    if pos == -1:
                        break
                        
                    line = buffer[:pos].strip()
                    buffer = buffer[pos+1:]
                    
                    if line:
                        if callback:
                            await callback(line) if asyncio.iscoroutinefunction(callback) else callback(line)
                        async with self._lock:
                            if task_id in self.tasks:
                                self.tasks[task_id].logs.append(line)

        await asyncio.gather(
            read_stream(process.stdout, on_stdout),
            read_stream(process.stderr, on_stderr)
        )
        
        await process.wait()
        
        async with self._lock:
            self.processes.pop(task_id, None)
        
        return process.returncode

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task and terminate its subprocess (including child processes)."""
        process = None
        async with self._lock:
            if task_id in self.processes:
                process = self.processes[task_id]
        
        if process is None:
            # No subprocess found, but maybe we can update task status
            async with self._lock:
                if task_id in self.tasks and self.tasks[task_id].status == "running":
                    self.tasks[task_id].status = "cancelled"
                    self.tasks[task_id].completed_at = datetime.now()
                    return True
            return False
        
        try:
            pid = process.pid
            
            # On Windows, use taskkill to kill entire process tree
            if sys.platform == "win32":
                # taskkill /T = kill process tree, /F = force
                try:
                    kill_proc = await asyncio.create_subprocess_exec(
                        "taskkill", "/T", "/F", "/PID", str(pid),
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                    await asyncio.wait_for(kill_proc.wait(), timeout=5.0)
                except Exception:
                    # Fallback to regular kill
                    process.kill()
            else:
                # On Unix, try to kill process group
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    process.terminate()
            
            # Wait for process to terminate
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                # Force kill if still running
                try:
                    process.kill()
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except Exception:
                    pass
            
            # Update task status
            async with self._lock:
                self.processes.pop(task_id, None)
                if task_id in self.tasks:
                    self.tasks[task_id].status = "cancelled"
                    self.tasks[task_id].completed_at = datetime.now()
            
            return True
        except Exception as e:
            # Log error but still try to clean up
            async with self._lock:
                self.processes.pop(task_id, None)
                if task_id in self.tasks:
                    self.tasks[task_id].status = "failed"
                    self.tasks[task_id].error = f"Cancel error: {e}"
                    self.tasks[task_id].completed_at = datetime.now()
            return False

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task information by ID."""
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, TaskInfo]:
        """Get all tasks."""
        return dict(self.tasks)

    def get_running_tasks(self) -> Dict[str, TaskInfo]:
        """Get all currently running tasks."""
        return {k: v for k, v in self.tasks.items() if v.status == "running"}

    async def update_progress(self, task_id: str, progress: float, message: str = ""):
        """Update task progress."""
        async with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].progress = progress


# Global process manager instance
process_manager = ProcessManager()
