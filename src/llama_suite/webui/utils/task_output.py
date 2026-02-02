"""Helpers for turning subprocess output into clean Web UI logs/progress."""

from __future__ import annotations

import re
from typing import Any, Protocol


class _WSManager(Protocol):
    async def send_progress(self, task_id: str, progress: float, message: str, status: str = "running") -> Any: ...
    async def send_log(self, task_id: str, line: str, level: str = "info") -> Any: ...


_STEP_RE = re.compile(r"^STEP\s+(?P<i>\d+)\s*/\s*(?P<n>\d+)\s*:\s*(?P<msg>.+)\s*$")


def classify_log_line(line: str, *, is_stderr: bool) -> str:
    """
    Map a raw subprocess output line to a UI log level.

    We do NOT treat all stderr output as an error because many tools (and Python logging's
    default StreamHandler) write normal info/warn output to stderr.
    """
    stripped = line.strip()
    if not stripped:
        return "info"

    upper = stripped.upper()

    # Common patterns (both "ERROR:" and "[ERROR]" styles)
    if upper.startswith("ERROR:") or upper.startswith("[ERROR]") or upper.startswith("FATAL:"):
        return "error"
    if "TRACEBACK (MOST RECENT CALL LAST)" in upper:
        return "error"
    if upper.startswith("WARN:") or upper.startswith("WARNING:") or upper.startswith("[WARN]") or upper.startswith("[WARNING]"):
        return "warning"

    # Default: stderr becomes warning (less alarming), stdout becomes info.
    return "warning" if is_stderr else "info"


async def handle_task_output(
    ws: _WSManager,
    task_id: str,
    line: str,
    *,
    is_stderr: bool,
    progress_style: str = "steps",
) -> None:
    """
    Process a single output line from a task.

    progress_style:
      - "steps": parse STEP i/n lines into a percentage (completed-steps style).
      - "indeterminate": keep progress bar indeterminate (-1) but update message.
      - "none": don't send progress updates from STEP lines.
    """
    stripped = line.strip()
    m = _STEP_RE.match(stripped)
    if m:
        i = int(m.group("i"))
        n = int(m.group("n"))
        msg = m.group("msg").strip()
        if n > 0 and i >= 1:
            if progress_style == "steps":
                pct = max(0.0, min(100.0, ((i - 1) / n) * 100.0))
                await ws.send_progress(task_id, pct, f"{msg} ({i}/{n})")
            elif progress_style == "indeterminate":
                await ws.send_progress(task_id, -1, f"{msg} ({i}/{n})")
            elif progress_style == "none":
                pass

    level = classify_log_line(line, is_stderr=is_stderr)
    await ws.send_log(task_id, line, level)

