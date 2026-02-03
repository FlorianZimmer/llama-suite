from __future__ import annotations

import asyncio


def test_handle_task_output_parses_step_lines_into_progress() -> None:
    from llama_suite.webui.utils.task_output import handle_task_output

    class FakeWS:
        def __init__(self) -> None:
            self.progress: list[tuple[float, str, str]] = []
            self.logs: list[tuple[str, str]] = []

        async def send_progress(self, task_id: str, progress: float, message: str, status: str = "running"):
            self.progress.append((progress, message, status))

        async def send_log(self, task_id: str, line: str, level: str = "info"):
            self.logs.append((line, level))

    ws = FakeWS()
    asyncio.run(handle_task_output(ws, "T", "STEP 2/4: Memory scan: Foo", is_stderr=False, progress_style="steps"))

    assert ws.progress == [(25.0, "Memory scan: Foo (2/4)", "running")]
    assert ws.logs == [("STEP 2/4: Memory scan: Foo", "info")]

