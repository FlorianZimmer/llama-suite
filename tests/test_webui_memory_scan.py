from __future__ import annotations

import re
from pathlib import Path


def test_webui_has_memory_scan_api_route() -> None:
    from llama_suite.webui.server import app

    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/api/memory/run" in paths


def test_webui_memory_scan_calls_memory_endpoint() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    app_js = (repo_root / "src" / "llama_suite" / "webui" / "static" / "js" / "app.js").read_text(encoding="utf-8")

    m = re.search(r"async\s+runMemoryScan\(\)\s*\{[\s\S]*?API\.post\('([^']+)'", app_js)
    assert m, "Could not find API.post call in runMemoryScan()"
    assert m.group(1) == "/api/memory/run"

