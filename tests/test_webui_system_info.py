from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def test_system_info_reports_ik_runtime(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLAMA_SUITE_ROOT", str(tmp_path))
    (tmp_path / "vendor" / "ik_llama.cpp").mkdir(parents=True)
    (tmp_path / "configs").mkdir()

    from llama_suite.webui.server import app

    client = TestClient(app)
    resp = client.get("/api/system/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ik_llama_cpp_installed"] is True
