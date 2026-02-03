from __future__ import annotations

from fastapi.testclient import TestClient


def test_system_links_reflect_env(monkeypatch):
    monkeypatch.setenv("LLAMA_SUITE_MODE", "gitops")
    monkeypatch.setenv("LLAMA_SUITE_SWAP_API_URL", "http://example.invalid:8080/v1")
    monkeypatch.setenv("LLAMA_SUITE_SWAP_UI_URL", "http://example.invalid:8080/ui")
    monkeypatch.setenv("LLAMA_SUITE_OPEN_WEBUI_URL", "http://example.invalid:3000")

    from llama_suite.webui.server import app

    client = TestClient(app)
    r = client.get("/api/system/links")
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "gitops"
    assert data["swap_api_url"] == "http://example.invalid:8080/v1"
    assert data["swap_ui_url"] == "http://example.invalid:8080/ui"
    assert data["open_webui_url"] == "http://example.invalid:3000"


def test_gitops_blocks_mutation_and_subprocess_endpoints(monkeypatch):
    monkeypatch.setenv("LLAMA_SUITE_MODE", "gitops")

    from llama_suite.webui.server import app

    client = TestClient(app)

    # Config mutation
    r = client.put("/api/config", json={"content": "a: 1\n"})
    assert r.status_code == 403

    # Model mutation
    r = client.post("/api/models", params={"name": "m1"}, json={"model_path": "./models/m1.gguf"})
    assert r.status_code == 403

    # Subprocess tasks (watcher/system)
    r = client.post("/api/watcher/start", json={})
    assert r.status_code == 403

    r = client.post("/api/system/update", json={})
    assert r.status_code == 403

    # Results deletion
    r = client.delete("/api/results/bench/some-run")
    assert r.status_code == 403

