from __future__ import annotations

from fastapi.testclient import TestClient

from llama_suite.webui.server import app


def test_effective_config_includes_yaml() -> None:
    client = TestClient(app)
    resp = client.get("/api/config/effective")
    assert resp.status_code == 200
    data = resp.json()
    assert "yaml" in data
    assert isinstance(data["yaml"], str)
    assert "models:" in data["yaml"]


def test_index_is_no_store() -> None:
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers.get("cache-control") == "no-store"

