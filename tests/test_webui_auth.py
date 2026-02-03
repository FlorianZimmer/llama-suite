from __future__ import annotations

from fastapi.testclient import TestClient


def test_optional_api_key_auth_flow(monkeypatch):
    monkeypatch.setenv("LLAMA_SUITE_API_KEY", "secret")
    monkeypatch.delenv("LLAMA_SUITE_MODE", raising=False)

    from llama_suite.webui.server import app

    client = TestClient(app)

    # Public status endpoint for SPA boot
    r = client.get("/api/auth/status")
    assert r.status_code == 200
    assert r.json()["enabled"] is True
    assert r.json()["authenticated"] is False

    # Protected endpoint without auth
    r = client.get("/api/system/links")
    assert r.status_code == 401

    # Login sets a session cookie
    r = client.post("/api/auth/login", json={"api_key": "secret"})
    assert r.status_code == 200

    # Now should be authenticated
    r = client.get("/api/auth/status")
    assert r.status_code == 200
    assert r.json()["authenticated"] is True

    # Protected endpoint now works
    r = client.get("/api/system/links")
    assert r.status_code == 200


def test_login_rejects_bad_key(monkeypatch):
    monkeypatch.setenv("LLAMA_SUITE_API_KEY", "secret")

    from llama_suite.webui.server import app

    client = TestClient(app)
    r = client.post("/api/auth/login", json={"api_key": "wrong"})
    assert r.status_code == 401

