from __future__ import annotations

from pathlib import Path
import yaml
from fastapi.testclient import TestClient


def test_create_model_sets_required_cmd_defaults(monkeypatch, tmp_path: Path):
    # Ensure we run in local mode so mutation endpoint is enabled.
    monkeypatch.setenv("LLAMA_SUITE_MODE", "local")

    # Isolate config mutations to a temp file (do not touch repo configs/config.base.yaml).
    base_config_path = tmp_path / "config.base.yaml"
    base_config_path.write_text("models: {}\n", encoding="utf-8")
    monkeypatch.setenv("LLAMA_SUITE_BASE_CONFIG_PATH", str(base_config_path))

    from llama_suite.webui.server import app

    client = TestClient(app)

    name = "Test-Model-Create-Defaults"
    # Ensure it doesn't already exist (idempotent test runs).
    client.delete(f"/api/models/{name}")

    r = client.post(f"/api/models?name={name}", json={"model_path": "./models/foo.gguf", "ctx_size": 8192, "gpu_layers": -1})
    assert r.status_code == 200, r.text

    # Validate the resulting YAML contains required keys for the new model.
    content = client.get("/api/config").json()["content"]
    cfg = yaml.safe_load(content)
    model_cfg = cfg["models"][name]
    cmd = model_cfg["cmd"]
    assert cmd.get("bin")
    assert cmd.get("port")
    assert cmd.get("model")
    assert cmd.get("ctx-size")
