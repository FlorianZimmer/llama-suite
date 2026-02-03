from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from llama_suite.webui.server import app


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_temp_project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "configs" / "overrides").mkdir(parents=True, exist_ok=True)
    return root


def _write_base_config(root: Path) -> None:
    _write_text(
        root / "configs" / "config.base.yaml",
        "\n".join(
            [
                "startPort: 9000",
                "healthCheckTimeout: 120",
                "logLevel: info",
                "QWEN3_SAMPLING:",
                "  temp: 0.7",
                "models:",
                "  m1:",
                "    cmd:",
                "      bin: llama-server",
                "      port: 9001",
                "      model: models/m1.gguf",
                "      ctx-size: 8192",
                "      jinja: false",
                "  m2:",
                "    cmd:",
                "      bin: llama-server",
                "      port: 9002",
                "      model: models/m2.gguf",
                "      ctx-size: 8192",
                "",
            ]
        ),
    )


def test_config_studio_schema_served(tmp_path: Path, monkeypatch) -> None:
    root = _make_temp_project(tmp_path)
    _write_base_config(root)
    monkeypatch.setenv("LLAMA_SUITE_ROOT", str(root))

    client = TestClient(app)
    resp = client.get("/api/config/studio")
    assert resp.status_code == 200
    data = resp.json()
    assert "schema" in data
    assert "groups" in data["schema"]
    assert any(g.get("id") == "general" for g in (data["schema"].get("groups") or []))
    assert data["meta"]["models"] == ["m1", "m2"]


def test_config_studio_patch_override_delete_resets_to_base(tmp_path: Path, monkeypatch) -> None:
    root = _make_temp_project(tmp_path)
    _write_base_config(root)
    _write_text(
        root / "configs" / "overrides" / "ov1.yaml",
        "\n".join(
            [
                "models:",
                "  m1:",
                "    cmd:",
                "      jinja: true",
                "",
            ]
        ),
    )
    monkeypatch.setenv("LLAMA_SUITE_ROOT", str(root))

    client = TestClient(app)

    before = client.get("/api/config/studio?override=ov1").json()
    assert before["effective"]["models"]["m1"]["cmd"]["jinja"] is True

    resp = client.post(
        "/api/config/studio/patch",
        json={
            "target": {"kind": "override", "name": "ov1"},
            "ops": [{"op": "delete", "path": ["models", "m1", "cmd", "jinja"]}],
            "context_override": "ov1",
        },
    )
    assert resp.status_code == 200

    after = client.get("/api/config/studio?override=ov1").json()
    assert after["effective"]["models"]["m1"]["cmd"]["jinja"] is False


def test_config_studio_bulk_apply_models_subset(tmp_path: Path, monkeypatch) -> None:
    root = _make_temp_project(tmp_path)
    _write_base_config(root)
    monkeypatch.setenv("LLAMA_SUITE_ROOT", str(root))

    client = TestClient(app)
    resp = client.post(
        "/api/config/studio/bulk-apply",
        json={
            "target": {"kind": "override", "name": "ov2"},
            "models": ["m1"],
            "section": "cmd",
            "changes": {"parallel": 2},
            "context_override": "ov2",
        },
    )
    assert resp.status_code == 200

    override_yaml = (root / "configs" / "overrides" / "ov2.yaml").read_text(encoding="utf-8")
    assert "m1:" in override_yaml
    assert "parallel:" in override_yaml
    assert "m2:" not in override_yaml

