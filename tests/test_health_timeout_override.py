from __future__ import annotations

import sys
from pathlib import Path


def test_scan_model_memory_cli_timeout_overrides_config(monkeypatch, tmp_path: Path) -> None:
    """
    Regression test:
    If the caller explicitly passes --health-timeout (even if equal to the historical default 120),
    it must override config.healthCheckTimeout (often 60).
    """
    from llama_suite.bench import scan_model_memory as mod

    # Isolate any file outputs
    monkeypatch.setattr(mod.util, "LOGS_DIR", tmp_path / "logs", raising=False)
    monkeypatch.setattr(mod.util, "RESULTS_DIR", tmp_path / "results", raising=False)
    monkeypatch.setattr(mod.util, "timestamp_str", lambda: "TESTTS", raising=False)
    monkeypatch.setattr(mod.util, "register_signal_handlers", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(mod.util, "kill_lingering_servers_on_port", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(mod.util, "enforce_retention", lambda *a, **k: None, raising=False)

    # Provide a minimal effective config with a smaller healthCheckTimeout
    monkeypatch.setattr(
        mod.util,
        "load_and_process_config",
        lambda *a, **k: {
            "healthCheckTimeout": 60,
            "models": {"X": {"cmd": {"bin": "llama-server", "port": "${PORT}", "model": "./models/x.gguf", "ctx-size": 128}}},
        },
        raising=False,
    )

    # Avoid touching real filesystem in copy step
    monkeypatch.setattr(mod.shutil, "copy2", lambda *a, **k: None, raising=True)

    captured: dict[str, int] = {}

    def fake_run_memory_scan(*, health_timeout_s: int, **kwargs):
        captured["health_timeout_s"] = health_timeout_s

    monkeypatch.setattr(mod, "run_memory_scan", fake_run_memory_scan, raising=True)

    # Base/override files must exist (script checks)
    base = tmp_path / "config.base.yaml"
    base.write_text("models: {}\n", encoding="utf-8")

    argv = [
        "scan_model_memory.py",
        "--config",
        str(base),
        "--override",
        str(base),
        "--health-timeout",
        "120",
        "--model",
        "X",
    ]
    monkeypatch.setattr(sys, "argv", argv, raising=True)

    mod.main()

    assert captured["health_timeout_s"] == 120

