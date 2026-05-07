from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_update_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "tools" / "scripts" / "update.py"
    spec = importlib.util.spec_from_file_location("llama_suite_update_script", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_auto_build_backend_uses_cuda_when_windows_cuda_is_available(monkeypatch):
    update = _load_update_module()

    monkeypatch.setattr(update.platform, "system", lambda: "Windows")
    monkeypatch.setenv("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")
    monkeypatch.setattr(update.shutil, "which", lambda name: None)

    assert update.effective_gpu_backend_for_build("auto") == "cuda"

