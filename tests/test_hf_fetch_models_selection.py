from __future__ import annotations

import re
from pathlib import Path


def test_hf_fetch_supports_models_filter() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    hf_fetch = (repo_root / "tools" / "scripts" / "hf_fetch.py").read_text(encoding="utf-8")

    assert "--models" in hf_fetch
    assert re.search(r"selected_models\s*:\s*Optional\[Set\[str\]\]", hf_fetch)
    assert "name not in selected_models" in hf_fetch


def test_system_download_passes_models_to_hf_fetch() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    system_py = (repo_root / "src" / "llama_suite" / "webui" / "api" / "system.py").read_text(encoding="utf-8")
    assert "--models" in system_py

