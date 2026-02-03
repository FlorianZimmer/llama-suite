from __future__ import annotations

from pathlib import Path


def test_webui_has_stop_buttons_and_progress_elements() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    index_html = (repo_root / "src" / "llama_suite" / "webui" / "static" / "index.html").read_text(encoding="utf-8")

    required_ids = [
        "btn-bench-start",
        "btn-bench-stop",
        "btn-memory-start",
        "btn-memory-stop",
        "memory-progress-text",
        "memory-progress-fill",
        "btn-eval-harness-start",
        "btn-eval-harness-stop",
        "eval-harness-progress-text",
        "eval-harness-progress-fill",
        "btn-eval-custom-start",
        "btn-eval-custom-stop",
        "eval-custom-progress-text",
        "eval-custom-progress-fill",
    ]

    for element_id in required_ids:
        assert f'id="{element_id}"' in index_html


def test_webui_has_all_models_confirm_guardrails() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    app_js = (repo_root / "src" / "llama_suite" / "webui" / "static" / "js" / "app.js").read_text(encoding="utf-8")

    # These are intentionally stable strings; if we change copy we should keep the guardrail behavior.
    required_prompts = [
        "Run benchmark for ALL models?",
        "Run memory scan for ALL models?",
        "Run evaluation harness for ALL models?",
        "Run custom eval for ALL models?",
    ]

    for prompt in required_prompts:
        assert prompt in app_js

