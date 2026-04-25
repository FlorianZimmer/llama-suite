from __future__ import annotations

from pathlib import Path

from llama_suite.utils.runtime_registry import (
    all_runtime_server_candidates,
    normalize_runtime_name,
    runtime_default_bin_hint,
)


def test_runtime_aliases_normalize() -> None:
    assert normalize_runtime_name("llama_cpp") == "llama.cpp"
    assert normalize_runtime_name("dflash") == "llama.cpp-dflash"
    assert normalize_runtime_name("ik") == "ik_llama.cpp"
    assert normalize_runtime_name("ik-llama-cpp") == "ik_llama.cpp"


def test_runtime_default_bin_hint_uses_vendor_layout() -> None:
    assert runtime_default_bin_hint("llama.cpp") == "vendor/llama.cpp/bin/llama-server"
    assert runtime_default_bin_hint("llama.cpp-dflash") == "vendor/llama.cpp-dflash/bin/llama-server"
    assert runtime_default_bin_hint("ik_llama.cpp") == "vendor/ik_llama.cpp/bin/llama-server"


def test_all_runtime_server_candidates_include_ik_runtime() -> None:
    repo_root = Path("/repo")
    candidates = [str(p).replace("\\", "/") for p in all_runtime_server_candidates(repo_root, "llama-server")]

    assert "/repo/vendor/llama.cpp/bin/llama-server" in candidates
    assert "/repo/vendor/llama.cpp-dflash/bin/llama-server" in candidates
    assert "/repo/vendor/ik_llama.cpp/bin/llama-server" in candidates
