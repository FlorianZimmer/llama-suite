from __future__ import annotations

from llama_suite.proxy.opencode import ProxyConfig, normalize_upstream_base_url, prepare_chat_payload, upstream_url


def _cfg(slots: int = 1) -> ProxyConfig:
    return ProxyConfig(
        upstream_base_url="http://127.0.0.1:8080/v1/",
        slots=slots,
        default_slot=0,
        cache_reuse=256,
        force_cache_prompt=True,
        stream_timeout_s=60,
        request_timeout_s=60,
    )


def test_prepare_chat_payload_injects_llama_cpp_cache_controls() -> None:
    payload = {
        "model": "Qwen3.6-35B-A3B-UD-Q4_K_XL",
        "messages": [
            {"role": "system", "content": "stable prefix"},
            {"role": "user", "content": "request"},
        ],
    }

    out, slot, cache_key = prepare_chat_payload(payload, _cfg())

    assert slot == 0
    assert cache_key
    assert out["cache_prompt"] is True
    assert out["n_cache_reuse"] == 256
    assert out["id_slot"] == 0
    assert out["prompt_cache_key"] == cache_key


def test_prepare_chat_payload_strips_llamasuite_provider_prefix() -> None:
    payload = {
        "model": "llamasuite/Qwen3.6-35B-A3B-UD-Q4_K_XL",
        "messages": [{"role": "user", "content": "hello"}],
    }

    out, _, _ = prepare_chat_payload(payload, _cfg())

    assert out["model"] == "Qwen3.6-35B-A3B-UD-Q4_K_XL"


def test_prepare_chat_payload_applies_qwen_general_sampling_preset() -> None:
    payload = {
        "model": "llamasuite/Qwen3.6-27B-MLX-NVFP4-MTP-General",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }

    out, _, _ = prepare_chat_payload(payload, _cfg())

    assert out["model"] == "Qwen3.6-27B-MLX-NVFP4-MTP-General"
    assert out["temperature"] == 1.0
    assert out["top_p"] == 0.95
    assert out["top_k"] == 20
    assert out["min_p"] == 0.0
    assert out["presence_penalty"] == 0.0


def test_prepare_chat_payload_applies_qwen_coding_sampling_preset() -> None:
    payload = {
        "model": "Qwen3.6-27B-UD-Q5_K_XL-Coding",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 1,
    }

    out, _, _ = prepare_chat_payload(payload, _cfg())

    assert out["temperature"] == 0.6
    assert out["top_p"] == 0.95
    assert out["top_k"] == 20
    assert out["min_p"] == 0.0
    assert out["presence_penalty"] == 0.0


def test_prepare_chat_payload_preserves_explicit_cache_key() -> None:
    payload = {
        "model": "m",
        "prompt_cache_key": "opencode-prefix-123",
        "messages": [{"role": "user", "content": "hello"}],
    }

    out, _, cache_key = prepare_chat_payload(payload, _cfg(slots=4))

    assert cache_key == "opencode-prefix-123"
    assert out["prompt_cache_key"] == "opencode-prefix-123"


def test_prepare_chat_payload_uses_stable_prefix_not_latest_user_message() -> None:
    base = [
        {"role": "system", "content": "large stable prefix"},
        {"role": "assistant", "content": "also stable"},
    ]
    first = {"model": "m", "messages": [*base, {"role": "user", "content": "first"}]}
    second = {"model": "m", "messages": [*base, {"role": "user", "content": "second"}]}

    _, first_slot, first_key = prepare_chat_payload(first, _cfg(slots=8))
    _, second_slot, second_key = prepare_chat_payload(second, _cfg(slots=8))

    assert first_key == second_key
    assert first_slot == second_slot


def test_normalize_upstream_base_url_adds_v1_when_missing() -> None:
    assert normalize_upstream_base_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080/v1/"
    assert normalize_upstream_base_url("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080/v1/"


def test_upstream_url_preserves_v1_for_openai_routes() -> None:
    assert upstream_url(_cfg(), "/v1/completions") == "http://127.0.0.1:8080/v1/completions"
