from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import requests
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse


DEFAULT_UPSTREAM_BASE_URL = "http://127.0.0.1:8080/v1"


@dataclass(frozen=True)
class ProxyConfig:
    upstream_base_url: str
    slots: int
    default_slot: int
    cache_reuse: int
    force_cache_prompt: bool
    stream_timeout_s: float
    request_timeout_s: float


def normalize_upstream_base_url(value: str) -> str:
    base = value.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/"


def make_config() -> ProxyConfig:
    return ProxyConfig(
        upstream_base_url=normalize_upstream_base_url(os.getenv("LLAMA_SUITE_PROXY_UPSTREAM", DEFAULT_UPSTREAM_BASE_URL)),
        slots=max(1, int(os.getenv("LLAMA_SUITE_PROXY_SLOTS", "1"))),
        default_slot=max(0, int(os.getenv("LLAMA_SUITE_PROXY_DEFAULT_SLOT", "0"))),
        cache_reuse=max(0, int(os.getenv("LLAMA_SUITE_PROXY_CACHE_REUSE", "256"))),
        force_cache_prompt=os.getenv("LLAMA_SUITE_PROXY_FORCE_CACHE_PROMPT", "1").lower() not in {"0", "false", "no"},
        stream_timeout_s=float(os.getenv("LLAMA_SUITE_PROXY_STREAM_TIMEOUT", "3600")),
        request_timeout_s=float(os.getenv("LLAMA_SUITE_PROXY_REQUEST_TIMEOUT", "3600")),
    )


def _stable_cache_key(payload: dict[str, Any]) -> str:
    explicit = payload.get("prompt_cache_key") or payload.get("cache_key")
    if isinstance(explicit, str) and explicit:
        return explicit

    model = str(payload.get("model") or "")
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        stable_prefix = messages[:-1] if len(messages) > 1 else messages
    else:
        stable_prefix = payload.get("prompt") or ""

    raw = json.dumps({"model": model, "prefix": stable_prefix}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _slot_for(payload: dict[str, Any], cfg: ProxyConfig) -> int:
    if cfg.slots == 1:
        return min(cfg.default_slot, cfg.slots - 1)
    key = _stable_cache_key(payload)
    digest = hashlib.blake2s(key.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % cfg.slots


def prepare_chat_payload(payload: dict[str, Any], cfg: ProxyConfig) -> tuple[dict[str, Any], int, str]:
    out = dict(payload)
    cache_key = _stable_cache_key(out)
    slot = _slot_for(out, cfg)

    if cfg.force_cache_prompt:
        out["cache_prompt"] = True
    if cfg.cache_reuse:
        out.setdefault("n_cache_reuse", cfg.cache_reuse)

    # llama.cpp's OpenAI-compatible server honors id_slot for slot affinity.
    # OpenCode/AI SDK cache keys are intentionally nonstandard; preserve them
    # for upstreams that understand them, but do not rely on them for routing.
    out.setdefault("id_slot", slot)
    out.setdefault("prompt_cache_key", cache_key)
    return out, slot, cache_key


def upstream_url(cfg: ProxyConfig, path: str) -> str:
    clean = path.lstrip("/")
    if clean.startswith("v1/"):
        clean = clean[3:]
    return urljoin(cfg.upstream_base_url, clean)


def create_app(cfg: ProxyConfig | None = None) -> FastAPI:
    cfg = cfg or make_config()
    app = FastAPI(title="llama-suite OpenCode proxy")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "upstream_base_url": cfg.upstream_base_url.rstrip("/"),
            "slots": cfg.slots,
            "cache_reuse": cfg.cache_reuse,
            "force_cache_prompt": cfg.force_cache_prompt,
        }

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_v1(path: str, request: Request) -> Response:
        url = upstream_url(cfg, path)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in {"host", "content-length"}}
        params = dict(request.query_params)
        body = await request.body()
        slot: int | None = None
        cache_key = ""

        if request.method.upper() == "POST" and path == "chat/completions" and body:
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                payload, slot, cache_key = prepare_chat_payload(payload, cfg)
                body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                headers["content-type"] = "application/json"

        started = time.perf_counter()
        try:
            upstream = requests.request(
                request.method,
                url,
                params=params,
                data=body if body else None,
                headers=headers,
                stream=True,
                timeout=cfg.stream_timeout_s if path == "chat/completions" else cfg.request_timeout_s,
            )
        except requests.RequestException as exc:
            return JSONResponse({"error": f"upstream request failed: {exc}"}, status_code=502)

        response_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in {"content-encoding", "transfer-encoding", "connection"}
        }
        response_headers["x-llama-suite-proxy"] = "opencode"
        response_headers["x-llama-suite-upstream-ms"] = str(int((time.perf_counter() - started) * 1000))
        if slot is not None:
            response_headers["x-llama-suite-slot"] = str(slot)
        if cache_key:
            response_headers["x-llama-suite-cache-key"] = cache_key[:32]

        return StreamingResponse(
            upstream.iter_content(chunk_size=8192),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
        )

    return app


app = create_app()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible OpenCode proxy for llama.cpp prompt cache reuse.")
    parser.add_argument("--host", default=os.getenv("LLAMA_SUITE_PROXY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LLAMA_SUITE_PROXY_PORT", "8081")))
    parser.add_argument("--upstream", default=os.getenv("LLAMA_SUITE_PROXY_UPSTREAM", DEFAULT_UPSTREAM_BASE_URL))
    parser.add_argument("--slots", type=int, default=int(os.getenv("LLAMA_SUITE_PROXY_SLOTS", "1")))
    parser.add_argument("--cache-reuse", type=int, default=int(os.getenv("LLAMA_SUITE_PROXY_CACHE_REUSE", "256")))
    args = parser.parse_args(argv)

    os.environ["LLAMA_SUITE_PROXY_UPSTREAM"] = args.upstream
    os.environ["LLAMA_SUITE_PROXY_SLOTS"] = str(args.slots)
    os.environ["LLAMA_SUITE_PROXY_CACHE_REUSE"] = str(args.cache_reuse)

    import uvicorn

    uvicorn.run("llama_suite.proxy.opencode:create_app", factory=True, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
