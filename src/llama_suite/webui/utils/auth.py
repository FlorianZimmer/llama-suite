from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from typing import Optional

from fastapi import HTTPException, Request, WebSocket, status


def _get_configured_api_key() -> Optional[str]:
    key = (os.getenv("LLAMA_SUITE_API_KEY") or "").strip()
    return key or None


def auth_enabled() -> bool:
    return _get_configured_api_key() is not None


def cookie_name() -> str:
    return (os.getenv("LLAMA_SUITE_AUTH_COOKIE_NAME") or "llama_suite_auth").strip() or "llama_suite_auth"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * ((4 - (len(data) % 4)) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("utf-8"))


def _sign(secret: str, msg: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).digest()
    return _b64url(digest)


def make_session_token(secret: str, ttl_seconds: int) -> str:
    exp = int(time.time()) + max(60, int(ttl_seconds))
    nonce = secrets.token_urlsafe(16)
    payload = f"{exp}.{nonce}"
    sig = _sign(secret, payload)
    return f"v1.{payload}.{sig}"


def _verify_session_token(secret: str, token: str) -> bool:
    # Back-compat: allow raw API key in cookie (not recommended).
    if token == secret:
        return True

    parts = token.split(".")
    if len(parts) != 4:
        return False
    version, exp_s, nonce, sig = parts
    if version != "v1":
        return False

    try:
        exp = int(exp_s)
    except ValueError:
        return False

    if exp <= int(time.time()):
        return False

    expected = _sign(secret, f"{exp_s}.{nonce}")
    return hmac.compare_digest(sig, expected)


def _extract_presented_key(request: Request) -> Optional[str]:
    hdr = (request.headers.get("X-LLAMA-SUITE-API-KEY") or "").strip()
    if hdr:
        return hdr
    return request.cookies.get(cookie_name())


def require_api_key(request: Request) -> None:
    secret = _get_configured_api_key()
    if not secret:
        return

    presented = _extract_presented_key(request)
    if not presented:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    if presented == secret:
        return
    if _verify_session_token(secret, presented):
        return

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")


def websocket_authenticated(websocket: WebSocket) -> bool:
    secret = _get_configured_api_key()
    if not secret:
        return True

    presented = None
    try:
        presented = websocket.headers.get("X-LLAMA-SUITE-API-KEY")
    except Exception:
        presented = None
    presented = (presented or "").strip() or websocket.cookies.get(cookie_name())
    if not presented:
        return False
    if presented == secret:
        return True
    return _verify_session_token(secret, presented)


def cookie_secure() -> bool:
    v = (os.getenv("LLAMA_SUITE_SECURE_COOKIES") or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def auth_ttl_seconds() -> int:
    raw = (os.getenv("LLAMA_SUITE_AUTH_TTL_SECONDS") or "").strip()
    try:
        return int(raw) if raw else 60 * 60 * 24
    except ValueError:
        return 60 * 60 * 24

