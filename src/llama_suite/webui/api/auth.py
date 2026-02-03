from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel

from llama_suite.webui.utils.auth import (
    _get_configured_api_key,
    auth_enabled,
    auth_ttl_seconds,
    cookie_name,
    cookie_secure,
    make_session_token,
)
from llama_suite.webui.utils.auth import require_api_key as _require_api_key


router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    api_key: str


@router.get("/status")
async def get_auth_status(request: Request):
    """
    Returns whether auth is enabled and whether the current request is authenticated.

    This endpoint is intentionally public so the SPA can decide whether to prompt for login.
    """
    if not auth_enabled():
        return {"enabled": False, "authenticated": True}
    try:
        _require_api_key(request)
        return {"enabled": True, "authenticated": True}
    except HTTPException:
        return {"enabled": True, "authenticated": False}


@router.post("/login")
async def login(response: Response, body: LoginRequest):
    secret = _get_configured_api_key()
    if not secret:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="LLAMA_SUITE_API_KEY is not set")

    presented = (body.api_key or "").strip()
    if presented != secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = make_session_token(secret, ttl_seconds=auth_ttl_seconds())
    response.set_cookie(
        key=cookie_name(),
        value=token,
        httponly=True,
        secure=cookie_secure(),
        samesite="lax",
        path="/",
        max_age=auth_ttl_seconds(),
    )
    return {"status": "ok"}


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(key=cookie_name(), path="/")
    return {"status": "ok"}
