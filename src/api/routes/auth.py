from __future__ import annotations

from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr

from src.core.config import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None


def _build_supabase_headers() -> dict[str, str]:
    settings = get_settings()
    api_key = settings.supabase_anon_key or settings.supabase_key
    if not settings.supabase_url or not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase authentication is not configured",
        )
    return {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


async def _supabase_request(method: str, url: str, *, json: dict[str, Any]) -> dict[str, Any]:
    import logging
    logger = logging.getLogger(__name__)

    headers = _build_supabase_headers()
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.request(method, url, headers=headers, json=json)
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if response.status_code >= 400:
        logger.error(f"Supabase error: status={response.status_code}, payload={payload}, url={url}, request_body={json}")
        error_detail = (
            payload.get("error_description")
            or payload.get("msg")
            or payload.get("message")
            or "Authentication request failed"
        )
        raise HTTPException(status_code=response.status_code, detail=error_detail)
    return payload


@router.post("/login")
async def login(request: LoginRequest) -> dict[str, Any]:
    settings = get_settings()
    if not settings.supabase_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase authentication is not configured",
        )
    url = f"{settings.supabase_url}/auth/v1/token?grant_type=password"
    payload = await _supabase_request(
        "POST",
        url,
        json={
            "email": request.email,
            "password": request.password,
        },
    )
    access_token = payload.get("access_token")
    if not access_token:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Supabase did not return an access token")

    user_data = payload.get("user", {})
    return {
        "access_token": access_token,
        "refresh_token": payload.get("refresh_token"),
        "expires_in": payload.get("expires_in"),
        "token_type": payload.get("token_type"),
        "user": {
            "id": user_data.get("id"),
            "email": user_data.get("email"),
            "full_name": user_data.get("user_metadata", {}).get("full_name") or user_data.get("email"),
        },
    }


@router.post("/signup")
async def signup(request: SignupRequest) -> dict[str, Any]:
    settings = get_settings()
    if not settings.supabase_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase authentication is not configured",
        )
    url = f"{settings.supabase_url}/auth/v1/signup"
    metadata: dict[str, Any] = {}
    if request.full_name:
        metadata["full_name"] = request.full_name
    payload = await _supabase_request(
        "POST",
        url,
        json={
            "email": request.email,
            "password": request.password,
            "data": metadata or None,
        },
    )
    # Supabase may require email confirmation before issuing a token
    user_data = payload.get("user", {})
    response: dict[str, Any] = {
        "user": {
            "id": user_data.get("id"),
            "email": user_data.get("email"),
            "full_name": user_data.get("user_metadata", {}).get("full_name") or user_data.get("email"),
        },
        "access_token": payload.get("access_token"),
        "refresh_token": payload.get("refresh_token"),
        "expires_in": payload.get("expires_in"),
        "token_type": payload.get("token_type"),
    }
    if response["access_token"] is None:
        response["detail"] = "Confirmation required. Please check your email to activate the account."
    return response
