from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.core.config import get_settings


bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class SupabaseUser:
    """Lightweight representation of the authenticated Supabase user."""

    id: str
    email: Optional[str]
    role: Optional[str]
    claims: Mapping[str, Any]


def _decode_supabase_token(token: str, *, secret: str) -> Mapping[str, Any]:
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase JWT secret is not configured",
        )

    secret = secret.strip().strip('"')
    try:
        return jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Supabase token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Supabase token") from exc


def _build_user_from_claims(claims: Mapping[str, Any]) -> SupabaseUser:
    user_id = claims.get("sub") or claims.get("user_id")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Supabase token missing user id")

    email = claims.get("email") or claims.get("user_metadata", {}).get("email")
    role = claims.get("role") or claims.get("app_metadata", {}).get("role")

    return SupabaseUser(id=str(user_id), email=email, role=role, claims=claims)


async def require_supabase_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> SupabaseUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")

    token = credentials.credentials.strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Supabase token is required")

    settings = get_settings()
    claims = _decode_supabase_token(token, secret=settings.supabase_jwt_secret)
    return _build_user_from_claims(claims)
