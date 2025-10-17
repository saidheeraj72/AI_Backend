from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.core.config import Settings, get_settings

auth_scheme = HTTPBearer(auto_error=False)


@dataclass
class SupabaseUser:
    """Represents an authenticated Supabase user."""

    id: str
    claims: Dict[str, Any]


def require_supabase_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(auth_scheme),
    settings: Settings = Depends(get_settings),
) -> SupabaseUser:
    """Validate the Supabase access token from the Authorization header."""

    if not settings.supabase_jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase JWT secret is not configured on the server",
        )

    unauthorized_headers = {"WWW-Authenticate": "Bearer"}

    if credentials is None or not credentials.scheme or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization credentials were not provided",
            headers=unauthorized_headers,
        )

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must use Bearer scheme",
            headers=unauthorized_headers,
        )

    token = credentials.credentials

    try:
        claims = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token has expired",
            headers=unauthorized_headers,
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token",
            headers=unauthorized_headers,
        ) from exc

    user_id = str(claims.get("sub") or claims.get("user_id") or "").strip()
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token does not include a user identifier",
            headers=unauthorized_headers,
        )

    return SupabaseUser(id=user_id, claims=claims)
