from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.core.config import get_settings

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/public", response_model=dict[str, object])
async def get_public_config() -> dict[str, object]:
    settings = get_settings()

    supabase_url = settings.supabase_url
    supabase_anon_key = settings.supabase_anon_key

    if not supabase_url or not supabase_anon_key:
        raise HTTPException(
            status_code=503,
            detail="Supabase public configuration is not available",
        )

    return {
        "supabase": {
            "url": supabase_url,
            "anon_key": supabase_anon_key,
            "storage_bucket": settings.supabase_storage_bucket,
        }
    }
