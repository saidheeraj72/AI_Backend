from __future__ import annotations

import logging
import os
from logging.config import dictConfig

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.api.routes.chat import rag_router, router as chat_router
from src.api.routes.config import router as config_router
from src.api.routes.documents import router as documents_router
from src.api.routes.settings import router as settings_router
from src.api.routes.team_profiles import router as team_profiles_router
from src.api.routes.notifications import router as notifications_router
from src.core.config import get_settings

settings = get_settings()


def configure_logging() -> None:
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": log_level,
                }
            },
            "root": {"handlers": ["console"], "level": log_level},
        }
    )


def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(title=settings.app_name)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    application.include_router(chat_router)
    application.include_router(rag_router)
    application.include_router(documents_router)
    application.include_router(config_router)
    application.include_router(settings_router)
    application.include_router(team_profiles_router)
    application.include_router(notifications_router)

    @application.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    # Serve React App
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dist_dir = os.path.join(base_dir, "smart-biz-desk", "dist")

    if os.path.exists(dist_dir):
        # Mount assets if the directory exists
        assets_dir = os.path.join(dist_dir, "assets")
        if os.path.isdir(assets_dir):
            application.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @application.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            # Check if file exists in dist
            file_path = os.path.join(dist_dir, full_path)
            if os.path.isfile(file_path):
                return FileResponse(file_path)
            
            # Fallback to index.html for SPA routing
            return FileResponse(os.path.join(dist_dir, "index.html"))
    else:
        # Use root logger as specific logger might not be configured for this scope or rely on existing setup
        logging.getLogger("uvicorn").warning(f"Frontend dist directory not found at {dist_dir}. SPA serving disabled.")

    return application


app = create_app()
