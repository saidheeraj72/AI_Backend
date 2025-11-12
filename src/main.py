from __future__ import annotations

import logging
from logging.config import dictConfig

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.chat import rag_router, router as chat_router
from src.api.routes.config import router as config_router
from src.api.routes.documents import router as documents_router
from src.api.routes.settings import router as settings_router
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

    @application.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
