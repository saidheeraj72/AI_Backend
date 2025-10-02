from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


class Settings:
    """Application configuration loaded from environment variables."""

    def __init__(self) -> None:
        base_dir = Path(os.getenv("BASE_DIR", ".")).resolve()
        self.app_name = os.getenv("APP_NAME", "AI Backend Service")
        pdf_directory_default = "data"
        metadata_default = "documents_metadata.json"
        faiss_index_default = "faiss_index"

        self.pdf_directory = (base_dir / os.getenv("PDF_DIRECTORY", pdf_directory_default)).resolve()
        self.documents_metadata_path = (
            base_dir / os.getenv("DOCUMENTS_METADATA_PATH", metadata_default)
        ).resolve()
        self.faiss_index_path = (
            base_dir / os.getenv("FAISS_INDEX_PATH", faiss_index_default)
        ).resolve()
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.serper_api_key = os.getenv("SERPER_API_KEY", "")

        self.supabase_url = os.getenv("SUPABASE_URL", "")
        self.supabase_key = os.getenv("SUPABASE_KEY", "")
        self.supabase_chat_table = os.getenv("SUPABASE_CHAT_TABLE", "chat_history")

        origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*")
        methods_env = os.getenv("CORS_ALLOW_METHODS", "*")
        headers_env = os.getenv("CORS_ALLOW_HEADERS", "*")

        self.cors_allow_origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()] or ["*"]
        self.cors_allow_methods = [method.strip().upper() for method in methods_env.split(",") if method.strip()] or ["*"]
        self.cors_allow_headers = [header.strip() for header in headers_env.split(",") if header.strip()] or ["*"]
        self.cors_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

        self.pdf_directory.mkdir(parents=True, exist_ok=True)
        self.documents_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()
