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

        self.pdf_directory = (base_dir / os.getenv("PDF_DIRECTORY", pdf_directory_default)).resolve()
        self.documents_metadata_path = (
            base_dir / os.getenv("DOCUMENTS_METADATA_PATH", metadata_default)
        ).resolve()
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.serper_api_key = os.getenv("SERPER_API_KEY", "")

        self.supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.supabase_key = os.getenv("SUPABASE_KEY", "")
        self.supabase_chat_table = os.getenv("SUPABASE_CHAT_TABLE", "chat_history")
        self.supabase_storage_bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "pdfs")
        self.supabase_documents_table = os.getenv(
            "SUPABASE_DOCUMENTS_TABLE", "documents_metadata"
        )
        self.supabase_configured = bool(
            self.supabase_url and self.supabase_key and self.supabase_storage_bucket
        )

        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_index = os.getenv("PINECONE_INDEX", "")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "")
        self.pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "")
        self.pinecone_configured = bool(
            self.pinecone_api_key and self.pinecone_index
        )
        self.supabase_signed_url_ttl = int(os.getenv("SUPABASE_SIGNED_URL_TTL", "3600"))
        if self.supabase_url and self.supabase_storage_bucket:
            self.supabase_public_storage_base = (
                f"{self.supabase_url}/storage/v1/object/public/{self.supabase_storage_bucket}"
            )
        else:
            self.supabase_public_storage_base = ""

        origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*")
        methods_env = os.getenv("CORS_ALLOW_METHODS", "*")
        headers_env = os.getenv("CORS_ALLOW_HEADERS", "*")

        self.cors_allow_origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()] or ["*"]
        vercel_origin = "https://smart-biz-desk.vercel.app"
        if "*" not in self.cors_allow_origins and vercel_origin not in self.cors_allow_origins:
            self.cors_allow_origins.append(vercel_origin)
        self.cors_allow_methods = [method.strip().upper() for method in methods_env.split(",") if method.strip()] or ["*"]
        self.cors_allow_headers = [header.strip() for header in headers_env.split(",") if header.strip()] or ["*"]
        self.cors_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

        if not self.supabase_configured:
            self.pdf_directory.mkdir(parents=True, exist_ok=True)
        self.documents_metadata_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()
