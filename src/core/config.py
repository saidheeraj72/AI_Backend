from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


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
        
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.serper_api_key = os.getenv("SERPER_API_KEY", "")
        
        self.supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.supabase_key = os.getenv("SUPABASE_KEY", "")
        self.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY", "")
        self.supabase_jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "")
        self.supabase_chat_table = os.getenv("SUPABASE_CHAT_TABLE", "chat_history")
        self.supabase_storage_bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "pdfs")
        self.supabase_documents_table = os.getenv(
            "SUPABASE_DOCUMENTS_TABLE", "documents_metadata"
        )
        self.default_org_id = os.getenv("DEFAULT_ORGANIZATION_ID", "").strip()
        self.supabase_configured = bool(
            self.supabase_url and self.supabase_key and self.supabase_storage_bucket
        )
        
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_index = os.getenv("PINECONE_INDEX", "")
        self.pinecone_chat_index = os.getenv("PINECONE_CHAT_INDEX", "chat-session")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "")
        self.pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "")
        self.pinecone_configured = bool(
            self.pinecone_api_key and self.pinecone_index and self.pinecone_chat_index
        )
        
        self.supabase_signed_url_ttl = int(os.getenv("SUPABASE_SIGNED_URL_TTL", "3600"))
        
        if self.supabase_url and self.supabase_storage_bucket:
            self.supabase_public_storage_base = (
                f"{self.supabase_url}/storage/v1/object/public/{self.supabase_storage_bucket}"
            )
        else:
            self.supabase_public_storage_base = ""
        
        # CORS Configuration - UPDATED
        origins_env = os.getenv("CORS_ALLOW_ORIGINS", "")
        methods_env = os.getenv("CORS_ALLOW_METHODS", "GET,POST,PUT,DELETE,OPTIONS,PATCH")
        headers_env = os.getenv("CORS_ALLOW_HEADERS", "*")
        
        # Parse origins from environment
        if origins_env:
            self.cors_allow_origins = [
                origin.strip() for origin in origins_env.split(",") if origin.strip()
            ]
        else:
            # Default allowed origins
            self.cors_allow_origins = [
                "https://smart-biz-desk.vercel.app",
                "http://localhost:3000",
                "http://localhost:8080",
                "http://localhost:5173",
                "http://localhost:5174",
            ]
        
        # Add ngrok domains dynamically
        ngrok_patterns = [
            "https://*.ngrok-free.app",
            "https://*.ngrok.io",
            "https://*.ngrok.app",
        ]
        
        # Check if wildcard is used
        if "*" not in self.cors_allow_origins:
            # Ensure Vercel origin is included
            vercel_origin = "https://smart-biz-desk.vercel.app"
            if vercel_origin not in self.cors_allow_origins:
                self.cors_allow_origins.append(vercel_origin)
        
        # Parse methods
        self.cors_allow_methods = [
            method.strip().upper() for method in methods_env.split(",") if method.strip()
        ] or ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        
        # Parse headers
        if headers_env == "*":
            self.cors_allow_headers = ["*"]
        else:
            self.cors_allow_headers = [
                header.strip() for header in headers_env.split(",") if header.strip()
            ] or ["*"]
        
        # Allow credentials
        self.cors_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
        
        # Expose headers for frontend to read
        self.cors_expose_headers = ["*"]
        
        # Max age for preflight cache (1 hour)
        self.cors_max_age = int(os.getenv("CORS_MAX_AGE", "3600"))
        
        if not self.supabase_configured:
            self.pdf_directory.mkdir(parents=True, exist_ok=True)
            self.documents_metadata_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()