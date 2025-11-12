from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Sequence

from supabase import Client, create_client

from src.core.config import Settings
from src.models.schemas import RAGSource
from src.services.llm import LLMService, LLM_MODELS
from src.services.metadata import MetadataService
from src.services.vectorstore import VectorStoreService


class RAGChatService:
    """Coordinate retrieval-augmented generation over the indexed document corpus."""

    def __init__(
        self,
        *,
        settings: Settings,
        vector_service: VectorStoreService,
        metadata_service: MetadataService,
        llm_service: LLMService,
        logger: Optional[logging.Logger] = None,
        max_snippet_chars: int = 1200,
    ) -> None:
        self.settings = settings
        self.vector_service = vector_service
        self.metadata_service = metadata_service
        self.llm_service = llm_service
        self.logger = logger or logging.getLogger(__name__)
        self.max_snippet_chars = max_snippet_chars
        self._supabase_client: Optional[Client] = None
        self._supabase_bucket = settings.supabase_storage_bucket
        self._supabase_signed_url_ttl = max(settings.supabase_signed_url_ttl, 1)

        if settings.supabase_url and settings.supabase_key and self._supabase_bucket:
            try:
                self._supabase_client = create_client(settings.supabase_url, settings.supabase_key)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to initialize Supabase client for RAG links: %s", exc)
                self._supabase_client = None

    def chat_with_documents(
        self,
        *,
        model_key: str,
        question: str,
        document_paths: Optional[Sequence[str]],
        top_k: int,
        document_records: Optional[Sequence[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("Question must not be empty")

        document_records = list(document_records or self.metadata_service.list_documents())
        if not document_records:
            raise ValueError("No documents are available. Please ingest documents before chatting.")

        normalized_to_original: dict[str, str] = {}
        directory_to_docs: dict[str, list[str]] = {}
        for record in document_records:
            relative_path = record.get("relative_path")
            if not relative_path:
                continue
            normalized_path = _normalize_relative_path(relative_path)
            normalized_to_original[normalized_path] = relative_path

            parent = PurePosixPath(normalized_path).parent
            directory_key = _normalize_relative_path(parent.as_posix())
            directory_to_docs.setdefault(directory_key, []).append(relative_path)

        if not normalized_to_original:
            raise ValueError("No documents are available. Please ingest documents before chatting.")

        allowed_paths = None
        if document_paths is not None:
            if not document_paths:
                raise ValueError("Select at least one document or enable use_all")

            resolved_paths: list[str] = []
            unknown_entries: list[str] = []

            for entry in document_paths:
                normalized_entry = _normalize_relative_path(entry)

                if normalized_entry in normalized_to_original:
                    resolved_paths.append(normalized_to_original[normalized_entry])
                    continue

                directory_docs = directory_to_docs.get(normalized_entry, [])
                if directory_docs:
                    resolved_paths.extend(directory_docs)
                    continue

                unknown_entries.append(entry)

            if unknown_entries:
                raise ValueError(
                    "Unknown documents selected: "
                    + ", ".join(sorted(set(unknown_entries)))
                )

            if not resolved_paths:
                raise ValueError("No valid documents found for the selected folders.")

            # Preserve order while removing duplicates
            seen: set[str] = set()
            allowed_paths = []
            for path in resolved_paths:
                if path not in seen:
                    seen.add(path)
                    allowed_paths.append(path)

        results = self.vector_service.search(
            question,
            document_paths=allowed_paths,
            top_k=top_k,
        )

        model_info = LLM_MODELS.get(model_key)
        if model_info is None:
            raise ValueError(f"Unknown model key: {model_key}")

        if not results:
            self.logger.info("No vector matches found for question")
            return {
                "model_key": model_key,
                "model_name": model_info["name"],
                "response": (
                    "I couldn't find any matching context in the selected documents. "
                    "Try broadening your selection or ingesting more files."
                ),
                "usage": None,
                "question": question,
                "sources": [],
            }

        sources: list[RAGSource] = []
        context_sections: list[str] = []
        for document, _score in results:
            raw_source = document.metadata.get("source", "unknown")
            download_url = self._build_download_url(raw_source)
            snippet = self._prepare_snippet(document.page_content)
            filename = self._extract_filename(raw_source)
            sources.append(RAGSource(filename=filename, download_url=download_url))
            context_sections.append(
                f"Source: {raw_source}\nSnippet: {snippet}"
            )

        context_block = "\n\n".join(context_sections)
        prompt = (
            "You are a helpful assistant replying with information grounded in the provided "
            "document excerpts. Base your answer only on that context, and cite the source "
            "filenames when relevant. If the context is insufficient, clearly say so.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        llm_result = self.llm_service.chat(model_key, prompt=prompt)

        return {
            **llm_result,
            "question": question,
            "sources": [source.dict() for source in sources],
        }

    def _prepare_snippet(self, text: str) -> str:
        collapsed = " ".join(text.split())
        if len(collapsed) <= self.max_snippet_chars:
            return collapsed
        return collapsed[: self.max_snippet_chars].rstrip() + "..."

    def _build_download_url(self, relative_path: str) -> Optional[str]:
        if not relative_path:
            return None

        normalized = relative_path.lstrip("/")

        if self._supabase_client and self._supabase_bucket:
            try:
                bucket = self._supabase_client.storage.from_(self._supabase_bucket)
                signed = bucket.create_signed_url(
                    normalized, self._supabase_signed_url_ttl
                )
                signed_url = signed.get("signedURL") or signed.get("signedUrl")
                if signed_url:
                    return signed_url
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error(
                    "Failed to create signed URL for %s: %s", normalized, exc
                )

        if self.settings.supabase_public_storage_base:
            return f"{self.settings.supabase_public_storage_base}/{normalized}"

        return None

    def _extract_filename(self, relative_path: str) -> str:
        if not relative_path or relative_path == "unknown":
            return "unknown"

        return Path(relative_path).name or relative_path


def _normalize_relative_path(path: Optional[str]) -> str:
    if not path:
        return ""
    candidate = PurePosixPath(path)
    parts = [
        segment
        for segment in candidate.parts
        if segment not in {"", ".", ".."}
    ]
    return "/".join(parts)
