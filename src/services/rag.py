from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

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

    def chat_with_documents(
        self,
        *,
        model_key: str,
        question: str,
        document_paths: Optional[Sequence[str]],
        top_k: int,
    ) -> dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("Question must not be empty")

        available_documents = {
            item["relative_path"] for item in self.metadata_service.list_documents()
        }
        if not available_documents:
            raise ValueError("No documents are available. Please ingest documents before chatting.")

        if document_paths is not None and not document_paths:
            raise ValueError("Select at least one document or enable use_all")

        allowed_paths = None
        if document_paths:
            invalid = sorted(set(document_paths) - available_documents)
            if invalid:
                raise ValueError(
                    f"Unknown documents requested: {', '.join(invalid)}"
                )
            allowed_paths = list(dict.fromkeys(document_paths))

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
        for document, score in results:
            raw_source = document.metadata.get("source", "unknown")
            snippet = self._prepare_snippet(document.page_content)
            sources.append(
                RAGSource(document=raw_source, score=float(score), snippet=snippet)
            )
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
