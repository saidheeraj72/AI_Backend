from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.core.config import Settings


class VectorStoreService:
    """Manage FAISS persistence for document embeddings."""

    def __init__(self, settings: Settings, *, logger: Optional[logging.Logger] = None) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)
        self.vectorstore: Optional[FAISS] = None
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    @property
    def index_path(self) -> Path:
        return self.settings.faiss_index_path

    def load(self) -> bool:
        """Load an existing FAISS index from disk."""
        try:
            self.vectorstore = FAISS.load_local(
                str(self.index_path),
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
            self.logger.info("Loaded FAISS index from %s", self.index_path)
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Unable to load FAISS index: %s", exc)
            self.vectorstore = None
            return False

    def _ensure_index(self) -> None:
        if self.vectorstore is None:
            self.load()

    def search(
        self,
        query: str,
        *,
        document_paths: Optional[Sequence[str]] = None,
        top_k: int = 4,
        oversample: int = 3,
    ) -> list[Tuple[Document, float]]:
        """Retrieve the most similar chunks for a query, optionally scoped to selected documents."""

        if not query.strip():
            raise ValueError("Query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        self._ensure_index()
        if self.vectorstore is None:
            raise ValueError("No document index is available. Ingest documents before querying.")

        allowed_sources = {path for path in (document_paths or []) if path}

        sample_size = max(top_k, top_k * max(1, oversample))
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=sample_size)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Vector search failed: %s", exc)
            raise ValueError("Vector search failed") from exc

        if allowed_sources:
            filtered = [
                (document, score)
                for document, score in results
                if document.metadata.get("source") in allowed_sources
            ]
        else:
            filtered = results

        return filtered[:top_k]

    def index_documents(self, documents: Iterable[Document]) -> int:
        """Embed documents and persist them to the FAISS index."""
        docs = [doc for doc in documents if doc]
        if not docs:
            self.logger.warning("No documents provided for indexing")
            return 0

        splits = self.text_splitter.split_documents(docs)
        self.logger.info("Prepared %s text chunks for indexing", len(splits))

        self._ensure_index()
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(splits, self.embedding_model)
            self.logger.info("Created new FAISS index at %s", self.index_path)
        else:
            self.vectorstore.add_documents(splits)
            self.logger.info("Appended documents to existing FAISS index")

        self.vectorstore.save_local(str(self.index_path))
        self.logger.info("Persisted FAISS index to disk")
        return len(splits)
