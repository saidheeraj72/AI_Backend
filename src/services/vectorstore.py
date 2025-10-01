from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

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
