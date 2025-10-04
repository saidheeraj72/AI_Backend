from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence, Tuple

try:
    from pinecone import Pinecone
except ImportError:  # pragma: no cover - optional dependency
    Pinecone = None

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.core.config import Settings


class VectorStoreService:
    """Manage document embeddings in Pinecone."""

    def __init__(self, settings: Settings, *, logger: Optional[logging.Logger] = None) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self._pinecone_client = None
        self._pinecone_index = None
        self._pinecone_namespace = (self.settings.pinecone_namespace or "").strip()

        if not self.settings.pinecone_configured:
            raise RuntimeError(
                "Pinecone credentials are required; set PINECONE_API_KEY and PINECONE_INDEX"
            )

        if Pinecone is None:
            raise RuntimeError(
                "The 'pinecone' package is not installed. Run 'pip install pinecone'."
            )

        try:
            client_kwargs = {"api_key": self.settings.pinecone_api_key}
            if getattr(self.settings, "pinecone_environment", ""):
                client_kwargs["environment"] = self.settings.pinecone_environment

            self._pinecone_client = Pinecone(**client_kwargs)
            self._pinecone_index = self._pinecone_client.Index(
                self.settings.pinecone_index
            )
            self.logger.info(
                "Initialized Pinecone index '%s'",
                self.settings.pinecone_index,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to initialize Pinecone client: %s", exc)
            raise

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

        return self._search_pinecone(
            query, document_paths=document_paths, top_k=top_k, oversample=oversample
        )

    def index_documents(self, documents: Iterable[Document]) -> int:
        """Embed documents and persist them to the configured vector store."""
        docs = [doc for doc in documents if doc]
        if not docs:
            self.logger.warning("No documents provided for indexing")
            return 0

        splits = self.text_splitter.split_documents(docs)
        self.logger.info("Prepared %s text chunks for indexing", len(splits))

        return self._index_pinecone(splits)

    def remove_documents(self, sources: Sequence[str]) -> int:
        """Remove all embeddings associated with the provided source paths."""
        return self._remove_pinecone(sources)

    # Pinecone helpers -------------------------------------------------

    def _index_pinecone(self, splits: list[Document]) -> int:
        if self._pinecone_index is None:
            raise ValueError("Pinecone index is not configured")

        payloads: list[dict] = []
        texts = [chunk.page_content for chunk in splits]
        if not texts:
            self.logger.warning("No text chunks available for Pinecone indexing")
            return 0

        embeddings = self.embedding_model.embed_documents(texts)
        per_source_counts: dict[str, int] = {}

        for chunk, vector in zip(splits, embeddings):
            source = chunk.metadata.get("source", "")
            counter = per_source_counts.get(source, 0)
            per_source_counts[source] = counter + 1
            vector_id = f"{source}::chunk-{counter}"
            payloads.append(
                {
                    "id": vector_id,
                    "values": vector,
                    "metadata": {
                        "source": source,
                        "text": chunk.page_content,
                    },
                }
            )

        if not payloads:
            return 0

        namespace = self._pinecone_namespace or None
        try:
            self._pinecone_index.upsert(vectors=payloads, namespace=namespace)
            self.logger.info(
                "Upserted %s vectors to Pinecone index %s",
                len(payloads),
                self.settings.pinecone_index,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Failed to upsert vectors to Pinecone: %s", exc)
            raise

        return len(payloads)

    def _search_pinecone(
        self,
        query: str,
        *,
        document_paths: Optional[Sequence[str]] = None,
        top_k: int,
        oversample: int,
    ) -> list[Tuple[Document, float]]:
        if self._pinecone_index is None:
            raise ValueError("Pinecone index is not configured")

        vector = self.embedding_model.embed_query(query)
        allowed_sources = [path for path in (document_paths or []) if path]
        metadata_filter = None
        if allowed_sources:
            if len(allowed_sources) == 1:
                metadata_filter = {"source": {"$eq": allowed_sources[0]}}
            else:
                metadata_filter = {"source": {"$in": allowed_sources}}

        sample_size = max(top_k, top_k * max(1, oversample))
        sample_size = max(1, min(sample_size, 100))  # Pinecone limit safeguard

        namespace = self._pinecone_namespace or None
        try:
            response = self._pinecone_index.query(
                vector=vector,
                top_k=sample_size,
                include_metadata=True,
                namespace=namespace,
                filter=metadata_filter,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Pinecone query failed: %s", exc)
            raise ValueError("Vector search failed") from exc

        matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])
        results: list[Tuple[Document, float]] = []
        for match in matches:
            raw_metadata = match.get("metadata") or {}
            metadata = dict(raw_metadata)
            text = metadata.pop("text", "")
            if not text:
                continue
            doc = Document(page_content=text, metadata=metadata)
            score = match.get("score")
            results.append((doc, score))

        return results[:top_k]

    def _remove_pinecone(self, sources: Sequence[str]) -> int:
        if self._pinecone_index is None:
            self.logger.warning("Pinecone index not available; nothing to remove")
            return 0

        normalized = [source for source in sources if source]
        if not normalized:
            return 0

        if len(normalized) == 1:
            metadata_filter = {"source": {"$eq": normalized[0]}}
        else:
            metadata_filter = {"source": {"$in": normalized}}

        namespace = self._pinecone_namespace or None

        try:
            stats = self._pinecone_index.describe_index_stats(filter=metadata_filter)
        except Exception:  # pragma: no cover - defensive logging
            stats = {}

        namespace_key = self._pinecone_namespace or ""
        if isinstance(stats, dict):
            vector_count = (
                stats.get("namespaces", {})
                .get(namespace_key, {})
                .get("vector_count", 0)
            )
        else:
            vector_count = 0

        try:
            self._pinecone_index.delete(
                filter=metadata_filter,
                namespace=namespace,
            )
            self.logger.info(
                "Requested deletion of vectors with sources %s from Pinecone",
                sorted(normalized),
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Failed to delete vectors from Pinecone: %s", exc)
            raise

        return int(vector_count)
