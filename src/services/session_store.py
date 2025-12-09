from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np
from langchain_core.documents import Document

class SessionStoreService:
    """
    In-memory storage for session-specific document embeddings.
    This allows for temporary RAG within a chat session without persistent storage.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        # Structure: { session_id: { "vectors": np.ndarray, "documents": list[Document] } }
        self._store: dict[str, dict[str, Any]] = {}

    def add_documents(
        self, 
        session_id: str, 
        documents: list[Document], 
        embeddings: list[list[float]]
    ) -> None:
        """Add document chunks and their embeddings to the session store."""
        if not documents or not embeddings:
            return

        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")

        # Convert embeddings to numpy array for efficient calculation
        new_vectors = np.array(embeddings, dtype=np.float32)

        if session_id not in self._store:
            self._store[session_id] = {
                "vectors": new_vectors,
                "documents": documents
            }
        else:
            # Append to existing
            existing_vectors = self._store[session_id]["vectors"]
            self._store[session_id]["vectors"] = np.vstack((existing_vectors, new_vectors))
            self._store[session_id]["documents"].extend(documents)

        self.logger.info(
            "Added %d chunks to session %s (Total: %d)", 
            len(documents), 
            session_id, 
            len(self._store[session_id]["documents"])
        )

    def search(
        self, 
        session_id: str, 
        query_vector: list[float], 
        top_k: int = 4
    ) -> list[Tuple[Document, float]]:
        """
        Perform cosine similarity search for the session.
        Returns a list of (Document, score).
        """
        if session_id not in self._store:
            return []

        session_data = self._store[session_id]
        vectors = session_data["vectors"]  # shape (N, D)
        documents = session_data["documents"]
        
        query_vec = np.array(query_vector, dtype=np.float32) # shape (D,)

        # Compute cosine similarity
        # Similarity = (A . B) / (||A|| ||B||)
        # Assuming query_vec and vectors are already normalized (HuggingFaceBgeEmbeddings does this if encode_kwargs={'normalize_embeddings': True})
        # If not, we should normalize them here. 
        # But let's assume the embedding model handles normalization or we do it cheaply here.
        
        # Safe normalization
        norm_vectors = np.linalg.norm(vectors, axis=1)
        norm_query = np.linalg.norm(query_vec)
        
        if norm_query == 0:
            return []

        # Avoid division by zero
        norm_vectors[norm_vectors == 0] = 1e-10

        scores = np.dot(vectors, query_vec) / (norm_vectors * norm_query)
        
        # Get top K indices
        # argsort returns indices that sort the array, we want descending order
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((documents[idx], float(scores[idx])))
            
        return results

    def clear_session(self, session_id: str) -> None:
        """Remove all data for a session."""
        if session_id in self._store:
            del self._store[session_id]
            self.logger.info("Cleared session %s", session_id)
