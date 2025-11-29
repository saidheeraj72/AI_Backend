from __future__ import annotations

import logging
from typing import Any, Optional

from supabase import Client, create_client

from src.core.config import Settings


class ChatHistoryService:
    """Persist chat interactions to the new chat_sessions/chat_messages tables."""

    def __init__(self, settings: Settings, *, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self._client: Optional[Client] = None
        
        if settings.supabase_url and settings.supabase_key:
            try:
                self._client = create_client(settings.supabase_url, settings.supabase_key)
            except Exception as exc:
                self.logger.error("Failed to initialize Supabase client: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _require_client(self) -> Client:
        if self._client is None:
            raise RuntimeError("Supabase chat history client is not configured")
        return self._client

    def save_chat(
        self,
        *,
        chat_id: str,
        user_id: str,
        model_key: str,
        model_name: str,
        user_input: Optional[str],
        response: str,
        usage: Optional[dict[str, Any]],
        has_image: bool,
        image_mime_type: Optional[str],
        interaction_type: str = "standard",
        metadata: Optional[dict[str, Any]] = None,
        org_id: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return

        client = self._require_client()

        # Update session (upsert)
        session_payload = {
            "id": chat_id,
            "user_id": user_id,
            "model": model_key,
            # "org_id": org_id # No longer in schema, potentially in metadata if needed?
        }
        try:
            client.table("chat_sessions").upsert(session_payload, on_conflict="id").execute()
        except Exception as exc:
            self.logger.error("Failed to upsert chat session %s: %s", chat_id, exc)

        # We are storing "interactions" where one message row contains both user input (in metadata) and response.
        # This is to maintain compatibility with existing frontend logic without full refactor.
        # Ideally, we should store two rows: one user message, one assistant message.
        
        message_payload = {
            "session_id": chat_id,
            "role": "assistant", # It's the response
            "content": response,
            "metadata": {
                "model_key": model_key,
                "model_name": model_name,
                "user_input": user_input,
                "usage": usage,
                "has_image": has_image,
                "image_mime_type": image_mime_type,
                "interaction_type": interaction_type,
                "extra": metadata or {},
            },
        }

        try:
            client.table("chat_messages").insert(message_payload).execute()
        except Exception as exc:
            self.logger.error("Failed to persist chat message: %s", exc)

    def get_chat_history(
        self,
        *,
        chat_id: str,
        user_id: str,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        client = self._require_client()
        query = (
            client.table("chat_messages")
            .select("session_id, role, content, metadata, created_at")
            .eq("session_id", chat_id)
            .order("created_at", desc=False)
        )
        if limit is not None:
            query = query.limit(limit)

        try:
            response = query.execute()
        except Exception as exc:
            self.logger.error("Failed to fetch chat history: %s", exc)
            raise

        messages: list[dict[str, Any]] = []
        for record in response.data or []:
            meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            messages.append(
                {
                    "chat_id": record.get("session_id"),
                    "user_id": user_id,
                    "interaction_type": meta.get("interaction_type", "standard"),
                    "model_key": meta.get("model_key"),
                    "model_name": meta.get("model_name"),
                    "user_input": meta.get("user_input"),
                    "response": record.get("content"),
                    "usage": meta.get("usage"),
                    "has_image": meta.get("has_image", False),
                    "image_mime_type": meta.get("image_mime_type"),
                    "metadata": meta.get("extra") or {},
                    "created_at": record.get("created_at"),
                }
            )
        return messages

    def list_user_sessions(self, *, user_id: str) -> list[dict[str, Any]]:
        client = self._require_client()
        try:
            sessions_response = (
                client.table("chat_sessions")
                .select("id, user_id, updated_at")
                .eq("user_id", user_id)
                .order("updated_at", desc=True)
                .execute()
            )
        except Exception as exc:
            self.logger.error("Failed to list chat sessions: %s", exc)
            raise

        session_ids = [record.get("id") for record in sessions_response.data or [] if record.get("id")]
        if not session_ids:
            return []

        try:
            messages_response = (
                client.table("chat_messages")
                .select("session_id, created_at, metadata, content")
                .in_("session_id", session_ids)
                .order("created_at", desc=True)
                .execute()
            )
        except Exception as exc:
            self.logger.error("Failed to load chat messages for sessions: %s", exc)
            raise

        session_stats: dict[str, dict[str, Any]] = {
            session_id: {
                "chat_id": session_id,
                "user_id": user_id,
                "message_count": 0,
                "last_message": None,
                "last_user_input": None,
                "last_model_key": None,
                "last_interaction_type": None,
                "last_interaction_at": None,
            }
            for session_id in session_ids
        }

        for record in messages_response.data or []:
            session_id = record.get("session_id")
            stats = session_stats.get(session_id)
            if not stats:
                continue
            stats["message_count"] += 1
            created_at = record.get("created_at")
            meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            if created_at and (stats["last_interaction_at"] is None or created_at >= stats["last_interaction_at"]):
                stats["last_interaction_at"] = created_at
                stats["last_message"] = record.get("content")
                stats["last_user_input"] = meta.get("user_input")
                stats["last_model_key"] = meta.get("model_key")
                stats["last_interaction_type"] = meta.get("interaction_type")

        return list(session_stats.values())

    def delete_chat_session(self, *, chat_id: str, user_id: str) -> int:
        client = self._require_client()
        try:
            # Cascade delete should handle messages, but we can delete explicitly too.
            client.table("chat_messages").delete().eq("session_id", chat_id).execute()
            response = (
                client.table("chat_sessions")
                .delete()
                .eq("id", chat_id)
                .eq("user_id", user_id)
                .execute()
            )
        except Exception as exc:
            self.logger.error("Failed to delete chat session: %s", exc)
            raise

        deleted = getattr(response, "count", None)
        if isinstance(deleted, int):
            return deleted
        data = getattr(response, "data", None)
        if isinstance(data, list):
            return len(data)
        return 0