from __future__ import annotations

import logging
from typing import Any, Optional

from supabase import Client, create_client

from src.core.config import Settings


class ChatHistoryService:
    """Persist chat interactions to Supabase."""

    def __init__(self, settings: Settings, *, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.table_name = settings.supabase_chat_table
        self._client: Optional[Client] = None

        if settings.supabase_url and settings.supabase_key:
            try:
                self._client = create_client(settings.supabase_url, settings.supabase_key)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to initialize Supabase client: %s", exc)
        else:
            self.logger.info("Supabase credentials not provided; chat history persistence disabled")

    @property
    def enabled(self) -> bool:
        return self._client is not None

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
    ) -> None:
        if not self.enabled:
            return

        payload = {
            "chat_id": chat_id,
            "user_id": user_id,
            "interaction_type": interaction_type,
            "model_key": model_key,
            "model_name": model_name,
            "user_input": user_input,
            "response": response,
            "usage": usage,
            "has_image": has_image,
            "image_mime_type": image_mime_type,
            "metadata": metadata,
        }

        try:
            self._client.table(self.table_name).insert(payload).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to persist chat history: %s", exc)

    def get_chat_history(
        self,
        *,
        chat_id: str,
        user_id: str,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            raise RuntimeError("Supabase chat history client is not configured")

        query = (
            self._client.table(self.table_name)
            .select("*")
            .eq("chat_id", chat_id)
            .eq("user_id", user_id)
            .order("created_at", desc=False)
        )

        if limit is not None:
            query = query.limit(limit)

        try:
            response = query.execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to fetch chat history: %s", exc)
            raise

        return response.data or []

    def list_user_sessions(self, *, user_id: str) -> list[dict[str, Any]]:
        if not self.enabled:
            raise RuntimeError("Supabase chat history client is not configured")

        try:
            response = (
                self._client.table(self.table_name)
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to list chat sessions: %s", exc)
            raise

        records = response.data or []
        sessions: dict[str, dict[str, Any]] = {}
        ordered_ids: list[str] = []

        for record in records:
            chat_id = record.get("chat_id")
            if not chat_id:
                continue

            created_at = record.get("created_at")
            session = sessions.get(chat_id)
            if session is None:
                session = {
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "message_count": 0,
                    "last_message": None,
                    "last_user_input": None,
                    "last_model_key": None,
                    "last_interaction_type": None,
                    "last_interaction_at": created_at,
                }
                sessions[chat_id] = session
                ordered_ids.append(chat_id)

            session["message_count"] += 1

            if created_at and (
                session["last_interaction_at"] is None
                or created_at >= session["last_interaction_at"]
            ):
                session["last_interaction_at"] = created_at
                session["last_message"] = record.get("response")
                session["last_user_input"] = record.get("user_input")
                session["last_model_key"] = record.get("model_key")
                session["last_interaction_type"] = record.get("interaction_type")

        return [sessions[chat_id] for chat_id in ordered_ids]

    def delete_chat_session(self, *, chat_id: str, user_id: str) -> int:
        if not self.enabled:
            raise RuntimeError("Supabase chat history client is not configured")

        try:
            response = (
                self._client.table(self.table_name)
                .delete()
                .eq("chat_id", chat_id)
                .eq("user_id", user_id)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to delete chat session: %s", exc)
            raise

        deleted = getattr(response, "count", None)
        if isinstance(deleted, int):
            return deleted

        # If Supabase client doesn't return a count, fall back to length of data
        data = getattr(response, "data", None)
        if isinstance(data, list):
            return len(data)

        return 0
