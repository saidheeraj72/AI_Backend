from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from supabase import Client, create_client

from src.core.config import Settings


ACCESS_RANK = {
    "view": 0,
    "comment": 1,
    "edit": 2,
    "admin": 3,
}


@dataclass
class DocumentPermission:
    document_id: str
    access_level: str

    @property
    def can_view(self) -> bool:
        return ACCESS_RANK.get(self.access_level, 0) >= ACCESS_RANK["view"]

    @property
    def can_upload(self) -> bool:
        return ACCESS_RANK.get(self.access_level, 0) >= ACCESS_RANK["edit"]

    @property
    def can_delete(self) -> bool:
        return ACCESS_RANK.get(self.access_level, 0) >= ACCESS_RANK["admin"]


class DocumentPermissionsService:
    """Service for managing the document_permissions table."""

    def __init__(self, settings: Settings, *, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self._settings = settings
        self._client: Optional[Client] = None

        if settings.supabase_url and settings.supabase_key:
            try:
                self._client = create_client(settings.supabase_url, settings.supabase_key)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to initialise Supabase client for document permissions: %s", exc)
        else:
            self.logger.warning("Supabase credentials missing; document permissions disabled")

    def _require_client(self) -> Client:
        if self._client is None:
            raise RuntimeError("Supabase client is not configured")
        return self._client

    def _fetch_permissions(self, *, user_id: str | None = None, group_ids: Sequence[str] | None = None, role_ids: Sequence[str] | None = None) -> list[dict[str, object]]:
        client = self._require_client()
        conditions = []
        if user_id:
            conditions.append(f"user_id.eq.{user_id}")
        if group_ids:
            groups_clause = ",".join(group_ids)
            conditions.append(f"group_id.in.({groups_clause})")
        if role_ids:
            roles_clause = ",".join(role_ids)
            conditions.append(f"role_id.in.({roles_clause})")

        if not conditions:
            return []

        filter_string = "or(" + ",".join(conditions) + ")"

        try:
            response = (
                client.table("document_permissions")
                .select("document_id, user_id, group_id, role_id, branch_id, access_level")
                .or_(filter_string)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to fetch document permissions: %s", exc)
            return []

        return response.data or []

    def get_permissions_for_user_context(
        self,
        *,
        user_id: str,
        group_ids: Sequence[str],
        role_ids: Sequence[str],
    ) -> Dict[str, DocumentPermission]:
        records = self._fetch_permissions(user_id=user_id, group_ids=group_ids, role_ids=role_ids)
        aggregated: Dict[str, DocumentPermission] = {}
        for record in records:
            document_id = record.get("document_id")
            access_level = str(record.get("access_level") or "view")
            if not document_id:
                continue
            existing = aggregated.get(document_id)
            if existing is None or ACCESS_RANK.get(access_level, 0) > ACCESS_RANK.get(existing.access_level, 0):
                aggregated[document_id] = DocumentPermission(document_id=document_id, access_level=access_level)
        return aggregated

    def get_all_user_document_permissions(self, user_id: str) -> list[DocumentPermission]:
        records = self._fetch_permissions(user_id=user_id)
        return [
            DocumentPermission(document_id=str(record.get("document_id")), access_level=str(record.get("access_level") or "view"))
            for record in records
            if record.get("document_id")
        ]

    def revoke_all_for_user(self, user_id: str) -> None:
        client = self._require_client()
        try:
            client.table("document_permissions").delete().eq("user_id", user_id).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError("Failed to clear document permissions for user") from exc

    def grant_document_permission(
        self,
        *,
        document_id: str,
        access_level: str,
        user_id: str | None = None,
        group_id: str | None = None,
        role_id: str | None = None,
        branch_id: str | None = None,
    ) -> None:
        client = self._require_client()
        targets = [value for value in [user_id, group_id, role_id, branch_id] if value]
        if not targets:
            raise RuntimeError("At least one permission target must be provided")

        payload = {
            "document_id": document_id,
            "access_level": access_level,
        }
        if user_id:
            payload["user_id"] = user_id
        if group_id:
            payload["group_id"] = group_id
        if role_id:
            payload["role_id"] = role_id
        if branch_id:
            payload["branch_id"] = branch_id

        try:
            client.table("document_permissions").insert(payload).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError("Failed to grant document permission") from exc

    def revoke_document_permission(
        self,
        *,
        document_id: str,
        user_id: str | None = None,
        group_id: str | None = None,
        role_id: str | None = None,
        branch_id: str | None = None,
    ) -> None:
        client = self._require_client()
        query = client.table("document_permissions").delete().eq("document_id", document_id)
        if user_id:
            query = query.eq("user_id", user_id)
        if group_id:
            query = query.eq("group_id", group_id)
        if role_id:
            query = query.eq("role_id", role_id)
        if branch_id:
            query = query.eq("branch_id", branch_id)

        try:
            query.execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError("Failed to revoke document permission") from exc
