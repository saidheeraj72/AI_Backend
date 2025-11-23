from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

from supabase import Client, create_client

from src.core.config import Settings
from src.services.document_permissions import ACCESS_RANK, DocumentPermission
from src.utils.paths import normalize_relative_path


def normalize_folder_path(value: str | None) -> str:
    """Normalize folder paths into a consistent relative form (root becomes '')."""
    normalized = normalize_relative_path(value or "")
    if normalized == "/":
        return ""
    return normalized


def _document_in_folder(document_directory: str, folder_path: str) -> bool:
    if folder_path == "":
        return True
    return document_directory == folder_path or document_directory.startswith(f"{folder_path}/")


def expand_folder_permissions_to_documents(
    folder_permissions: Iterable["FolderPermission"],
    documents: Sequence[Dict[str, Any]],
) -> dict[str, DocumentPermission]:
    mapping: dict[str, DocumentPermission] = {}
    for entry in folder_permissions:
        for record in documents:
            document_directory = normalize_relative_path(str(record.get("directory") or ""))
            if not _document_in_folder(document_directory, entry.folder_path):
                continue

            document_id = str(record.get("id") or record.get("relative_path") or "")
            if not document_id:
                continue

            existing = mapping.get(document_id)
            if existing is None or ACCESS_RANK.get(entry.access_level, 0) > ACCESS_RANK.get(existing.access_level, 0):
                mapping[document_id] = DocumentPermission(document_id=document_id, access_level=entry.access_level)
    return mapping


@dataclass(frozen=True)
class FolderPermission:
    folder_path: str
    access_level: str
    branch_id: Optional[str]
    user_id: Optional[str]
    group_id: Optional[str]
    role_id: Optional[str]
    org_id: Optional[str]

    @property
    def can_view(self) -> bool:
        return ACCESS_RANK.get(self.access_level, 0) >= ACCESS_RANK["view"]

    @property
    def can_upload(self) -> bool:
        return ACCESS_RANK.get(self.access_level, 0) >= ACCESS_RANK["edit"]

    @property
    def can_delete(self) -> bool:
        return ACCESS_RANK.get(self.access_level, 0) >= ACCESS_RANK["admin"]


class FolderPermissionsService:
    """Manage folder ACLs and group-folder mappings in Supabase."""

    def __init__(self, settings: Settings, *, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self._settings = settings
        self._client: Optional[Client] = None

        if settings.supabase_url and settings.supabase_key:
            try:
                self._client = create_client(settings.supabase_url, settings.supabase_key)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to initialise Supabase client for folder permissions: %s", exc)
        else:
            self.logger.warning("Supabase credentials missing; folder permissions disabled")

    def _require_client(self) -> Client:
        if self._client is None:
            raise RuntimeError("Supabase client is not configured for folder permissions")
        return self._client

    def _fetch_permissions(
        self,
        *,
        user_id: str | None = None,
        group_ids: Sequence[str] | None = None,
        role_ids: Sequence[str] | None = None,
        org_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        client = self._require_client()
        conditions: list[str] = []
        if user_id:
            conditions.append(f"user_id.eq.{user_id}")
        if group_ids:
            sanitized_groups = [group_id for group_id in group_ids if group_id]
            if sanitized_groups:
                group_clause = ",".join(sanitized_groups)
                conditions.append(f"group_id.in.({group_clause})")
        if role_ids:
            role_clause = ",".join(role_ids)
            conditions.append(f"role_id.in.({role_clause})")

        if not conditions:
            return []

        filter_string = "or(" + ",".join(conditions) + ")"
        query = client.table("folder_permissions").select(
            "folder_path, access_level, branch_id, user_id, group_id, role_id, org_id"
        )
        if org_id:
            query = query.eq("org_id", org_id)

        try:
            response = query.or_(filter_string).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self._handle_missing_table(exc, "folder_permissions")
            self.logger.error("Failed to fetch folder permissions: %s", exc)
            return []

        return response.data or []

    def get_permissions_for_user_context(
        self,
        *,
        user_id: str,
        group_ids: Sequence[str],
        role_ids: Sequence[str],
        org_id: Optional[str] = None,
    ) -> Dict[str, FolderPermission]:
        records = self._fetch_permissions(user_id=user_id, group_ids=group_ids, role_ids=role_ids, org_id=org_id)
        result: Dict[str, FolderPermission] = {}
        for record in records:
            raw_path = record.get("folder_path")
            folder_path = normalize_folder_path(str(raw_path or ""))
            access_level = str(record.get("access_level") or "view")
            existing = result.get(folder_path)
            if existing and ACCESS_RANK.get(existing.access_level, 0) >= ACCESS_RANK.get(access_level, 0):
                continue
            result[folder_path] = FolderPermission(
                folder_path=folder_path,
                access_level=access_level,
                branch_id=str(record.get("branch_id")) if record.get("branch_id") else None,
                user_id=str(record.get("user_id")) if record.get("user_id") else None,
                group_id=str(record.get("group_id")) if record.get("group_id") else None,
                role_id=str(record.get("role_id")) if record.get("role_id") else None,
                org_id=str(record.get("org_id")) if record.get("org_id") else None,
            )

        if org_id:
            group_paths = self._fetch_group_folder_paths(group_ids=group_ids, org_id=org_id)
            for raw_path in group_paths:
                folder_path = normalize_folder_path(raw_path)
                existing = result.get(folder_path)
                if existing and ACCESS_RANK.get(existing.access_level, 0) >= ACCESS_RANK.get("view", 0):
                    continue
                result[folder_path] = FolderPermission(
                    folder_path=folder_path,
                    access_level="view",
                    branch_id=None,
                    user_id=None,
                    group_id=None,
                    role_id=None,
                    org_id=org_id,
                )
        return result

    def _fetch_group_folder_paths(self, *, group_ids: Sequence[str], org_id: str) -> list[str]:
        client = self._require_client()
        sanitized = [group_id for group_id in group_ids if group_id]
        if not sanitized:
            return []

        try:
            response = (
                client.table("group_folder_permissions")
                .select("folder_path")
                .in_("group_id", sanitized)
                .eq("org_id", org_id)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self._handle_missing_table(exc, "group_folder_permissions")
            self.logger.error("Failed to fetch group folder paths: %s", exc)
            return []

        return [record.get("folder_path") or "" for record in response.data or []]

    def _handle_missing_table(self, exc: Exception, table_name: str) -> None:
        details = None
        if hasattr(exc, "args") and exc.args:
            first = exc.args[0]
            if isinstance(first, dict):
                details = first
        if isinstance(details, dict) and details.get("code") == "42P01":
            self.logger.warning(
                "Database table %s is missing (code 42P01). Create it via migrations/001_create_folder_permissions.sql",
                table_name,
            )

    def grant_folder_permission(
        self,
        *,
        folder_path: str,
        access_level: str,
        user_id: str | None = None,
        group_id: str | None = None,
        role_id: str | None = None,
        branch_id: str | None = None,
        org_id: Optional[str] = None,
    ) -> None:
        client = self._require_client()
        payload: dict[str, Any] = {
            "folder_path": normalize_folder_path(folder_path),
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
        if org_id:
            payload["org_id"] = org_id

        try:
            client.table("folder_permissions").insert(payload).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to grant folder permission: %s", exc)
            raise RuntimeError("Failed to grant folder permission") from exc

    def revoke_all_for_user(self, user_id: str) -> None:
        client = self._require_client()
        try:
            client.table("folder_permissions").delete().eq("user_id", user_id).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError("Failed to clear folder permissions for user") from exc

    def list_group_folder_permissions(self, *, org_id: str) -> list[dict[str, Any]]:
        client = self._require_client()
        try:
            response = (
                client.table("group_folder_permissions")
                .select("id, group_id, folder_path")
                .eq("org_id", org_id)
                .order("folder_path")
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to fetch group folder permissions: %s", exc)
            return []
        return response.data or []

    def update_group_folder_permissions(self, *, group_id: str, folder_paths: Sequence[str], org_id: str) -> None:
        client = self._require_client()
        try:
            client.table("group_folder_permissions").delete().eq("group_id", group_id).eq("org_id", org_id).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to clear group folder permissions for %s: %s", group_id, exc)
            raise RuntimeError("Failed to update group folder permissions") from exc

        payload: list[dict[str, Any]] = []
        for folder_path in folder_paths:
            normalized = normalize_folder_path(folder_path)
            payload.append(
                {
                    "group_id": group_id,
                    "org_id": org_id,
                    "folder_path": normalized,
                }
            )

        if not payload:
            return

        try:
            client.table("group_folder_permissions").insert(payload).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to insert group folder permissions: %s", exc)
            raise RuntimeError("Failed to update group folder permissions") from exc
