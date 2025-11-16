from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from supabase import Client, create_client

from src.core.config import Settings


@dataclass
class DocumentPermission:
    """Document-level permission for a specific user."""

    user_id: str
    document_path: str
    can_upload: bool
    can_view: bool
    can_delete: bool


class DocumentPermissionsService:
    """Service for managing per-document permissions."""

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

    def get_user_document_permission(
        self,
        user_id: str,
        document_path: str,
    ) -> Optional[DocumentPermission]:
        """Get permission for a specific user and document."""
        try:
            client = self._require_client()
        except RuntimeError:
            # If Supabase is not configured, grant full access (for local development)
            return DocumentPermission(
                user_id=user_id,
                document_path=document_path,
                can_upload=True,
                can_view=True,
                can_delete=True,
            )

        try:
            response = (
                client.table("document_permissions")
                .select("can_upload, can_view, can_delete")
                .eq("user_id", user_id)
                .eq("document_path", document_path)
                .limit(1)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "Failed to fetch document permission for user %s, document %s: %s",
                user_id,
                document_path,
                exc,
            )
            return None

        records = response.data or []
        if not records:
            return None

        record = records[0]
        return DocumentPermission(
            user_id=user_id,
            document_path=document_path,
            can_upload=bool(record.get("can_upload", False)),
            can_view=bool(record.get("can_view", False)),
            can_delete=bool(record.get("can_delete", False)),
        )

    def get_all_user_document_permissions(self, user_id: str) -> list[DocumentPermission]:
        """Get all document permissions for a specific user."""
        try:
            client = self._require_client()
        except RuntimeError:
            return []

        try:
            response = (
                client.table("document_permissions")
                .select("document_path, can_upload, can_view, can_delete")
                .eq("user_id", user_id)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to fetch document permissions for user %s: %s", user_id, exc)
            return []

        permissions = []
        for record in response.data or []:
            permissions.append(
                DocumentPermission(
                    user_id=user_id,
                    document_path=str(record.get("document_path", "")),
                    can_upload=bool(record.get("can_upload", False)),
                    can_view=bool(record.get("can_view", False)),
                    can_delete=bool(record.get("can_delete", False)),
                )
            )
        return permissions

    def user_can_view_document(self, user_id: str, document_path: str, is_superadmin: bool = False) -> bool:
        """Check if user has view permission for a specific document."""
        if is_superadmin:
            return True

        # Get all folder permissions for this user
        permissions = self.get_all_user_document_permissions(user_id)

        self.logger.info(f"[user_can_view_document] Checking document: {document_path}")
        self.logger.info(f"[user_can_view_document] User {user_id} has {len(permissions)} permissions")
        for perm in permissions:
            self.logger.info(f"  - Path: {perm.document_path}, can_view: {perm.can_view}")

        # Extract the folder path from the document path
        if "/" in document_path:
            folder_path = document_path.rsplit("/", 1)[0]
        else:
            folder_path = ""

        self.logger.info(f"[user_can_view_document] Extracted folder: '{folder_path}'")

        # Check if any permission matches the folder or parent folders
        for perm in permissions:
            if perm.can_view:
                self.logger.info(f"[user_can_view_document] Checking permission path '{perm.document_path}' against folder '{folder_path}'")

                # Check if document is in this folder or subfolder
                if folder_path == perm.document_path or folder_path.startswith(perm.document_path + "/"):
                    self.logger.info(f"[user_can_view_document] ✓ MATCH - Folder match!")
                    return True
                # Also check exact document path match
                if document_path == perm.document_path:
                    self.logger.info(f"[user_can_view_document] ✓ MATCH - Exact document match!")
                    return True

        self.logger.info(f"[user_can_view_document] ✗ NO MATCH - Access denied")
        return False

    def user_can_upload_document(self, user_id: str, document_path: str, is_superadmin: bool = False) -> bool:
        """Check if user has upload permission for a specific document."""
        if is_superadmin:
            return True

        # Get all folder permissions for this user
        permissions = self.get_all_user_document_permissions(user_id)

        # Extract the folder path from the document path
        if "/" in document_path:
            folder_path = document_path.rsplit("/", 1)[0]
        else:
            folder_path = ""

        # Check if any permission matches the folder or parent folders
        for perm in permissions:
            if perm.can_upload:
                # Check if document is in this folder or subfolder
                if folder_path == perm.document_path or folder_path.startswith(perm.document_path + "/"):
                    return True
                # Also check exact document path match
                if document_path == perm.document_path:
                    return True

        return False

    def user_can_delete_document(self, user_id: str, document_path: str, is_superadmin: bool = False) -> bool:
        """Check if user has delete permission for a specific document."""
        if is_superadmin:
            return True

        # Get all folder permissions for this user
        permissions = self.get_all_user_document_permissions(user_id)

        # Extract the folder path from the document path
        if "/" in document_path:
            folder_path = document_path.rsplit("/", 1)[0]
        else:
            folder_path = ""

        # Check if any permission matches the folder or parent folders
        for perm in permissions:
            if perm.can_delete:
                # Check if document is in this folder or subfolder
                if folder_path == perm.document_path or folder_path.startswith(perm.document_path + "/"):
                    return True
                # Also check exact document path match
                if document_path == perm.document_path:
                    return True

        return False

    def grant_document_permission(
        self,
        user_id: str,
        document_path: str,
        can_upload: bool = False,
        can_view: bool = False,
        can_delete: bool = False,
        granted_by: Optional[str] = None,
    ) -> None:
        """Grant or update document permission for a user."""
        client = self._require_client()

        payload = {
            "user_id": user_id,
            "document_path": document_path,
            "can_upload": can_upload,
            "can_view": can_view,
            "can_delete": can_delete,
        }
        if granted_by:
            payload["granted_by"] = granted_by

        try:
            client.table("document_permissions").upsert(
                payload,
                on_conflict="user_id,document_path",
            ).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "Failed to grant document permission for user %s, document %s: %s",
                user_id,
                document_path,
                exc,
            )
            raise RuntimeError("Failed to grant document permission") from exc

    def revoke_document_permission(self, user_id: str, document_path: str) -> None:
        """Revoke document permission for a user."""
        client = self._require_client()

        try:
            client.table("document_permissions").delete().eq("user_id", user_id).eq(
                "document_path", document_path
            ).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "Failed to revoke document permission for user %s, document %s: %s",
                user_id,
                document_path,
                exc,
            )
            raise RuntimeError("Failed to revoke document permission") from exc
