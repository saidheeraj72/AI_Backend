from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from supabase import Client, create_client


class MetadataService:
    """
    Manages documents and folders in Supabase using the new schema.
    """

    def __init__(
        self,
        metadata_path: Path,
        *,
        logger: Optional[logging.Logger] = None,
        supabase_url: str = "",
        supabase_key: str = "",
        supabase_table: str = "documents", # Default to documents, but we also use folders
        default_org_id: str | None = None,
    ) -> None:
        self.metadata_path = metadata_path
        self.logger = logger or logging.getLogger(__name__)
        self._supabase_client: Optional[Client] = None
        self._supabase_table = supabase_table
        
        # We might need default branch_id instead of org_id now, but let's keep this for fallback or config
        self._default_org_id = (default_org_id or "").strip() or None

        if supabase_url and supabase_key:
            try:
                self._supabase_client = create_client(supabase_url, supabase_key)
            except Exception as exc:
                self.logger.error("Failed to initialize Supabase client: %s", exc)

    def _require_client(self) -> Client:
        if self._supabase_client is None:
            raise RuntimeError("Supabase client not configured")
        return self._supabase_client

    def _get_folder_id_by_path(self, branch_id: str, path: str, create: bool = False, created_by: Optional[str] = None) -> Optional[str]:
        """
        Resolves a folder path (e.g., "finance/reports") to a folder_id for a given branch.
        If create is True, creates missing folders.
        """
        client = self._require_client()
        
        clean_path = path.strip("/").split("/")
        if not clean_path or clean_path == [""]:
            return None # Root

        current_parent_id = None
        
        for folder_name in clean_path:
            if not folder_name:
                continue
            
            query = client.table("folders").select("id").eq("branch_id", branch_id).eq("name", folder_name)
            if current_parent_id:
                query = query.eq("parent_id", current_parent_id)
            else:
                query = query.is_("parent_id", "null")
            
            response = query.maybe_single().execute()
            
            if response and response.data:
                current_parent_id = response.data["id"]
            elif create:
                payload = {
                    "branch_id": branch_id,
                    "name": folder_name,
                    "parent_id": current_parent_id,
                    "created_by": created_by
                }
                new_folder = client.table("folders").insert(payload).execute()
                if new_folder.data:
                    current_parent_id = new_folder.data[0]["id"]
                else:
                    return None # Failed to create
            else:
                return None
                
        return current_parent_id

    def record_document(
        self,
        relative_path: Path,
        chunks_indexed: int,
        branch_id: Optional[str] = None,
        branch_name: Optional[str] = None,
        *,
        org_id: Optional[str] = None,
        created_by: Optional[str] = None,
        description: Optional[str] = None,
        status: str = "pending_review",
    ) -> None:
        if not self._supabase_client:
             self.logger.warning("Supabase not configured, skipping DB record")
             return

        if not branch_id:
            self.logger.error("branch_id is required for recording documents in new schema")
            return

        title = relative_path.name
        storage_path = relative_path.as_posix()
        directory_path = relative_path.parent.as_posix()
        if directory_path == ".":
            directory_path = ""

        # Resolve folder_id
        folder_id = self._get_folder_id_by_path(branch_id, directory_path, create=True, created_by=created_by)

        payload = {
            "branch_id": branch_id,
            "folder_id": folder_id,
            "owner_id": created_by,
            "title": title,
            "storage_path": storage_path,
            "description": description,
            "status": status,
            "metadata": {
                "chunks_indexed": chunks_indexed,
                "branch_name": branch_name
            }
        }

        try:
            # We use storage_path as unique constraint or just insert
            # The schema doesn't strictly enforce unique storage_path per branch but it's good practice.
            # Let's check if it exists first to update or insert (Upsert)
            # We need a unique key for upsert. The schema allows duplicate storage paths technically unless unique index added.
            # But let's assume we want to update if exists.
            
            # Find existing by branch_id and storage_path
            existing = self._supabase_client.table("documents").select("id").eq("branch_id", branch_id).eq("storage_path", storage_path).maybe_single().execute()
            
            if existing and existing.data:
                self._supabase_client.table("documents").update(payload).eq("id", existing.data["id"]).execute()
            else:
                self._supabase_client.table("documents").insert(payload).execute()
                
            self.logger.info("Recorded metadata for %s in Supabase", storage_path)
        except Exception as exc:
            self.logger.error("Failed to upsert Supabase metadata for %s: %s", storage_path, exc)

    def list_documents(self, *, org_id: Optional[str] = None, branch_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._supabase_client:
            return []
            
        try:
            # Join with profiles to get owner email and reviewer email using aliases
            # Note: We assume 'owner_id' and 'reviewed_by' are the foreign keys to 'profiles'
            query = self._supabase_client.table("documents").select(
                "id, title, storage_path, branch_id, description, status, metadata, folders(name), "
                "owner:profiles!owner_id(email), "
                "reviewer:profiles!reviewed_by(email), "
                "reviewed_at"
            )
            if branch_id:
                query = query.eq("branch_id", branch_id)
            if status:
                query = query.eq("status", status)
            
            response = query.order("storage_path").execute()
            
            records = response.data or []
            items: List[Dict[str, Any]] = []
            for record in records:
                storage_path = record.get("storage_path")
                metadata = record.get("metadata") or {}
                owner_profile = record.get("owner") or {}
                reviewer_profile = record.get("reviewer") or {}
                
                items.append({
                    "id": record.get("id"),
                    "filename": record.get("title"),
                    "relative_path": storage_path,
                    "directory": record.get("folders", {}).get("name") if record.get("folders") else "",
                    "chunks_indexed": int((metadata or {}).get("chunks_indexed", 0)),
                    "branch_id": record.get("branch_id"),
                    "branch_name": (metadata or {}).get("branch_name"),
                    "description": record.get("description"),
                    "status": record.get("status", "pending_review"),
                    "owner_email": owner_profile.get("email") if owner_profile else None,
                    "reviewer_email": reviewer_profile.get("email") if reviewer_profile else None,
                    "reviewed_at": record.get("reviewed_at"),
                })
            return items
        except Exception as exc:
            self.logger.error("Failed to list documents: %s", exc)
            return []

    def update_document_status(self, document_id: str, status: str, reviewed_by: Optional[str] = None) -> bool:
        if not self._supabase_client:
            return False
        try:
            payload = {"status": status}
            if reviewed_by:
                payload["reviewed_by"] = reviewed_by
                payload["reviewed_at"] = "now()" # Let Postgres handle timestamp or import datetime
            
            self._supabase_client.table("documents").update(payload).eq("id", document_id).execute()
            return True
        except Exception as exc:
            self.logger.error("Failed to update document status: %s", exc)
            return False

    def get_document_by_id(self, document_id: str, *, org_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not self._supabase_client:
            return None
        try:
            response = self._supabase_client.table("documents").select("*").eq("id", document_id).single().execute()
            if response.data:
                record = response.data
                metadata = record.get("metadata") or {}
                return {
                    "id": record.get("id"),
                    "relative_path": record.get("storage_path"),
                    "filename": record.get("title"),
                    "branch_id": record.get("branch_id"),
                    "chunks_indexed": int((metadata or {}).get("chunks_indexed", 0)),
                }
        except Exception as exc:
            self.logger.error(f"Error fetching document {document_id}: {exc}")
        return None

    def delete_document(self, relative_path: str, *, org_id: Optional[str] = None) -> bool:
        if not self._supabase_client:
            return False
        try:
            # Again, risky without branch_id. RLS should protect cross-tenant deletes.
            self._supabase_client.table("documents").delete().eq("storage_path", relative_path).execute()
            return True
        except Exception as exc:
            self.logger.error("Failed to delete document: %s", exc)
            return False

    def create_folder(self, branch_id: str, name: str, parent_id: Optional[str] = None, description: Optional[str] = None, created_by: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not self._supabase_client:
            return None
        
        try:
            payload = {
                "branch_id": branch_id,
                "name": name,
                "parent_id": parent_id,
                "description": description,
                "created_by": created_by
            }
            
            response = self._supabase_client.table("folders").insert(payload).execute()
            if response.data:
                return response.data[0]
        except Exception as exc:
            self.logger.error("Failed to create folder: %s", exc)
        return None