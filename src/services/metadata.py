from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from supabase import Client, create_client


class MetadataService:
    """
    Manages documents and folders in Supabase using the new schema (Many-to-Many).
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

    def _get_folder_id_by_path(self, branch_ids: List[str], path: str, create: bool = False, created_by: Optional[str] = None) -> Optional[str]:
        """
        Resolves a folder path (e.g., "finance/reports") to a SINGLE folder_id shared by the given branches.
        Strategy:
        1. For each level, look for an existing folder (name, parent) linked to ANY of the branch_ids.
        2. If found, use it and ensure it's linked to ALL branch_ids.
        3. If not found, create it and link to ALL branch_ids.
        """
        client = self._require_client()
        
        clean_path = path.strip("/").split("/")
        if not clean_path or clean_path == [""]:
            return None # Root

        current_parent_id = None
        
        for folder_name in clean_path:
            if not folder_name:
                continue
            
            # 1. Try to find existing folder in ANY of the branches
            # We need to query folders that match name, parent, AND exist in folder_branches for these branches.
            # This is complex in one query. Let's try to find by name/parent first, then filter?
            # Or better: Find ANY folder with this name/parent that is linked to one of our branches.
            
            # Step A: Get candidates by name/parent
            query = client.table("folders").select("id, folder_branches!inner(branch_id)").eq("name", folder_name)
            if current_parent_id:
                query = query.eq("parent_id", current_parent_id)
            else:
                query = query.is_("parent_id", "null")
            
            # Filter by branches in the inner join if possible, or post-filter
            # supabase-py 'in' filter on foreign table might be tricky.
            # Let's fetch candidates and pick one that matches our branches.
            response = query.execute()
            candidates = response.data or []
            
            chosen_folder_id = None
            
            # Filter candidates to see if any belong to our requested branch_ids
            for cand in candidates:
                # cand['folder_branches'] is a list of dicts due to !inner
                linked_branches = {b['branch_id'] for b in cand.get('folder_branches', [])}
                if any(bid in linked_branches for bid in branch_ids):
                    chosen_folder_id = cand['id']
                    break
            
            # Step B: If no suitable folder found, and create=True, make one.
            if not chosen_folder_id:
                if create:
                    payload = {
                        "name": folder_name,
                        "parent_id": current_parent_id,
                        "created_by": created_by,
                        "description": "Created via upload"
                    }
                    new_folder = client.table("folders").insert(payload).execute()
                    if new_folder.data:
                        chosen_folder_id = new_folder.data[0]["id"]
                    else:
                        return None
                else:
                    return None
            
            # Step C: Ensure chosen_folder_id is linked to ALL requested branch_ids
            if chosen_folder_id:
                links_to_create = []
                for bid in branch_ids:
                    links_to_create.append({"folder_id": chosen_folder_id, "branch_id": bid})
                
                if links_to_create:
                    # Upserting into folder_branches (ignoring duplicates)
                    # Supabase upsert requires primary key or unique constraint. 
                    # We have unique(folder_id, branch_id).
                    client.table("folder_branches").upsert(links_to_create, on_conflict="folder_id, branch_id", ignore_duplicates=True).execute()
            
            current_parent_id = chosen_folder_id
                
        return current_parent_id

    def record_document(
        self,
        relative_path: Path,
        chunks_indexed: int,
        branch_ids: List[str],
        branch_name: Optional[str] = None, # Legacy/informative
        *,
        org_id: Optional[str] = None,
        created_by: Optional[str] = None,
        description: Optional[str] = None,
        # status param removed
    ) -> None:
        if not self._supabase_client:
             self.logger.warning("Supabase not configured, skipping DB record")
             return

        if not branch_ids:
            self.logger.error("branch_ids list is required for recording documents")
            return

        title = relative_path.name
        storage_path = relative_path.as_posix()
        directory_path = relative_path.parent.as_posix()
        if directory_path == ".":
            directory_path = ""

        # Resolve folder_id (ensuring it exists and is linked to all branches)
        folder_id = self._get_folder_id_by_path(branch_ids, directory_path, create=True, created_by=created_by)

        # 1. Upsert Document (Global)
        payload = {
            "folder_id": folder_id,
            "owner_id": created_by,
            "title": title,
            "storage_path": storage_path,
            "description": description,
            "metadata": {
                "chunks_indexed": chunks_indexed,
                "branch_names": branch_name # Just for ref
            }
        }

        try:
            # Upsert document by storage_path (assuming unique path per file system)
            # Use select first to get ID if exists
            existing = self._supabase_client.table("documents").select("id").eq("storage_path", storage_path).maybe_single().execute()
            
            doc_id = None
            if existing and existing.data:
                doc_id = existing.data["id"]
                self._supabase_client.table("documents").update(payload).eq("id", doc_id).execute()
            else:
                res = self._supabase_client.table("documents").insert(payload).execute()
                if res.data:
                    doc_id = res.data[0]["id"]

            if doc_id:
                # 2. Link to Branches
                links = [{"document_id": doc_id, "branch_id": bid} for bid in branch_ids]
                self._supabase_client.table("document_branches").upsert(links, on_conflict="document_id, branch_id", ignore_duplicates=True).execute()
                
                self.logger.info("Recorded metadata for %s in Supabase (Branches: %s)", storage_path, branch_ids)
        except Exception as exc:
            self.logger.error("Failed to upsert Supabase metadata for %s: %s", storage_path, exc)

    def list_documents(self, *, org_id: Optional[str] = None, branch_id: Optional[str] = None, status: Optional[str] = None, folder_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Lists documents.
        If branch_id is provided, filters by availability in that branch.
        """
        if not self._supabase_client:
            return []
            
        try:
            # Select documents, join with folder and owner
            # Also filter by document_branches if branch_id set
            # We fetch branch name via nested join: document_branches -> branches -> name
            query = self._supabase_client.table("documents").select(
                "id, title, storage_path, description, metadata, created_at, folder_id, folders(name), "
                "owner:profiles!owner_id(email), "
                "document_branches!inner(branches(name))" 
            )
            
            if branch_id:
                # Filter the INNER JOINED table via the junction table's branch_id column
                # Note: filtering on nested resource 'document_branches.branch_id' requires careful syntax if we didn't select it.
                # However, since we use !inner, we can filter on the foreign table.
                # But wait, we changed the select to fetch name. The join condition is implicit.
                # To filter by ID we might need to filter the relation.
                # The safest way with supabase-py is usually .eq("document_branches.branch_id", branch_id)
                # But we changed the select. Let's assume the filter still works on the relation or add branch_id to select.
                query = query.eq("document_branches.branch_id", branch_id)
            
            if folder_ids:
                query = query.in_("folder_id", folder_ids)
            
            response = query.order("storage_path").execute()
            
            records = response.data or []
            items: List[Dict[str, Any]] = []
            for record in records:
                storage_path = record.get("storage_path")
                metadata = record.get("metadata") or {}
                owner_profile = record.get("owner") or {}
                
                # Extract branch name
                # document_branches is a list of dicts: [{'branches': {'name': 'Branch A'}}, ...]
                doc_branches = record.get("document_branches") or []
                branch_name = None
                if doc_branches and isinstance(doc_branches, list):
                    # Just take the first one for now as per singular 'branch_name' request
                    first_branch = doc_branches[0]
                    if first_branch.get("branches"):
                        branch_name = first_branch.get("branches").get("name")
                
                items.append({
                    "id": record.get("id"),
                    "folder_id": record.get("folder_id"),
                    "filename": record.get("title"),
                    "relative_path": storage_path,
                    "directory": record.get("folders", {}).get("name") if record.get("folders") else "",
                    "chunks_indexed": int((metadata or {}).get("chunks_indexed", 0)),
                    "created_at": record.get("created_at"),
                    "branch_name": branch_name,
                    "description": record.get("description"),
                    "owner_email": owner_profile.get("email") if owner_profile else None,
                })
            return items
        except Exception as exc:
            self.logger.error("Failed to list documents: %s", exc)
            return []

    def list_folders(self, branch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lists folders.
        If branch_id is provided, filters by availability in that branch.
        """
        if not self._supabase_client:
            return []
            
        try:
            query = self._supabase_client.table("folders").select("id, parent_id, name, description, created_at, updated_at, folder_branches!inner(branch_id)")
            
            if branch_id:
                query = query.eq("folder_branches.branch_id", branch_id)
            
            response = query.order("name").execute()
            return response.data or []
        except Exception as exc:
            self.logger.error("Failed to list folders: %s", exc)
            return []

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
                    "chunks_indexed": int((metadata or {}).get("chunks_indexed", 0)),
                }
        except Exception as exc:
            self.logger.error(f"Error fetching document {document_id}: {exc}")
        return None

    def delete_document(self, relative_path: str, *, org_id: Optional[str] = None) -> bool:
        if not self._supabase_client:
            return False
        try:
            self._supabase_client.table("documents").delete().eq("storage_path", relative_path).execute()
            return True
        except Exception as exc:
            self.logger.error("Failed to delete document: %s", exc)
            return False

    def delete_folder(self, folder_id: str) -> bool:
        """
        Deletes a folder by ID. 
        Note: This does NOT recursively delete documents/subfolders if FK constraints don't cascade.
        It is expected that the caller cleans up contents first.
        """
        if not self._supabase_client:
            return False
        try:
            self._supabase_client.table("folders").delete().eq("id", folder_id).execute()
            return True
        except Exception as exc:
            self.logger.error("Failed to delete folder %s: %s", folder_id, exc)
            return False

    def create_folder(self, branch_ids: List[str], name: str, parent_id: Optional[str] = None, description: Optional[str] = None, created_by: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Explicitly creates a folder and links it to the provided branch_ids.
        """
        if not self._supabase_client:
            return None
        
        try:
            # 1. Create Folder (Global)
            payload = {
                "name": name,
                "parent_id": parent_id,
                "description": description,
                "created_by": created_by
            }
            
            response = self._supabase_client.table("folders").insert(payload).execute()
            if response.data:
                folder_data = response.data[0]
                folder_id = folder_data["id"]
                
                # 2. Link to Branches
                if branch_ids:
                    links = [{"folder_id": folder_id, "branch_id": bid} for bid in branch_ids]
                    self._supabase_client.table("folder_branches").upsert(links, on_conflict="folder_id, branch_id", ignore_duplicates=True).execute()
                
                return folder_data
        except Exception as exc:
            self.logger.error("Failed to create folder: %s", exc)
        return None