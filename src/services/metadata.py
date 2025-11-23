from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from supabase import Client, create_client


class MetadataService:
    """Persist metadata about stored documents and their directories."""

    def __init__(
        self,
        metadata_path: Path,
        *,
        logger: Optional[logging.Logger] = None,
        supabase_url: str = "",
        supabase_key: str = "",
        supabase_table: str = "",
        default_org_id: str | None = None,
    ) -> None:
        self.metadata_path = metadata_path
        self.logger = logger or logging.getLogger(__name__)
        self._supabase_client: Optional[Client] = None
        self._supabase_table = supabase_table
        self._default_org_id = (default_org_id or "").strip() or None

        if supabase_url and supabase_key and supabase_table:
            try:
                self._supabase_client = create_client(supabase_url, supabase_key)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to initialize Supabase metadata client: %s", exc)

        if not self._supabase_enabled:
            self._ensure_file()

    def _ensure_file(self) -> None:
        if self._supabase_enabled:
            return
        if not self.metadata_path.exists():
            default_payload = {"documents": {}, "directories": {".": {"files": [], "subdirectories": []}}}
            self.metadata_path.write_text(json.dumps(default_payload, indent=2), encoding="utf-8")
            self.logger.info("Created metadata store at %s", self.metadata_path)

    def _load(self) -> Dict[str, Any]:
        if self._supabase_enabled:
            return {"documents": {}, "directories": {".": {"files": [], "subdirectories": []}}}
        try:
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            self.logger.warning("Metadata file %s is not valid JSON; resetting", self.metadata_path)
            return {"documents": {}, "directories": {".": {"files": [], "subdirectories": []}}}
        except FileNotFoundError:  # pragma: no cover - defensive
            self.logger.warning("Metadata file %s missing; recreating", self.metadata_path)
            return {"documents": {}, "directories": {".": {"files": [], "subdirectories": []}}}

    def _save(self, payload: Dict[str, Any]) -> None:
        if self._supabase_enabled:
            return
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _require_supabase_org_id(self, override: Optional[str] = None) -> str:
        org_id = override or self._default_org_id
        if not org_id:
            raise RuntimeError("DEFAULT_ORGANIZATION_ID must be configured when Supabase metadata is enabled")
        return org_id

    def record_document(
        self,
        relative_path: Path,
        chunks_indexed: int,
        branch_id: Optional[str] = None,
        branch_name: Optional[str] = None,
        *,
        org_id: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> None:
        if self._supabase_enabled:
            relative_key = relative_path.as_posix()
            directory_key = relative_path.parent.as_posix()
            resolved_org_id = self._require_supabase_org_id(org_id)
            payload = {
                "org_id": resolved_org_id,
                "title": relative_path.name,
                "storage_path": relative_key,
                "folder_path": directory_key if directory_key != "." else "",
                "metadata": {
                    "chunks_indexed": chunks_indexed,
                    "branch_name": branch_name,
                },
            }

            # Add branch information if provided
            if branch_id is not None:
                payload["branch_id"] = branch_id
            if created_by is not None:
                payload["created_by"] = created_by

            try:
                assert self._supabase_client is not None
                self._supabase_client.table(self._supabase_table).upsert(
                    payload,
                    on_conflict="org_id,storage_path",
                ).execute()
                self.logger.info("Recorded metadata for %s in Supabase", relative_key)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error(
                    "Failed to upsert Supabase metadata for %s: %s", relative_key, exc
                )
            return

        data = self._load()
        documents = data.setdefault("documents", {})
        directories = data.setdefault("directories", {".": {"files": [], "subdirectories": []}})

        relative_path = relative_path.as_posix()
        directory_path = Path(relative_path).parent
        directory_key = directory_path.as_posix() if directory_path.as_posix() != "." else "."

        documents[relative_path] = {
            "filename": Path(relative_path).name,
            "directory": directory_key if directory_key != "." else "",
            "chunks_indexed": chunks_indexed,
        }

        # Ensure directory hierarchy is present and populated
        parts = [part for part in Path(relative_path).parent.parts]
        for index in range(len(parts) + 1):
            segment_parts = parts[:index]
            segment_path = Path(*segment_parts) if segment_parts else Path(".")
            segment_key = segment_path.as_posix() if segment_path.as_posix() != "." else "."
            segment_entry = directories.setdefault(
                segment_key, {"files": [], "subdirectories": []}
            )

            if index < len(parts):
                child_parts = parts[: index + 1]
                child_path = Path(*child_parts) if child_parts else Path(".")
                child_key = child_path.as_posix() if child_path.as_posix() != "." else "."
                if child_key != segment_key and child_key not in segment_entry["subdirectories"]:
                    segment_entry["subdirectories"].append(child_key)
            else:
                if relative_path not in segment_entry["files"]:
                    segment_entry["files"].append(relative_path)

        self._save(data)
        self.logger.info("Recorded metadata for %s", relative_path)

    def list_documents(self, *, org_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if self._supabase_enabled:
            try:
                assert self._supabase_client is not None
                resolved_org = self._require_supabase_org_id(org_id)
                response = (
                    self._supabase_client.table(self._supabase_table)
                    .select("id, title, storage_path, folder_path, branch_id, metadata")
                    .eq("org_id", resolved_org)
                    .order("storage_path")
                    .execute()
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to list documents from Supabase: %s", exc)
                return []

            records = response.data or []
            items: List[Dict[str, Any]] = []
            for record in records:
                storage_path = record.get("storage_path")
                if not storage_path:
                    continue
                metadata = record.get("metadata") or {}
                chunks_indexed = metadata.get("chunks_indexed") if isinstance(metadata, dict) else None
                branch_name = metadata.get("branch_name") if isinstance(metadata, dict) else None
                items.append(
                    {
                        "id": record.get("id"),
                        "filename": record.get("title") or Path(storage_path).name,
                        "relative_path": storage_path,
                        "directory": record.get("folder_path") or "",
                        "chunks_indexed": int(chunks_indexed or 0),
                        "branch_id": record.get("branch_id"),
                        "branch_name": branch_name,
                    }
                )

            items.sort(key=lambda entry: entry["relative_path"].lower())
            return items

        data = self._load()
        documents = data.get("documents", {})
        items: List[Dict[str, Any]] = []

        for relative_path, info in documents.items():
            directory = info.get("directory", "")
            items.append(
                {
                    "id": relative_path,
                    "filename": info.get("filename", Path(relative_path).name),
                    "relative_path": relative_path,
                    "directory": directory,
                    "chunks_indexed": info.get("chunks_indexed", 0),
                }
            )

        items.sort(key=lambda entry: entry["relative_path"].lower())
        return items

    def get_directory_tree(self, *, org_id: Optional[str] = None) -> Dict[str, Any]:
        if self._supabase_enabled:
            return self._build_directory_tree_from_records(self.list_documents(org_id=org_id))
        data = self._load()
        return data.get("directories", {})

    def get_document(self, relative_path: str, *, org_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self._supabase_enabled:
            try:
                assert self._supabase_client is not None
                resolved_org = self._require_supabase_org_id(org_id)
                response = (
                    self._supabase_client.table(self._supabase_table)
                    .select("id, title, storage_path, folder_path, branch_id, metadata")
                    .eq("org_id", resolved_org)
                    .eq("storage_path", relative_path)
                    .limit(1)
                    .execute()
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to fetch metadata for %s from Supabase: %s", relative_path, exc)
                return None

            records = response.data or []
            if not records:
                return None

            record = records[0]
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            return {
                "id": record.get("id"),
                "relative_path": record.get("storage_path", relative_path),
                "filename": record.get("title", Path(relative_path).name),
                "directory": record.get("folder_path", ""),
                "chunks_indexed": int((metadata or {}).get("chunks_indexed", 0)),
                "branch_id": record.get("branch_id"),
                "branch_name": (metadata or {}).get("branch_name"),
            }

        data = self._load()
        document = data.get("documents", {}).get(relative_path)
        if not document:
            return None

        return {
            "id": relative_path,
            "relative_path": relative_path,
            **document,
        }

    def get_document_by_id(self, document_id: str, *, org_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self._supabase_enabled:
            try:
                assert self._supabase_client is not None
                resolved_org = self._require_supabase_org_id(org_id)
                response = (
                    self._supabase_client.table(self._supabase_table)
                    .select("id, title, storage_path, folder_path, branch_id, metadata")
                    .eq("org_id", resolved_org)
                    .eq("id", document_id)
                    .limit(1)
                    .execute()
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to fetch metadata for document %s: %s", document_id, exc)
                return None

            records = response.data or []
            if not records:
                return None

            record = records[0]
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            return {
                "id": record.get("id"),
                "relative_path": record.get("storage_path"),
                "filename": record.get("title"),
                "directory": record.get("folder_path", ""),
                "branch_id": record.get("branch_id"),
                "branch_name": metadata.get("branch_name") if isinstance(metadata, dict) else None,
                "chunks_indexed": int((metadata or {}).get("chunks_indexed", 0)),
            }

        # File-based metadata: treat the identifier as the relative path key
        return self.get_document(document_id, org_id=org_id)

    def delete_document(self, relative_path: str, *, org_id: Optional[str] = None) -> bool:
        if self._supabase_enabled:
            try:
                assert self._supabase_client is not None
                resolved_org = self._require_supabase_org_id(org_id)
                response = (
                    self._supabase_client.table(self._supabase_table)
                    .delete()
                    .eq("org_id", resolved_org)
                    .eq("storage_path", relative_path)
                    .execute()
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to delete metadata for %s from Supabase: %s", relative_path, exc)
                return False

            deleted = getattr(response, "count", None)
            if isinstance(deleted, int):
                result = deleted > 0
            else:
                data = getattr(response, "data", None)
                result = bool(data)

            if result:
                self.logger.info("Deleted metadata entry for %s from Supabase", relative_path)
            return result

        data = self._load()
        documents = data.setdefault("documents", {})
        if relative_path not in documents:
            return False

        documents.pop(relative_path, None)
        directories = data.setdefault("directories", {".": {"files": [], "subdirectories": []}})

        for entry in directories.values():
            files = entry.get("files", [])
            if relative_path in files:
                entry["files"] = [item for item in files if item != relative_path]

        self._save(data)
        self.logger.info("Deleted metadata entry for %s", relative_path)
        return True

    @property
    def _supabase_enabled(self) -> bool:
        return self._supabase_client is not None and bool(self._supabase_table)

    def _build_directory_tree_from_records(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        directories: Dict[str, Dict[str, List[str]]] = {".": {"files": [], "subdirectories": []}}

        for record in records:
            relative_path = record.get("relative_path") or record.get("storage_path")
            if not relative_path:
                continue

            posix_path = Path(relative_path)
            parts = list(posix_path.parent.parts)

            for index in range(len(parts) + 1):
                segment_parts = parts[:index]
                segment_path = Path(*segment_parts) if segment_parts else Path(".")
                segment_key = segment_path.as_posix() if segment_path.as_posix() != "." else "."
                segment_entry = directories.setdefault(
                    segment_key,
                    {"files": [], "subdirectories": []},
                )

                if index < len(parts):
                    child_parts = parts[: index + 1]
                    child_path = Path(*child_parts) if child_parts else Path(".")
                    child_key = child_path.as_posix() if child_path.as_posix() != "." else "."
                    if child_key != segment_key and child_key not in segment_entry["subdirectories"]:
                        segment_entry["subdirectories"].append(child_key)
                else:
                    if relative_path not in segment_entry["files"]:
                        segment_entry["files"].append(relative_path)

        return directories
