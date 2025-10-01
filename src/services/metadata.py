from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetadataService:
    """Persist metadata about stored documents and their directories."""

    def __init__(self, metadata_path: Path, *, logger: Optional[logging.Logger] = None) -> None:
        self.metadata_path = metadata_path
        self.logger = logger or logging.getLogger(__name__)
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.metadata_path.exists():
            default_payload = {"documents": {}, "directories": {".": {"files": [], "subdirectories": []}}}
            self.metadata_path.write_text(json.dumps(default_payload, indent=2), encoding="utf-8")
            self.logger.info("Created metadata store at %s", self.metadata_path)

    def _load(self) -> Dict[str, Any]:
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
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def record_document(self, relative_path: Path, chunks_indexed: int) -> None:
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

    def list_documents(self) -> List[Dict[str, Any]]:
        data = self._load()
        documents = data.get("documents", {})
        items: List[Dict[str, Any]] = []

        for relative_path, info in documents.items():
            directory = info.get("directory", "")
            items.append(
                {
                    "filename": info.get("filename", Path(relative_path).name),
                    "relative_path": relative_path,
                    "directory": directory,
                    "chunks_indexed": info.get("chunks_indexed", 0),
                }
            )

        items.sort(key=lambda entry: entry["relative_path"].lower())
        return items

    def get_directory_tree(self) -> Dict[str, Any]:
        data = self._load()
        return data.get("directories", {})
