from __future__ import annotations

import asyncio
import logging
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Optional

from fastapi import UploadFile
from supabase import Client, create_client


@dataclass(slots=True)
class StoredDocument:
    """Result of persisting a document for downstream processing."""

    relative_path: Path
    local_path: Path
    should_cleanup: bool


class StorageService:
    """Persist uploads to Supabase storage with optional local fallback."""

    def __init__(
        self,
        base_dir: Path,
        *,
        supabase_url: str = "",
        supabase_key: str = "",
        supabase_bucket: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.base_dir = base_dir
        self.logger = logger or logging.getLogger(__name__)
        self._supabase_client: Optional[Client] = None
        self._supabase_bucket = supabase_bucket or ""

        if supabase_url and supabase_key and self._supabase_bucket:
            try:
                self._supabase_client = create_client(supabase_url, supabase_key)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to initialize Supabase storage client: %s", exc)

    async def save_pdf(
        self, upload: UploadFile, *, relative_path: Optional[str] = None
    ) -> StoredDocument:
        """Persist an uploaded PDF to Supabase (and optionally disk for fallback)."""
        original_name = upload.filename or "document.pdf"
        target_relative = self._sanitize_relative_path(
            relative_path or original_name, fallback_name=original_name
        )
        content_type = upload.content_type or "application/pdf"

        if self._supabase_enabled:
            file_bytes = await upload.read()
            await upload.close()
            temp_path = self._write_temp_file(file_bytes)
            await self._upload_to_supabase_async(
                temp_path, target_relative, mimetype=content_type
            )
            self.logger.info(
                "Uploaded %s to Supabase storage", target_relative.as_posix()
            )
            return StoredDocument(
                relative_path=target_relative,
                local_path=temp_path,
                should_cleanup=True,
            )

        target_path = self._prepare_target_path(target_relative)
        final_relative = target_path.relative_to(self.base_dir)

        with target_path.open("wb") as destination:
            while True:
                chunk = await upload.read(1_048_576)
                if not chunk:
                    break
                destination.write(chunk)
        await upload.close()

        self.logger.info("Stored upload as %s", final_relative)
        await self._upload_to_supabase_async(
            target_path, final_relative, mimetype=content_type
        )
        return StoredDocument(
            relative_path=final_relative,
            local_path=target_path,
            should_cleanup=False,
        )

    async def save_pdf_bytes(
        self, data: bytes, *, relative_path: str
    ) -> StoredDocument:
        """Persist PDF bytes decoded from archive uploads."""
        target_relative = self._sanitize_relative_path(
            relative_path, fallback_name=relative_path
        )

        if self._supabase_enabled:
            temp_path = self._write_temp_file(data)
            await self._upload_to_supabase_async(
                temp_path, target_relative, mimetype="application/pdf"
            )
            self.logger.info(
                "Uploaded archive member %s to Supabase storage",
                target_relative.as_posix(),
            )
            return StoredDocument(
                relative_path=target_relative,
                local_path=temp_path,
                should_cleanup=True,
            )

        target_path = self._prepare_target_path(target_relative)
        final_relative = target_path.relative_to(self.base_dir)
        await asyncio.to_thread(target_path.write_bytes, data)
        self.logger.info("Stored archive member as %s", final_relative)
        await self._upload_to_supabase_async(
            target_path, final_relative, mimetype="application/pdf"
        )
        return StoredDocument(
            relative_path=final_relative,
            local_path=target_path,
            should_cleanup=False,
        )

    def _sanitize_relative_path(self, path: str, *, fallback_name: str) -> Path:
        candidate = PurePosixPath(path)
        parts = []
        for part in candidate.parts:
            if part in {"", ".", ".."}:
                continue
            safe_part = re.sub(r"[^a-zA-Z0-9._-]", "_", part) or "_"
            parts.append(safe_part)

        if not parts:
            fallback_stem = re.sub(r"[^a-zA-Z0-9._-]", "_", Path(fallback_name).stem) or "document"
            parts = [f"{fallback_stem}.pdf"]

        relative = Path(*parts)
        if relative.suffix.lower() != ".pdf":
            relative = relative.with_suffix(".pdf")
        return relative

    def _prepare_target_path(self, relative_path: Path) -> Path:
        target_path = self.base_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            target_path = target_path.with_name(f"{target_path.stem}_{timestamp}{target_path.suffix}")
        return target_path

    def delete_file(self, relative_path: str) -> bool:
        path_obj = Path(relative_path)
        target_path = (self.base_dir / path_obj).resolve()
        try:
            target_path.relative_to(self.base_dir)
        except ValueError:
            raise ValueError("Attempted to delete a path outside the storage directory")

        removed_local = False

        if target_path.exists():
            try:
                target_path.unlink()
                removed_local = True
                self._cleanup_empty_dirs(target_path.parent)
                self.logger.info("Deleted local file %s", path_obj.as_posix())
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to delete local file %s: %s", path_obj.as_posix(), exc)
                raise

        if self._supabase_enabled:
            try:
                bucket = self._supabase_client.storage.from_(self._supabase_bucket)
                bucket.remove([path_obj.as_posix()])
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error(
                    "Failed to delete %s from Supabase storage: %s",
                    path_obj.as_posix(),
                    exc,
                )
                raise

        return removed_local

    def cleanup_local_path(self, local_path: Path, *, allow_dir_cleanup: bool = False) -> None:
        """Delete a temporary local file created during ingestion."""
        if not local_path.exists():
            return

        try:
            local_path.unlink()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to remove local file %s: %s", local_path, exc)
            return

        if allow_dir_cleanup:
            try:
                local_path.parent.relative_to(self.base_dir)
            except ValueError:
                return
            self._cleanup_empty_dirs(local_path.parent)

    def _cleanup_empty_dirs(self, directory: Path) -> None:
        try:
            directory.relative_to(self.base_dir)
        except ValueError:
            return

        while directory != self.base_dir:
            try:
                directory.rmdir()
            except OSError:
                break
            directory = directory.parent

    @property
    def _supabase_enabled(self) -> bool:
        return self._supabase_client is not None and bool(self._supabase_bucket)

    async def _upload_to_supabase_async(
        self, file_path: Path, relative_path: Path, *, mimetype: str
    ) -> None:
        if not self._supabase_enabled:
            return

        try:
            await asyncio.to_thread(
                self._upload_to_supabase,
                file_path,
                relative_path.as_posix(),
                mimetype,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "Failed to upload %s to Supabase storage: %s",
                relative_path.as_posix(),
                exc,
            )

    def _upload_to_supabase(self, file_path: Path, storage_path: str, mimetype: str) -> None:
        if not self._supabase_enabled:
            return

        assert self._supabase_client is not None
        bucket = self._supabase_client.storage.from_(self._supabase_bucket)
        with file_path.open("rb") as file_obj:
            bucket.upload(
                storage_path,
                file_obj,
                {"content-type": mimetype or "application/pdf", "upsert": "true"},
            )

    def _write_temp_file(self, data: bytes) -> Path:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        with temp_file:
            temp_file.write(data)
        temp_path = Path(temp_file.name)
        self.logger.debug("Wrote temporary PDF to %s", temp_path)
        return temp_path
