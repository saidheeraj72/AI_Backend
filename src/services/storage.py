from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Optional

from fastapi import UploadFile


class StorageService:
    """Persist uploaded files to the configured data directory."""

    def __init__(self, base_dir: Path, *, logger: Optional[logging.Logger] = None) -> None:
        self.base_dir = base_dir
        self.logger = logger or logging.getLogger(__name__)

    async def save_pdf(self, upload: UploadFile, *, relative_path: Optional[str] = None) -> Path:
        """Store an uploaded PDF on disk, preserving optional relative directories."""
        original_name = upload.filename or "document.pdf"
        target_relative = self._sanitize_relative_path(relative_path or original_name, fallback_name=original_name)
        target_path = self._prepare_target_path(target_relative)

        with target_path.open("wb") as destination:
            while True:
                chunk = await upload.read(1_048_576)
                if not chunk:
                    break
                destination.write(chunk)
        await upload.close()

        self.logger.info("Stored upload as %s", target_path.relative_to(self.base_dir))
        return target_path

    def save_pdf_bytes(self, data: bytes, *, relative_path: str) -> Path:
        """Persist PDF bytes decoded from archive uploads."""
        target_relative = self._sanitize_relative_path(relative_path, fallback_name=relative_path)
        target_path = self._prepare_target_path(target_relative)
        target_path.write_bytes(data)
        self.logger.info("Stored archive member as %s", target_path.relative_to(self.base_dir))
        return target_path

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
