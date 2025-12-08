from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

class ContextCacheService:
    """Simple file-based cache for session context (extracted text from uploads)."""

    def __init__(self, data_dir: Path, *, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.context_dir = data_dir / "session_context"
        self.context_dir.mkdir(parents=True, exist_ok=True)

    def append_context(self, session_id: str, text: str) -> None:
        """Append text to the session's context file."""
        if not session_id or not text:
            return
        
        file_path = self.context_dir / f"{session_id}.txt"
        try:
            with file_path.open("a", encoding="utf-8") as f:
                f.write(text + "\n\n")
        except Exception as exc:
            self.logger.error("Failed to append context for session %s: %s", session_id, exc)

    def get_context(self, session_id: str) -> Optional[str]:
        """Retrieve the full context for a session."""
        if not session_id:
            return None
            
        file_path = self.context_dir / f"{session_id}.txt"
        if not file_path.exists():
            return None
            
        try:
            return file_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            self.logger.error("Failed to read context for session %s: %s", session_id, exc)
            return None

    def clear_context(self, session_id: str) -> None:
        """Clear the context for a session."""
        file_path = self.context_dir / f"{session_id}.txt"
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as exc:
            self.logger.error("Failed to clear context for session %s: %s", session_id, exc)
