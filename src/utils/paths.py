from __future__ import annotations

from pathlib import PurePosixPath


def normalize_relative_path(path: str | None) -> str:
    """Return a normalized, POSIX-style relative path without dot segments."""
    if not path:
        return ""
    parts = [
        segment
        for segment in PurePosixPath(path).parts
        if segment not in {"", ".", ".."}
    ]
    return "/".join(parts)


def directory_matches_filter(document_directory: str, directory_filter: str) -> bool:
    """Return True when document_directory is under the directory_filter (inclusive)."""
    if directory_filter == "":
        return document_directory == ""
    if document_directory == directory_filter:
        return True
    return document_directory.startswith(f"{directory_filter}/")
