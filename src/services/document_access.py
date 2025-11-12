from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from src.services.settings_manager import UserAccessProfile
from src.utils.paths import directory_matches_filter, normalize_relative_path


@dataclass
class DocumentAccessSnapshot:
    """Filtered view of documents a user is permitted to access."""

    profile: UserAccessProfile
    documents: list[Mapping[str, object]]
    normalized_allowed_directories: set[str]
    allow_all_directories: bool
    allowed_relative_paths: set[str]
    normalized_relative_paths: set[str]


def compute_directory_permissions(profile: UserAccessProfile) -> tuple[set[str], bool]:
    """Return the normalized directories granted to the user and whether all are allowed."""

    normalized_directories: set[str] = set()
    allow_all = profile.has_full_access

    for path in profile.folder_paths:
        normalized = normalize_relative_path(path)
        if normalized == "":
            allow_all = True
        elif normalized:
            normalized_directories.add(normalized)

    return normalized_directories, allow_all


def build_document_access_snapshot(
    documents: Sequence[Mapping[str, object]],
    profile: UserAccessProfile,
) -> DocumentAccessSnapshot:
    """Filter documents to those visible under the supplied profile."""

    normalized_allowed, allow_all = compute_directory_permissions(profile)

    if allow_all:
        accessible_docs = [dict(item) for item in documents]
        allowed_relative_paths = {
            str(doc.get("relative_path"))
            for doc in accessible_docs
            if doc.get("relative_path")
        }
        normalized_relative_paths = {
            normalize_relative_path(path) for path in allowed_relative_paths
        }
        return DocumentAccessSnapshot(
            profile=profile,
            documents=accessible_docs,
            normalized_allowed_directories=normalized_allowed,
            allow_all_directories=True,
            allowed_relative_paths=allowed_relative_paths,
            normalized_relative_paths=normalized_relative_paths,
        )

    accessible_docs: list[Mapping[str, object]] = []
    allowed_relative_paths: set[str] = set()
    normalized_relative_paths: set[str] = set()

    for doc in documents:
        directory_raw = doc.get("directory")
        directory = normalize_relative_path(str(directory_raw) if directory_raw is not None else "")

        include = False
        for allowed_dir in normalized_allowed:
            if directory_matches_filter(directory, allowed_dir):
                include = True
                break

        if include:
            accessible_docs.append(dict(doc))
            relative_path = doc.get("relative_path")
            if isinstance(relative_path, str) and relative_path:
                allowed_relative_paths.add(relative_path)
                normalized_relative_paths.add(normalize_relative_path(relative_path))

    return DocumentAccessSnapshot(
        profile=profile,
        documents=accessible_docs,
        normalized_allowed_directories=normalized_allowed,
        allow_all_directories=False,
        allowed_relative_paths=allowed_relative_paths,
        normalized_relative_paths=normalized_relative_paths,
    )


def is_directory_allowed(directory: str, snapshot: DocumentAccessSnapshot) -> bool:
    """Check whether a directory path is within the user's allowed scope."""

    if snapshot.allow_all_directories:
        return True

    normalized = normalize_relative_path(directory)
    if not normalized:
        return "" in snapshot.normalized_allowed_directories or snapshot.allow_all_directories

    if normalized in snapshot.normalized_allowed_directories:
        return True

    return any(
        directory_matches_filter(normalized, allowed)
        for allowed in snapshot.normalized_allowed_directories
    )


def validate_requested_paths(
    requested_paths: Sequence[str] | None,
    snapshot: DocumentAccessSnapshot,
) -> tuple[list[str], list[str]]:
    """Split requested paths into allowed and disallowed subsets."""

    if not requested_paths:
        return [], []

    allowed: list[str] = []
    denied: list[str] = []

    for entry in requested_paths:
        normalized = normalize_relative_path(entry)
        if snapshot.allow_all_directories:
            allowed.append(entry)
            continue

        if normalized in snapshot.normalized_relative_paths:
            allowed.append(entry)
            continue

        if is_directory_allowed(normalized, snapshot):
            allowed.append(entry)
            continue

        denied.append(entry)

    return allowed, denied


def directory_allowed_by_profile(profile: UserAccessProfile, directory: str) -> bool:
    """Return True if the supplied directory is accessible for the profile."""

    normalized_allowed, allow_all = compute_directory_permissions(profile)
    if allow_all:
        return True

    normalized = normalize_relative_path(directory)
    if not normalized:
        return "" in normalized_allowed or allow_all

    if normalized in normalized_allowed:
        return True

    return any(
        directory_matches_filter(normalized, allowed)
        for allowed in normalized_allowed
    )
