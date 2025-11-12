from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies.auth import SupabaseUser, require_supabase_user
from src.core.config import get_settings
from src.services.metadata import MetadataService
from src.services.settings_manager import SettingsManager

logger = logging.getLogger("ai_backend.settings")
router = APIRouter(prefix="/settings", tags=["settings"])

settings = get_settings()
settings_manager = SettingsManager(settings=settings, logger=logger)
metadata_service = MetadataService(
    settings.documents_metadata_path,
    logger=logger,
    supabase_url=settings.supabase_url,
    supabase_key=settings.supabase_key,
    supabase_table=settings.supabase_documents_table,
)


def _collect_unique_directories() -> list[str]:
    documents = metadata_service.list_documents()
    directories = {item.get("directory", "") for item in documents}
    # Remove empty directories to keep the tree cleaner; FolderTree ignores empty entries anyway
    directories.discard(None)  # type: ignore[arg-type]
    cleaned = {directory for directory in directories if isinstance(directory, str) and directory.strip()}
    return sorted(cleaned)


def _folder_sort_key(value: str) -> str:
    return value.lower()


def _build_folder_tree() -> list[dict[str, object]]:
    directory_index = metadata_service.get_directory_tree()
    root_entry = directory_index.get(".", {})
    root_children = root_entry.get("subdirectories", []) if isinstance(root_entry, dict) else []

    def build_node(path: str) -> dict[str, object]:
        entry = directory_index.get(path, {}) if isinstance(directory_index, dict) else {}
        children_paths = entry.get("subdirectories", []) if isinstance(entry, dict) else []
        sorted_children = sorted(
            (
                child
                for child in children_paths
                if isinstance(child, str) and child.strip() and child not in {"", "."}
            ),
            key=_folder_sort_key,
        )
        children_nodes = [build_node(child) for child in sorted_children]

        name = path.rsplit("/", 1)[-1] if "/" in path else path
        display_name = name or path or ""
        documents_here = (
            len(entry.get("files", [])) if isinstance(entry, dict) and isinstance(entry.get("files", []), list) else 0
        )
        descendant_docs = sum(
            child.get("total_documents", 0) for child in children_nodes if isinstance(child, dict)
        )

        node: dict[str, object] = {
            "name": display_name,
            "path": "" if path in {"", "."} else path,
            "documents": documents_here,
            "total_documents": documents_here + descendant_docs,
        }
        if children_nodes:
            node["children"] = children_nodes
        return node

    return [
        build_node(child_path)
        for child_path in sorted(
            (
                child
                for child in root_children
                if isinstance(child, str) and child.strip() and child not in {"", "."}
            ),
            key=_folder_sort_key,
        )
    ]


@router.get("/data", response_model=dict[str, Any])
async def get_settings_data(current_user: SupabaseUser = Depends(require_supabase_user)) -> dict[str, Any]:
    try:
        user_role = settings_manager.get_user_role(current_user.id)
        groups = settings_manager.list_groups()
        group_members = settings_manager.list_group_members()
        group_access = settings_manager.list_group_access()
        group_folder_permissions = settings_manager.list_group_folder_permissions()
        all_users = settings_manager.list_all_users()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    # Map group_id to folder paths for convenience when displaying in the UI
    folders_by_group: dict[str, list[str]] = {}
    for entry in group_folder_permissions:
        group_id = entry.get("group_id")
        folder_path = entry.get("folder_path")
        if not group_id or not isinstance(folder_path, str):
            continue
        folders_by_group.setdefault(group_id, [])
        if folder_path not in folders_by_group[group_id]:
            folders_by_group[group_id].append(folder_path)

    for group in groups:
        group_id = group.get("id")
        if group_id and group_id in folders_by_group:
            group["folder_paths"] = sorted(folders_by_group[group_id])

    directories = _collect_unique_directories()
    folder_tree = _build_folder_tree()

    return {
        "user_role": user_role,
        "groups": groups,
        "group_members": group_members,
        "group_access": group_access,
        "group_folder_permissions": group_folder_permissions,
        "folders": directories,
        "folder_tree": folder_tree,
        "all_users": all_users,
    }


@router.post("/users", response_model=dict[str, Any])
async def add_user_to_groups(
    payload: dict[str, Any],
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    email = payload.get("email")
    role = payload.get("role")
    group_ids = payload.get("group_ids") or []
    folder_paths = payload.get("folder_paths") or []

    if not isinstance(email, str) or not email.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email is required")

    try:
        if role is not None:
            role = str(role)
        group_ids = [str(value) for value in group_ids if isinstance(value, str)]
        folder_paths = [str(value) for value in folder_paths if isinstance(value, str)]

        result = settings_manager.create_user_assignments(
            email=email.strip(),
            role=role,
            group_ids=group_ids,
            folder_paths=folder_paths,
            acting_user_id=current_user.id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return {"status": "ok", "user_id": result.get("user_id")}


@router.post("/group-access", response_model=dict[str, str])
async def update_group_access(
    payload: dict[str, Any],
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, str]:
    group_id = payload.get("group_id")
    if not isinstance(group_id, str) or not group_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="group_id is required")

    accessible_group_ids = payload.get("accessible_group_ids") or []
    if not isinstance(accessible_group_ids, list):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="accessible_group_ids must be a list")

    try:
        cast_ids = [str(value) for value in accessible_group_ids if isinstance(value, str)]
        settings_manager.replace_group_access(
            group_id=group_id,
            accessible_group_ids=cast_ids,
            acting_user_id=current_user.id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return {"status": "ok"}


@router.post("/group-folder-permissions", response_model=dict[str, str])
async def update_group_folder_permissions(
    payload: dict[str, Any],
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, str]:
    group_id = payload.get("group_id")
    if not isinstance(group_id, str) or not group_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="group_id is required")

    folder_paths = payload.get("folder_paths") or []
    if not isinstance(folder_paths, list):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="folder_paths must be a list")

    try:
        cast_paths = [str(value) for value in folder_paths if isinstance(value, str)]
        settings_manager.replace_group_folder_permissions(
            group_id=group_id,
            folder_paths=cast_paths,
            acting_user_id=current_user.id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return {"status": "ok"}


@router.delete("/users/{target_user_id}", response_model=dict[str, str])
async def delete_user(
    target_user_id: str,
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, str]:
    try:
        settings_manager.delete_user(target_user_id=target_user_id, acting_user_id=current_user.id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return {"status": "ok"}
