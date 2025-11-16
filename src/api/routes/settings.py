from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies.auth import SupabaseUser, require_supabase_user
from src.core.config import get_settings
from src.services.document_permissions import DocumentPermissionsService
from src.services.metadata import MetadataService
from src.services.settings_manager import SettingsManager

logger = logging.getLogger("ai_backend.settings")
router = APIRouter(prefix="/settings", tags=["settings"])

settings = get_settings()
settings_manager = SettingsManager(settings=settings, logger=logger)
doc_permissions_service = DocumentPermissionsService(settings=settings, logger=logger)
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
        normalized_role = str(user_role).strip().lower().replace("-", "_").replace(" ", "_")

        # Only admin and superadmin can access settings
        if normalized_role not in {"admin", "superadmin", "super_admin"}:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin and superadmin users can access settings"
            )
        group_members = settings_manager.list_group_members()
        group_access = settings_manager.list_group_access()
        group_folder_permissions = settings_manager.list_group_folder_permissions()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    directories = _collect_unique_directories()
    folder_tree = _build_folder_tree()

    return {
        "user_role": user_role,
        "group_members": group_members,
        "group_access": group_access,
        "group_folder_permissions": group_folder_permissions,
        "folders": directories,
        "folder_tree": folder_tree,
    }


@router.get("/users/paginated", response_model=dict[str, Any])
async def get_paginated_users(
    offset: int = 0,
    limit: int = 10,
    search: str = "",
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    """Get paginated list of users with search"""
    try:
        user_role = settings_manager.get_user_role(current_user.id)
        normalized_role = str(user_role).strip().lower().replace("-", "_").replace(" ", "_")

        if normalized_role not in {"admin", "superadmin", "super_admin"}:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin and superadmin users can access settings"
            )

        result = settings_manager.list_users_paginated(offset=offset, limit=limit, search=search)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return result


@router.get("/groups/paginated", response_model=dict[str, Any])
async def get_paginated_groups(
    offset: int = 0,
    limit: int = 10,
    search: str = "",
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    """Get paginated list of groups with search"""
    try:
        user_role = settings_manager.get_user_role(current_user.id)
        normalized_role = str(user_role).strip().lower().replace("-", "_").replace(" ", "_")

        if normalized_role not in {"admin", "superadmin", "super_admin"}:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin and superadmin users can access settings"
            )

        result = settings_manager.list_groups_paginated(offset=offset, limit=limit, search=search)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return result


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


@router.get("/user-document-permissions/{user_id}", response_model=dict[str, Any])
async def get_user_document_permissions(
    user_id: str,
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    """Get all document permissions for a specific user"""
    # Check if current user is admin/superadmin
    user_role = settings_manager.get_user_role(current_user.id)
    if user_role not in {"admin", "superadmin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can view user document permissions"
        )

    permissions = doc_permissions_service.get_all_user_document_permissions(user_id)

    return {
        "user_id": user_id,
        "permissions": [
            {
                "user_id": perm.user_id,
                "document_path": perm.document_path,
                "can_upload": perm.can_upload,
                "can_view": perm.can_view,
                "can_delete": perm.can_delete,
            }
            for perm in permissions
        ],
    }


@router.post("/user-document-permissions/{user_id}", response_model=dict[str, str])
async def save_user_document_permissions(
    user_id: str,
    payload: dict[str, Any],
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, str]:
    """Save document permissions for a specific user"""
    # Check if current user is admin/superadmin
    user_role = settings_manager.get_user_role(current_user.id)
    if user_role not in {"admin", "superadmin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can manage user document permissions"
        )

    permissions = payload.get("permissions", [])
    if not isinstance(permissions, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="permissions must be a list"
        )

    # First, get all existing permissions and revoke them
    existing_perms = doc_permissions_service.get_all_user_document_permissions(user_id)
    for existing_perm in existing_perms:
        try:
            doc_permissions_service.revoke_document_permission(user_id, existing_perm.document_path)
        except Exception as exc:
            logger.warning(f"Failed to revoke permission for {existing_perm.document_path}: {exc}")

    # Then grant the new permissions
    for perm in permissions:
        if not isinstance(perm, dict):
            continue

        document_path = perm.get("document_path")
        can_upload = perm.get("can_upload", False)
        can_view = perm.get("can_view", False)
        can_delete = perm.get("can_delete", False)

        if not document_path:
            continue

        try:
            doc_permissions_service.grant_document_permission(
                user_id=user_id,
                document_path=str(document_path),
                can_upload=bool(can_upload),
                can_view=bool(can_view),
                can_delete=bool(can_delete),
                granted_by=current_user.id,
            )
        except Exception as exc:
            logger.error(f"Failed to grant permission for {document_path}: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to grant permission for {document_path}"
            ) from exc

    return {"status": "ok"}


@router.post("/groups", response_model=dict[str, Any])
async def create_group(
    payload: dict[str, Any],
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    """Create a new group"""
    # Check if current user is admin/superadmin
    user_role = settings_manager.get_user_role(current_user.id)
    if user_role not in {"admin", "superadmin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can create groups"
        )

    group_name = payload.get("name")
    if not isinstance(group_name, str) or not group_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Group name is required"
        )

    try:
        group_id = settings_manager.create_group(
            name=group_name.strip(),
            created_by=current_user.id,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc

    return {"status": "ok", "group_id": group_id}


@router.post("/groups/{group_id}/members", response_model=dict[str, str])
async def add_user_to_group(
    group_id: str,
    payload: dict[str, Any],
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, str]:
    """Add a user to a group"""
    # Check if current user is admin/superadmin
    user_role = settings_manager.get_user_role(current_user.id)
    if user_role not in {"admin", "superadmin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can add users to groups"
        )

    user_id = payload.get("user_id")
    if not isinstance(user_id, str) or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID is required"
        )

    try:
        settings_manager.add_user_to_groups(
            user_id=user_id.strip(),
            group_ids=[group_id],
            added_by=current_user.id,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc

    return {"status": "ok"}


@router.delete("/groups/{group_id}/members/{user_id}", response_model=dict[str, str])
async def remove_user_from_group(
    group_id: str,
    user_id: str,
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, str]:
    """Remove a user from a group"""
    # Check if current user is admin/superadmin
    user_role = settings_manager.get_user_role(current_user.id)
    if user_role not in {"admin", "superadmin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can remove users from groups"
        )

    try:
        settings_manager.remove_user_from_group(
            user_id=user_id,
            group_id=group_id,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc

    return {"status": "ok"}
