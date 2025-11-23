from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies.auth import SupabaseUser, require_supabase_user
from src.core.config import get_settings
from src.services.document_permissions import DocumentPermissionsService
from src.services.metadata import MetadataService
from src.services.settings_manager import SettingsManager
from src.services.folder_permissions import FolderPermissionsService, normalize_folder_path
from src.utils.paths import normalize_relative_path

logger = logging.getLogger("ai_backend.settings")
router = APIRouter(prefix="/settings", tags=["settings"])

settings = get_settings()
settings_manager = SettingsManager(settings=settings, logger=logger)
doc_permissions_service = DocumentPermissionsService(settings=settings, logger=logger)
folder_permissions_service = FolderPermissionsService(settings=settings, logger=logger)
metadata_service = MetadataService(
    settings.documents_metadata_path,
    logger=logger,
    supabase_url=settings.supabase_url,
    supabase_key=settings.supabase_key,
    supabase_table=settings.supabase_documents_table,
    default_org_id=settings.default_org_id,
)


def _normalize_branch_filter(branch_filter: Optional[Sequence[str] | str]) -> set[str]:
    if branch_filter is None:
        return set()
    if isinstance(branch_filter, str):
        return {branch_filter}
    return {value for value in branch_filter if isinstance(value, str) and value}


def _folder_contains_document(folder_path: str, document_directory: str) -> bool:
    normalized_folder = normalize_folder_path(folder_path)
    normalized_directory = normalize_relative_path(document_directory or "")
    if normalized_folder == "":
        return True
    if not normalized_directory:
        return False
    return (
        normalized_directory == normalized_folder
        or normalized_directory.startswith(f"{normalized_folder}/")
    )



def _collect_unique_directories(
    branch_id: Optional[str] = None,
    *,
    branch_ids: Optional[Sequence[str]] = None,
    org_id: Optional[str] = None,
) -> list[str]:
    documents = metadata_service.list_documents(org_id=org_id)

    # Filter by branch if specified
    allowed_branches = _normalize_branch_filter(branch_ids if branch_ids is not None else branch_id)
    if allowed_branches:
        documents = [doc for doc in documents if doc.get("branch_id") in allowed_branches]

    directories = {item.get("directory", "") for item in documents}
    # Remove empty directories to keep the tree cleaner; FolderTree ignores empty entries anyway
    directories.discard(None)  # type: ignore[arg-type]
    cleaned = {directory for directory in directories if isinstance(directory, str) and directory.strip()}
    return sorted(cleaned)


def _folder_sort_key(value: str) -> str:
    return value.lower()


def _build_folder_tree(
    branch_id: Optional[str] = None,
    *,
    branch_ids: Optional[Sequence[str]] = None,
    org_id: Optional[str] = None,
) -> list[dict[str, object]]:
    # Get directory tree filtered by branch
    allowed_branches = _normalize_branch_filter(branch_ids if branch_ids is not None else branch_id)
    if allowed_branches:
        all_documents = metadata_service.list_documents(org_id=org_id)
        filtered_documents = [doc for doc in all_documents if doc.get("branch_id") in allowed_branches]
        # Build directory tree from filtered documents
        directory_index = metadata_service._build_directory_tree_from_records(filtered_documents)
    else:
        directory_index = metadata_service.get_directory_tree(org_id=org_id)

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
        profile = settings_manager.get_user_access_profile(current_user.id)
        user_role = profile.role
        normalized_role = profile.normalized_role

        # Only OrgOwner and BranchManager can access settings
        has_admin_permission = (
            normalized_role in {"OrgOwner", "BranchManager"} or
            "ORG_MANAGE" in profile.permission_codes or
            "BRANCH_ADMIN" in profile.permission_codes
        )

        if not has_admin_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only OrgOwner and BranchManager users can access settings"
            )

        # OrgOwner has full access, BranchManager has limited access
        is_org_owner = (normalized_role == "OrgOwner" or "ORG_MANAGE" in profile.permission_codes)
        user_branch_ids = list(profile.branch_ids)

        group_members = settings_manager.list_group_members(
            requesting_user_id=current_user.id,
            org_id=profile.org_id,
        )
        group_access: list[dict[str, Any]] = []
        org_context = profile.org_id or settings.default_org_id
        if org_context:
            group_folder_permissions = folder_permissions_service.list_group_folder_permissions(org_id=org_context)
        else:
            group_folder_permissions = []
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    # Filter directories and folders by branch for non-superadmins
    directories = _collect_unique_directories(branch_ids=user_branch_ids, org_id=profile.org_id)
    folder_tree = _build_folder_tree(branch_ids=user_branch_ids, org_id=profile.org_id)

    # Get branches list (OrgOwner sees all, BranchManager sees only their branch)
    try:
        all_branches = settings_manager.list_branches(org_id=profile.org_id)
        if is_org_owner:
            branches = all_branches
        elif user_branch_ids:
            allowed = set(user_branch_ids)
            branches = [b for b in all_branches if b.get("id") in allowed]
        else:
            branches = []
    except Exception:
        branches = []

    return {
        "user_role": normalized_role,
        "group_members": group_members,
        "group_access": group_access,
        "group_folder_permissions": group_folder_permissions,
        "folders": directories,
        "folder_tree": folder_tree,
        "branches": branches,
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
        profile = settings_manager.get_user_access_profile(current_user.id)
        user_role = profile.normalized_role

        if user_role not in {"OrgOwner", "BranchManager"}:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only OrgOwner and BranchManager users can access settings"
            )

        result = settings_manager.list_users_paginated(
            offset=offset,
            limit=limit,
            search=search,
            requesting_user_id=current_user.id
        )
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
        profile = settings_manager.get_user_access_profile(current_user.id)
        user_role = profile.normalized_role

        if user_role not in {"OrgOwner", "BranchManager"}:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only OrgOwner and BranchManager users can access settings"
            )

        result = settings_manager.list_groups_paginated(
            offset=offset,
            limit=limit,
            search=search,
            requesting_user_id=current_user.id
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return result


@router.post("/users", response_model=dict[str, Any])
async def add_user_to_groups(
    payload: dict[str, Any],
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    import logging
    logger = logging.getLogger("ai_backend.settings")

    profile = settings_manager.get_user_access_profile(current_user.id)
    email = payload.get("email")
    role = payload.get("role")
    group_ids = payload.get("group_ids") or []
    folder_paths = payload.get("folder_paths") or []
    # Support both single branch (legacy) and multiple branches
    branch_ids = payload.get("branch_ids")
    branch_names = payload.get("branch_names")

    logger.info("Received user creation request: email=%s, role=%s, branch_ids=%s", email, role, branch_ids)

    # Fallback to single branch for backward compatibility
    if not branch_ids:
        branch_id = payload.get("branch_id")
        branch_name = payload.get("branch_name")
        if branch_id:
            branch_ids = [branch_id]
            branch_names = [branch_name] if branch_name else []
            logger.info("Converted single branch to array: branch_ids=%s", branch_ids)

    if not isinstance(email, str) or not email.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email is required")

    try:
        if role is not None:
            role = str(role)
        group_ids = [str(value) for value in group_ids if isinstance(value, str)]
        folder_paths = [str(value) for value in folder_paths if isinstance(value, str)]
        if branch_ids:
            branch_ids = [str(value) for value in branch_ids if isinstance(value, str)]
        if branch_names:
            branch_names = [str(value) for value in branch_names if isinstance(value, str)]

        result = settings_manager.create_user_assignments(
            email=email.strip(),
            role=role,
            group_ids=group_ids,
            folder_paths=folder_paths,
            acting_user_id=current_user.id,
            branch_ids=branch_ids if branch_ids else None,
            branch_names=branch_names if branch_names else None,
            org_id=profile.org_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return {"status": "ok", "user_id": result.get("user_id")}


@router.post("/group-access", response_model=dict[str, str])
async def update_group_access(
    payload: dict[str, Any],
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, str]:
    return {"status": "ignored", "message": "Group access chaining is managed via document permissions"}


@router.post("/group-folder-permissions", response_model=dict[str, str])
async def update_group_folder_permissions(
    payload: dict[str, Any],
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, str]:
    profile = settings_manager.get_user_access_profile(current_user.id)
    if profile.normalized_role not in {"OrgOwner", "BranchManager"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only OrgOwner and BranchManager can manage group folder permissions"
        )

    group_id = payload.get("group_id")
    folder_paths = payload.get("folder_paths") or []

    if not isinstance(group_id, str) or not group_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="group_id is required"
        )

    if not isinstance(folder_paths, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="folder_paths must be a list"
        )

    org_context = profile.org_id or settings.default_org_id
    if not org_context:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Organization context is not configured"
        )

    folder_paths = [str(path) for path in folder_paths if isinstance(path, str)]

    try:
        folder_permissions_service.update_group_folder_permissions(
            group_id=group_id.strip(),
            folder_paths=folder_paths,
            org_id=org_context,
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
    # Check if current user is admin
    profile = settings_manager.get_user_access_profile(current_user.id)
    if profile.normalized_role not in {"OrgOwner", "BranchManager"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only OrgOwner and BranchManager can view user document permissions"
        )

    target_profile = settings_manager.get_user_access_profile(user_id)
    role_ids = [assignment.role.id for assignment in getattr(target_profile, "assignments", []) if getattr(assignment, "role", None)]
    org_context = target_profile.org_id or profile.org_id or settings.default_org_id
    folder_entries = folder_permissions_service.get_permissions_for_user_context(
        user_id=user_id,
        group_ids=list(target_profile.direct_group_ids),
        role_ids=role_ids,
        org_id=org_context,
    )

    permissions_payload: list[dict[str, Any]] = []
    for entry in folder_entries.values():
        permissions_payload.append(
            {
                "user_id": user_id,
                "document_path": entry.folder_path,
                "has_access": entry.can_view,
            }
        )

    return {
        "user_id": user_id,
        "permissions": permissions_payload,
    }


@router.post("/user-document-permissions/{user_id}", response_model=dict[str, str])
async def save_user_document_permissions(
    user_id: str,
    payload: dict[str, Any],
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, str]:
    """Save document permissions for a specific user"""
    # Check if current user is admin
    profile = settings_manager.get_user_access_profile(current_user.id)
    if profile.normalized_role not in {"OrgOwner", "BranchManager"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only OrgOwner and BranchManager can manage user document permissions"
        )

    permissions = payload.get("permissions", [])
    if not isinstance(permissions, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="permissions must be a list"
        )

    target_profile = settings_manager.get_user_access_profile(user_id)
    org_context = target_profile.org_id or profile.org_id or settings.default_org_id
    if not org_context:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Organization context is not configured"
        )

    documents = metadata_service.list_documents(org_id=org_context)

    try:
        folder_permissions_service.revoke_all_for_user(user_id)
        doc_permissions_service.revoke_all_for_user(user_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    for perm in permissions:
        if not isinstance(perm, dict):
            continue

        folder_path = normalize_folder_path(str(perm.get("document_path") or ""))
        has_access = bool(perm.get("has_access"))
        if not has_access:
            continue

        try:
            folder_permissions_service.grant_folder_permission(
                folder_path=folder_path,
                access_level="view",
                user_id=user_id,
                org_id=org_context,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

        try:
            for record in documents:
                document_directory = str(record.get("directory") or "")
                if not _folder_contains_document(folder_path, document_directory):
                    continue

                document_id = str(record.get("id") or record.get("relative_path") or "")
                if not document_id:
                    continue

                doc_permissions_service.grant_document_permission(
                    document_id=document_id,
                    access_level="view",
                    user_id=user_id,
                    branch_id=str(record.get("branch_id")) if record.get("branch_id") else None,
                )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to grant permission for folder {folder_path}"
            ) from exc

    return {"status": "ok"}


@router.post("/groups", response_model=dict[str, Any])
async def create_group(
    payload: dict[str, Any],
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    """Create a new group"""
    # Check if current user is admin
    profile = settings_manager.get_user_access_profile(current_user.id)
    user_role = profile.normalized_role
    is_org_owner = (user_role == "OrgOwner")

    if user_role not in {"OrgOwner", "BranchManager"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only OrgOwner and BranchManager can create groups"
        )

    group_name = payload.get("name")
    if not isinstance(group_name, str) or not group_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Group name is required"
        )

    # Get the user's branch (for non-OrgOwner, groups are created in their branch)
    user_branch_id = None
    if not is_org_owner:
        user_branch_id = profile.branch_id

    try:
        group_id = settings_manager.create_group(
            name=group_name.strip(),
            created_by=current_user.id,
            branch_id=user_branch_id,
            org_id=profile.org_id,
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
    # Check if current user is admin
    profile = settings_manager.get_user_access_profile(current_user.id)
    if profile.normalized_role not in {"OrgOwner", "BranchManager"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only OrgOwner and BranchManager can add users to groups"
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
    # Check if current user is admin
    profile = settings_manager.get_user_access_profile(current_user.id)
    if profile.normalized_role not in {"OrgOwner", "BranchManager"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only OrgOwner and BranchManager can remove users from groups"
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


@router.get("/roles", response_model=dict[str, Any])
async def list_roles(
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    """List all available roles"""
    try:
        profile = settings_manager.get_user_access_profile(current_user.id)

        # Only OrgOwner and BranchManager can see roles
        if profile.normalized_role not in {"OrgOwner", "BranchManager"}:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only OrgOwner and BranchManager can view roles"
            )

        roles = settings_manager.list_roles()
        return {"roles": roles}
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        ) from exc


@router.get("/branches", response_model=dict[str, Any])
async def list_branches(
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    """List branches accessible to the current user"""
    try:
        profile = settings_manager.get_user_access_profile(current_user.id)
        is_org_owner = profile.normalized_role == "OrgOwner"

        all_branches = settings_manager.list_branches(org_id=profile.org_id)

        # OrgOwner sees all branches, others see only their assigned branches
        if is_org_owner:
            branches = all_branches
        else:
            user_branch_ids = set(profile.branch_ids)
            branches = [b for b in all_branches if b.get("id") in user_branch_ids]

        return {"branches": branches}
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        ) from exc


@router.get("/branches/search", response_model=dict[str, Any])
async def search_branches(
    q: str = "",
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    """Search branches by name"""
    try:
        profile = settings_manager.get_user_access_profile(current_user.id)
        all_branches = settings_manager.list_branches(org_id=profile.org_id)

        # Filter branches by search query (case-insensitive)
        query = q.strip().lower()
        if query:
            filtered_branches = [
                branch for branch in all_branches
                if query in branch.get("name", "").lower()
            ]
        else:
            filtered_branches = all_branches

        return {"branches": filtered_branches}
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        ) from exc


@router.post("/branches", response_model=dict[str, Any])
async def create_branch(
    payload: dict[str, Any],
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    """Create a new branch (OrgOwner only)"""
    profile = settings_manager.get_user_access_profile(current_user.id)
    if profile.normalized_role != "OrgOwner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only OrgOwner can create branches"
        )

    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Branch name is required"
        )

    profile = settings_manager.get_user_access_profile(current_user.id)

    try:
        branch_id = settings_manager.create_branch(name=name.strip(), org_id=profile.org_id)
        return {
            "branch_id": branch_id,
            "message": f"Branch '{name}' created successfully"
        }
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc
