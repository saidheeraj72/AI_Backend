from __future__ import annotations

import logging
from typing import Any, Sequence

from functools import partial

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from src.api.dependencies.auth import SupabaseUser, require_supabase_user
from src.core.config import get_settings
from src.models.schemas import (
    DocumentIngestResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentSearchRequest,
    DocumentSearchResponse,
)
from src.services.document_ingestor import DocumentIngestor, NoValidDocumentsError
from src.services.document_permissions import ACCESS_RANK, DocumentPermission, DocumentPermissionsService
from src.services.metadata import MetadataService
from src.services.pdf_processor import PDFProcessor
from src.services.storage import StorageService
from src.services.vectorstore import VectorStoreService
from src.services.settings_manager import SettingsManager
from src.services.folder_permissions import FolderPermissionsService, expand_folder_permissions_to_documents
from src.utils.paths import directory_matches_filter, normalize_relative_path

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)

settings = get_settings()
logger = logging.getLogger("ai_backend.documents")
pdf_processor = PDFProcessor(logger=logger)
vector_service = VectorStoreService(settings, logger=logger)
storage_service = StorageService(
    settings.pdf_directory,
    supabase_url=settings.supabase_url,
    supabase_key=settings.supabase_key,
    supabase_bucket=settings.supabase_storage_bucket,
    logger=logger,
)
metadata_service = MetadataService(
    settings.documents_metadata_path,
    logger=logger,
    supabase_url=settings.supabase_url,
    supabase_key=settings.supabase_key,
    supabase_table=settings.supabase_documents_table,
    default_org_id=settings.default_org_id,
)
document_ingestor = DocumentIngestor(
    settings=settings,
    storage_service=storage_service,
    pdf_processor=pdf_processor,
    vector_service=vector_service,
    metadata_service=metadata_service,
    logger=logger,
)
settings_manager = SettingsManager(settings=settings, logger=logger)
doc_permissions_service = DocumentPermissionsService(settings=settings, logger=logger)
folder_permissions_service = FolderPermissionsService(settings=settings, logger=logger)


def _user_has_permission(profile, code: str) -> bool:
    return bool(profile.has_full_access or (hasattr(profile, "permission_codes") and code in profile.permission_codes))


def _collect_user_doc_acl(
    profile,
    user_id: str,
    *,
    documents: Sequence[dict[str, Any]] | None = None,
) -> dict[str, DocumentPermission]:
    role_ids = [assignment.role.id for assignment in getattr(profile, "assignments", []) if getattr(assignment, "role", None)]
    doc_map = doc_permissions_service.get_permissions_for_user_context(
        user_id=user_id,
        group_ids=list(profile.direct_group_ids),
        role_ids=role_ids,
    )

    folder_entries = folder_permissions_service.get_permissions_for_user_context(
        user_id=user_id,
        group_ids=list(profile.direct_group_ids),
        role_ids=role_ids,
        org_id=profile.org_id,
    )

    if folder_entries:
        documents_for_folder_acl = documents if documents is not None else metadata_service.list_documents(org_id=profile.org_id)
        folder_doc_map = expand_folder_permissions_to_documents(folder_entries.values(), documents_for_folder_acl)
        for document_id, permission in folder_doc_map.items():
            existing = doc_map.get(document_id)
            if existing is None or ACCESS_RANK.get(permission.access_level, 0) > ACCESS_RANK.get(existing.access_level, 0):
                doc_map[document_id] = permission

    return doc_map


@router.post("/upload", response_model=DocumentIngestResponse)
async def upload_document(
    files: list[UploadFile] | None = File(default=None),
    file: UploadFile | None = File(default=None),
    relative_paths: list[str] | None = Form(default=None),
    branch_id: str | None = Form(default=None),
    branch_name: str | None = Form(default=None),
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> DocumentIngestResponse:
    profile = settings_manager.get_user_access_profile(current_user.id)
    is_org_owner = profile.normalized_role == "OrgOwner"
    can_upload_role = _user_has_permission(profile, "DOC_UPLOAD")

    if not (is_org_owner or can_upload_role):
        raise HTTPException(status_code=403, detail="You do not have permission to upload documents")

    uploads: list[UploadFile] = []
    if files:
        uploads.extend(files)
    if file is not None:
        uploads.append(file)

    if not uploads:
        raise HTTPException(status_code=400, detail="At least one file must be provided")

    sanitized_paths: list[str] = []
    if relative_paths is not None:
        if len(relative_paths) != len(uploads):
            raise HTTPException(
                status_code=400,
                detail="Number of relative_paths entries must match uploaded files",
            )
        for provided_path in relative_paths:
            normalized = normalize_relative_path(provided_path)
            sanitized_paths.append(normalized)

        for upload, normalized in zip(uploads, sanitized_paths, strict=False):
            if normalized:
                upload.filename = normalized

    # Determine branch context
    branch_candidates = set(profile.branch_ids)
    upload_branch_id = None
    upload_branch_name = None

    if is_org_owner:
        upload_branch_id = branch_id
        upload_branch_name = branch_name
    else:
        if branch_id:
            if branch_id not in branch_candidates:
                raise HTTPException(status_code=403, detail="You are not assigned to the selected branch")
            upload_branch_id = branch_id
        else:
            upload_branch_id = profile.branch_id
        upload_branch_name = profile.branch_name

    org_id = profile.org_id or settings.default_org_id
    if not org_id:
        raise HTTPException(status_code=500, detail="Organization context is not configured")

    try:
        return await document_ingestor.ingest_uploads(
            uploads,
            branch_id=upload_branch_id,
            branch_name=upload_branch_name,
            org_id=org_id,
            created_by=current_user.id,
        )
    except NoValidDocumentsError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "No valid PDF files were ingested",
                "failures": [failure.dict() for failure in exc.failures],
            },
        ) from exc


@router.get("/permissions/debug", response_model=dict[str, Any])
async def debug_user_permissions(
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    """Debug endpoint to check current user's permissions"""
    profile = settings_manager.get_user_access_profile(current_user.id)

    return {
        "user_id": current_user.id,
        "email": current_user.email,
        "role": profile.role,
        "normalized_role": profile.normalized_role,
        "branch_id": profile.branch_id,
        "branch_name": profile.branch_name,
        "branch_ids": list(profile.branch_ids),
        "branch_names": list(profile.branch_names),
        "has_full_access": profile.has_full_access,
        "direct_group_ids": list(profile.direct_group_ids),
        "accessible_group_ids": list(profile.accessible_group_ids),
        "folder_paths": list(profile.folder_paths),
        "can_upload": profile.normalized_role in {"admin", "superadmin"},
        "permission_codes": list(getattr(profile, "permission_codes", [])),
    }


@router.get("/permissions", response_model=dict[str, Any])
async def get_user_document_permissions(
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    """Get all document-level permissions for the current user"""
    profile = settings_manager.get_user_access_profile(current_user.id)
    is_org_owner = profile.normalized_role == "OrgOwner"
    is_branch_manager = profile.normalized_role == "BranchManager"

    # OrgOwner has full access to everything
    if is_org_owner or profile.has_full_access:
        return {
            "user_id": current_user.id,
            "is_org_owner": is_org_owner,
            "has_full_access": True,
            "can_upload_role": True,
            "can_delete_role": True,
            "document_permissions": {},
        }

    # BranchManager has full access to documents in their branch
    if is_branch_manager:
        return {
            "user_id": current_user.id,
            "is_org_owner": False,
            "is_branch_manager": True,
            "has_full_access": True,
            "can_upload_role": True,
            "can_delete_role": True,
            "document_permissions": {},
        }

    doc_acl = _collect_user_doc_acl(profile, current_user.id)
    permissions_map: dict[str, dict[str, bool]] = {}
    for document_id, permission in doc_acl.items():
        record = metadata_service.get_document_by_id(document_id, org_id=profile.org_id)
        key = record.get("relative_path") if record else document_id
        permissions_map[key] = {
            "can_upload": permission.can_upload,
            "can_view": permission.can_view,
            "can_delete": permission.can_delete,
        }

    return {
        "user_id": current_user.id,
        "is_org_owner": False,
        "has_full_access": False,
        "can_upload_role": _user_has_permission(profile, "DOC_UPLOAD"),
        "can_delete_role": _user_has_permission(profile, "DOC_DELETE"),
        "document_permissions": permissions_map,
    }


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    directory: str | None = Query(
        default=None, description="Filter documents by their directory path"
    ),
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> DocumentListResponse:
    directory_filter = normalize_relative_path(directory) if directory else None

    profile = settings_manager.get_user_access_profile(current_user.id)
    raw_documents = metadata_service.list_documents(org_id=profile.org_id)
    is_org_owner = profile.normalized_role == "OrgOwner"
    doc_acl = _collect_user_doc_acl(profile, current_user.id, documents=raw_documents)

    accessible_documents: list[DocumentListItem] = []
    is_branch_manager = profile.normalized_role == "BranchManager"

    for record in raw_documents:
        directory_value = normalize_relative_path(str(record.get("directory", "")))
        if directory_filter is not None and not directory_matches_filter(directory_value, directory_filter):
            continue

        doc_id = str(record.get("id") or record.get("relative_path") or "").strip()
        if not doc_id:
            continue

        branch_id = record.get("branch_id")
        allowed = False
        if is_org_owner:
            allowed = True
        elif is_branch_manager and branch_id and branch_id in profile.branch_ids:
            # BranchManager gets automatic access to all documents in their branch
            allowed = True
        elif doc_id in doc_acl and doc_acl[doc_id].can_view:
            allowed = True

        if not allowed:
            continue

        try:
            accessible_documents.append(
                DocumentListItem(
                    id=str(record.get("id") or record.get("relative_path")),
                    filename=str(record.get("filename") or ""),
                    relative_path=str(record.get("relative_path") or record.get("storage_path") or ""),
                    directory=str(record.get("directory") or ""),
                    chunks_indexed=int(record.get("chunks_indexed", 0)),
                    branch_id=str(record.get("branch_id")) if record.get("branch_id") else None,
                    branch_name=str(record.get("branch_name")) if record.get("branch_name") else None,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Skipping document with invalid payload: %s", exc)

    return DocumentListResponse(documents=accessible_documents)


@router.delete("", response_model=dict[str, Any])
async def delete_document(
    relative_path: str = Query(..., description="Relative path of the document to delete"),
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    if not relative_path:
        raise HTTPException(status_code=400, detail="relative_path must be provided")

    profile = settings_manager.get_user_access_profile(current_user.id)

    document = await run_in_threadpool(
        partial(metadata_service.get_document, relative_path, org_id=profile.org_id)
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    is_org_owner = profile.normalized_role == "OrgOwner"
    is_branch_manager = profile.normalized_role == "BranchManager"
    doc_acl = _collect_user_doc_acl(profile, current_user.id)
    document_id = str(document.get("id") or relative_path)
    branch_id = document.get("branch_id")

    allowed = False
    if is_org_owner:
        allowed = True
    elif is_branch_manager and branch_id and branch_id in profile.branch_ids:
        # BranchManager gets automatic delete access to all documents in their branch
        allowed = True
    elif document_id in doc_acl and doc_acl[document_id].can_delete:
        allowed = True
    elif branch_id and branch_id in profile.branch_ids and _user_has_permission(profile, "DOC_DELETE"):
        allowed = True

    if not allowed:
        raise HTTPException(status_code=403, detail="You do not have delete permission for this document")

    try:
        vectors_removed = await run_in_threadpool(
            vector_service.remove_documents,
            [relative_path],
        )
        local_deleted = await run_in_threadpool(
            storage_service.delete_file,
            relative_path,
        )
        metadata_deleted = await run_in_threadpool(
            partial(metadata_service.delete_document, relative_path, org_id=profile.org_id),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to delete document %s: %s", relative_path, exc)
        raise HTTPException(status_code=500, detail="Failed to delete document") from exc

    return {
        "relative_path": relative_path,
        "metadata_deleted": metadata_deleted,
        "vectors_removed": vectors_removed,
        "local_deleted": local_deleted,
    }


@router.delete(
    "/directory",
    response_model=dict[str, Any],
    summary="Delete all documents within a directory",
)
async def delete_directory(
    directory: str = Query(..., description="Directory whose documents should be deleted"),
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    profile = settings_manager.get_user_access_profile(current_user.id)
    is_org_owner = profile.normalized_role == "OrgOwner"
    is_branch_manager = profile.normalized_role == "BranchManager"

    if not (is_org_owner or is_branch_manager or _user_has_permission(profile, "DOC_DELETE")):
        raise HTTPException(status_code=403, detail="You do not have permission to delete directories")

    normalized_directory = normalize_relative_path(directory)
    documents = [DocumentListItem(**item) for item in metadata_service.list_documents(org_id=profile.org_id)]

    target_paths = [
        document.relative_path
        for document in documents
        if directory_matches_filter(
            normalize_relative_path(document.directory),
            normalized_directory,
        )
    ]

    if not target_paths:
        raise HTTPException(
            status_code=404,
            detail="No documents found in the specified directory",
        )

    try:
        vectors_removed = await run_in_threadpool(
            vector_service.remove_documents,
            target_paths,
        )

        storage_results: list[bool] = []
        metadata_results: list[bool] = []

        for relative_path in target_paths:
            storage_deleted = await run_in_threadpool(
                storage_service.delete_file,
                relative_path,
            )
            storage_results.append(storage_deleted)

        for relative_path in target_paths:
            metadata_deleted = await run_in_threadpool(
                partial(metadata_service.delete_document, relative_path, org_id=profile.org_id),
            )
            metadata_results.append(metadata_deleted)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception(
            "Failed to delete documents in directory %s: %s",
            normalized_directory,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to delete documents in directory",
        ) from exc

    return {
        "directory": normalized_directory,
        "documents_requested": len(target_paths),
        "vectors_removed": vectors_removed,
        "storage_deleted": sum(storage_results),
        "metadata_deleted": sum(metadata_results),
        "relative_paths": target_paths,
    }


@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(
    request: DocumentSearchRequest,
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> DocumentSearchResponse:
    """
    Global search across all documents in the database.
    Searches through all groups and users regardless of current user's permissions.
    Only OrgOwner and BranchManager can use this endpoint.
    """
    profile = settings_manager.get_user_access_profile(current_user.id)
    is_org_owner = profile.normalized_role == "OrgOwner"
    is_branch_manager = profile.normalized_role == "BranchManager"

    # Only OrgOwner and BranchManager can perform global searches
    if not is_org_owner and not is_branch_manager:
        raise HTTPException(
            status_code=403,
            detail="Only OrgOwner and BranchManager users can perform global searches"
        )

    # Get all documents from the database (no filtering by user permissions)
    raw_documents = metadata_service.list_documents(org_id=profile.org_id)

    # Filter documents based on search query
    query_lower = request.query.lower()
    matching_documents = []

    for record in raw_documents:
        relative_path = str(record.get("relative_path") or "")
        filename = str(record.get("filename") or "")
        directory = str(record.get("directory") or "")

        # Search in filename, relative path, and directory
        if (
            query_lower in filename.lower()
            or query_lower in relative_path.lower()
            or query_lower in directory.lower()
        ):
            try:
                matching_documents.append(
                    DocumentListItem(
                        filename=filename,
                        relative_path=relative_path,
                        directory=directory,
                        chunks_indexed=int(record.get("chunks_indexed", 0)),
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Skipping document with invalid payload: %s", exc)

    return DocumentSearchResponse(
        documents=matching_documents,
        total_count=len(matching_documents),
    )
