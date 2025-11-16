from __future__ import annotations

import logging
from typing import Any

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
from src.services.document_access import build_document_access_snapshot
from src.services.document_permissions import DocumentPermissionsService
from src.services.metadata import MetadataService
from src.services.pdf_processor import PDFProcessor
from src.services.storage import StorageService
from src.services.vectorstore import VectorStoreService
from src.services.settings_manager import SettingsManager
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


@router.post("/upload", response_model=DocumentIngestResponse)
async def upload_document(
    files: list[UploadFile] | None = File(default=None),
    file: UploadFile | None = File(default=None),
    relative_paths: list[str] | None = Form(default=None),
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> DocumentIngestResponse:
    # Check user permissions
    profile = settings_manager.get_user_access_profile(current_user.id)

    # Check if user has upload permission (admin or superadmin have automatic upload rights)
    normalized_role = profile.normalized_role.lower()
    can_upload_role = normalized_role in {"admin", "superadmin"}

    # For non-admin/superadmin users, check document-level permissions
    if not can_upload_role:
        # User role - check document-level permissions for each file
        pass  # Will be checked per-file below

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

    # Check permissions for each file
    is_superadmin = profile.normalized_role == "superadmin"
    is_admin = profile.normalized_role == "admin"

    # For regular users (not admin/superadmin), check document permissions
    if not is_admin and not is_superadmin:
        for upload in uploads:
            filename = str(upload.filename or "")

            # Check per-document upload permission for regular users
            can_upload = doc_permissions_service.user_can_upload_document(
                current_user.id, filename, is_superadmin=False
            )
            if not can_upload:
                raise HTTPException(
                    status_code=403,
                    detail=f"You do not have upload permission for document '{filename}'. Contact your administrator to grant access."
                )

    try:
        return await document_ingestor.ingest_uploads(uploads)
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
        "has_full_access": profile.has_full_access,
        "direct_group_ids": list(profile.direct_group_ids),
        "accessible_group_ids": list(profile.accessible_group_ids),
        "folder_paths": list(profile.folder_paths),
        "can_upload": profile.normalized_role in {"admin", "superadmin"},
    }


@router.get("/permissions", response_model=dict[str, Any])
async def get_user_document_permissions(
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    """Get all document-level permissions for the current user"""
    profile = settings_manager.get_user_access_profile(current_user.id)
    is_superadmin = profile.normalized_role == "superadmin"

    # Superadmins have full access to everything
    if is_superadmin or profile.has_full_access:
        return {
            "user_id": current_user.id,
            "is_superadmin": is_superadmin,
            "has_full_access": True,
            "document_permissions": {},
        }

    # Get all document permissions for this user
    permissions = doc_permissions_service.get_all_user_document_permissions(current_user.id)

    # Convert to a dictionary for easy frontend lookup
    permissions_map = {}
    for perm in permissions:
        permissions_map[perm.document_path] = {
            "can_upload": perm.can_upload,
            "can_view": perm.can_view,
            "can_delete": perm.can_delete,
        }

    return {
        "user_id": current_user.id,
        "is_superadmin": False,
        "has_full_access": False,
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

    raw_documents = metadata_service.list_documents()

    profile = settings_manager.get_user_access_profile(current_user.id)
    is_superadmin = profile.normalized_role == "superadmin"

    # Log user permissions for debugging
    logger.info(
        f"User {current_user.id} permissions - Role: {profile.role}, "
        f"Has full access: {profile.has_full_access}, "
        f"Folder paths: {list(profile.folder_paths)}"
    )

    is_admin = profile.normalized_role == "admin"

    # For superadmin and admin, use group-based access snapshot
    # For regular users, bypass access snapshot and check document permissions instead
    if is_superadmin or is_admin:
        access_snapshot = build_document_access_snapshot(raw_documents, profile)
        documents_to_check = access_snapshot.documents
    else:
        # Regular users: don't filter by groups, check all documents with document_permissions
        documents_to_check = [dict(item) for item in raw_documents]

    accessible_documents = []

    for record in documents_to_check:
        directory = normalize_relative_path(str(record.get("directory", "")))
        if directory_filter is not None and not directory_matches_filter(directory, directory_filter):
            continue
        relative_path_value = str(record.get("relative_path") or "")
        if not relative_path_value:
            continue

        # Check per-document view permission (folder-based)
        # For regular users (not admin/superadmin), check document permissions
        if not is_superadmin and not is_admin:
            can_view = doc_permissions_service.user_can_view_document(
                current_user.id, relative_path_value, is_superadmin=False
            )
            if not can_view:
                logger.debug(f"User {current_user.id} denied view access to {relative_path_value}")
                continue

        try:
            accessible_documents.append(
                DocumentListItem(
                    filename=str(record.get("filename") or ""),
                    relative_path=relative_path_value,
                    directory=str(record.get("directory") or ""),
                    chunks_indexed=int(record.get("chunks_indexed", 0)),
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

    document = await run_in_threadpool(metadata_service.get_document, relative_path)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    profile = settings_manager.get_user_access_profile(current_user.id)
    is_superadmin = profile.normalized_role == "superadmin"
    is_admin = profile.normalized_role == "admin"

    # For regular users (not admin/superadmin), check per-document delete permission
    if not is_superadmin and not is_admin:
        can_delete = doc_permissions_service.user_can_delete_document(
            current_user.id, relative_path, is_superadmin=False
        )
        if not can_delete:
            raise HTTPException(
                status_code=403,
                detail="You do not have delete permission for this document"
            )

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
            metadata_service.delete_document,
            relative_path,
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
    is_superadmin = profile.normalized_role == "superadmin"
    is_admin = profile.normalized_role == "admin"

    # Only admin and superadmin can delete entire directories
    if not is_admin and not is_superadmin:
        raise HTTPException(status_code=403, detail="Only admin and superadmin users can delete directories")

    normalized_directory = normalize_relative_path(directory)
    documents = [DocumentListItem(**item) for item in metadata_service.list_documents()]

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
                metadata_service.delete_document,
                relative_path,
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
    Only superadmins and admins can use this endpoint.
    """
    profile = settings_manager.get_user_access_profile(current_user.id)
    is_superadmin = profile.normalized_role == "superadmin"
    is_admin = profile.normalized_role == "admin"

    # Only admin and superadmin can perform global searches
    if not is_admin and not is_superadmin:
        raise HTTPException(
            status_code=403,
            detail="Only admin and superadmin users can perform global searches"
        )

    # Get all documents from the database (no filtering by user permissions)
    raw_documents = metadata_service.list_documents()

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
