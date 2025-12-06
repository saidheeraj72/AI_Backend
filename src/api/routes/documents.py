from __future__ import annotations

import logging
from typing import Any, List
from pathlib import Path

from functools import partial
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool

from src.api.dependencies.auth import SupabaseUser, require_supabase_user
from src.core.config import get_settings
from src.models.schemas import (
    DocumentIngestResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentSearchRequest,
    DocumentSearchResponse,
    Folder,
    FolderCreate,
)
from src.services.document_ingestor import DocumentIngestor, NoValidDocumentsError
from src.services.metadata import MetadataService
from src.services.pdf_processor import PDFProcessor
from src.services.storage import StorageService
from src.services.vectorstore import VectorStoreService
from src.services.organization_service import OrganizationService
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
    supabase_table="documents",
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
org_service = OrganizationService(settings=settings, logger=logger)

@router.post("/folders", response_model=Folder)
async def create_folder(
    folder: FolderCreate,
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> Folder:
    """
    Create a new folder explicitly.
    """
    if not folder.branch_ids:
         raise HTTPException(status_code=400, detail="At least one branch_id is required")
    
    result = metadata_service.create_folder(
        branch_ids=[str(bid) for bid in folder.branch_ids],
        name=folder.name,
        parent_id=str(folder.parent_id) if folder.parent_id else None,
        description=folder.description,
        created_by=current_user.id
    )
    
    if not result:
        raise HTTPException(status_code=400, detail="Failed to create folder")
        
    return Folder(**result)

@router.post("/upload")
async def upload_document(
    files: list[UploadFile] | None = File(default=None),
    file: UploadFile | None = File(default=None),
    relative_paths: list[str] | None = Form(default=None),
    branch_ids: list[str] = Form(default=[]),
    branch_name: str | None = Form(default=None),
    description: str | None = Form(default=None),
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> StreamingResponse:
    """
    Upload documents and return a streaming response with progress updates (NDJSON).
    """
    if not branch_ids:
         raise HTTPException(status_code=400, detail="At least one branch ID is required")

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

    return StreamingResponse(
        document_ingestor.ingest_uploads_stream(
            uploads,
            branch_ids=branch_ids,
            branch_name=branch_name,
            org_id=None,
            created_by=current_user.id,
            description=description,
        ),
        media_type="application/x-ndjson"
    )


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    directory: str | None = Query(
        default=None, description="Filter documents by their directory path"
    ),
    branch_id: str | None = Query(default=None, description="Filter by branch context"),
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> DocumentListResponse:
    
    raw_documents = metadata_service.list_documents(branch_id=branch_id)
    
    directory_filter = normalize_relative_path(directory) if directory else None
    
    documents = []
    for record in raw_documents:
        if directory_filter and not directory_matches_filter(normalize_relative_path(record.get("directory", "")), directory_filter):
            continue
            
        documents.append(DocumentListItem(**record))

    return DocumentListResponse(documents=documents)


@router.delete("", response_model=dict[str, Any])
async def delete_document(
    relative_path: str = Query(..., description="Relative path of the document to delete"),
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
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
            partial(metadata_service.delete_document, relative_path),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "relative_path": relative_path,
        "metadata_deleted": metadata_deleted,
        "vectors_removed": vectors_removed,
        "local_deleted": local_deleted,
    }

@router.get("/download")
async def download_document(
    relative_path: str = Query(..., description="Relative path of the document to download"),
    current_user: SupabaseUser = Depends(require_supabase_user),
):
    try:
        download_source, content_type = storage_service.get_download_info(relative_path)

        if isinstance(download_source, Path):
            # Local file
            return FileResponse(
                download_source,
                media_type=content_type,
                filename=Path(relative_path).name, # Use original filename
            )
        elif isinstance(download_source, str):
            # Supabase signed URL
            return RedirectResponse(url=download_source, status_code=303)
        else:
            raise HTTPException(status_code=500, detail="Unexpected download source type")

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Error downloading document %s: %s", relative_path, exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc

@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(
    request: DocumentSearchRequest,
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> DocumentSearchResponse:
    raw_documents = metadata_service.list_documents()
    query_lower = request.query.lower()
    # Check for underscore version too (e.g. "foo bar" matches "foo_bar")
    query_normalized = query_lower.replace(" ", "_")
    
    matching_documents = []

    for record in raw_documents:
        filename = str(record.get("filename", "")).lower()
        rel_path = str(record.get("relative_path", "")).lower()
        
        if (
            query_lower in filename
            or query_lower in rel_path
            or query_normalized in filename
            or query_normalized in rel_path
        ):
            matching_documents.append(DocumentListItem(**record))

    return DocumentSearchResponse(
        documents=matching_documents,
        total_count=len(matching_documents),
    )


@router.get("/permissions", response_model=dict[str, Any])
async def get_my_document_permissions(
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    orgs = org_service.list_user_organizations(UUID(current_user.id))
    if not orgs:
        if settings.default_org_id:
             org_id = UUID(settings.default_org_id)
        else:
             return {
                "is_superadmin": False,
                "has_full_access": False,
                "can_upload_role": False,
                "document_permissions": {}
             }
    else:
        org_id = orgs[0].id

    roles = org_service.get_user_roles(UUID(current_user.id), org_id)
    role_names = {r.name for r in roles}
    
    is_superadmin = "OrgOwner" in role_names
    is_admin_or_manager = is_superadmin or "Admin" in role_names or "BranchManager" in role_names
    
    perms = {}
    if not is_superadmin:
        perms = org_service.get_user_permissions_summary(UUID(current_user.id), org_id)

    return {
        "is_superadmin": is_superadmin,
        "has_full_access": is_admin_or_manager,
        "can_upload_role": is_admin_or_manager or any(p.get("can_upload") for p in perms.values()),
        "document_permissions": perms
    }