from __future__ import annotations

import logging
from typing import Any

from functools import partial
from uuid import UUID

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
    DocumentReviewRequest,
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
    # Verify user access to branch (omitted for brevity, assume handled or minimal check)
    # TODO: Strict permission check
    
    # parent_id is optional. If provided, it must exist.
    # We use the branch_id from request.
    
    result = metadata_service.create_folder(
        branch_id=str(folder.branch_id),
        name=folder.name,
        parent_id=str(folder.parent_id) if folder.parent_id else None,
        description=folder.description,
        created_by=current_user.id
    )
    
    if not result:
        raise HTTPException(status_code=400, detail="Failed to create folder")
        
    return Folder(**result)

@router.post("/upload", response_model=DocumentIngestResponse)
async def upload_document(
    files: list[UploadFile] | None = File(default=None),
    file: UploadFile | None = File(default=None),
    relative_paths: list[str] | None = Form(default=None),
    branch_id: str | None = Form(default=None),
    branch_name: str | None = Form(default=None),
    description: str | None = Form(default=None),
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> DocumentIngestResponse:
    # Verify user exists and get org/branch context
    # For simplicity, we assume the user belongs to the default org or the one implied by branch_id
    
    if not branch_id:
         # Try to find a primary branch for the user
         # We need an org_id first. Let's try to find the user's org(s).
         orgs = org_service.list_user_organizations(UUID(current_user.id))
         if not orgs:
             raise HTTPException(status_code=403, detail="User is not part of any organization")
         org_id = orgs[0].id
         
         # Get user's branches in this org
         memberships = org_service.get_user_branch_memberships(UUID(current_user.id), org_id)
         if memberships:
             branch_id = str(memberships[0].branch_id)
         else:
             raise HTTPException(status_code=400, detail="Branch ID is required and user has no default branch")
    else:
        # Verify user access to this branch
        # We need org_id for the branch to verify membership efficiently or just query branch_members
        # Let's just assume valid for now or implemented in service
        pass

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

    # TODO: Get Org ID properly from branch
    # For now passing None to ingest_uploads, it should be updated to not strictly require it or fetch it
    
    try:
        return await document_ingestor.ingest_uploads(
            uploads,
            branch_id=branch_id,
            branch_name=branch_name,
            org_id=None, # metadata_service will handle branch_id
            created_by=current_user.id,
            description=description,
        )
    except NoValidDocumentsError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "No valid PDF files were ingested",
                "failures": [failure.dict() for failure in exc.failures],
            },
        ) from exc


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    directory: str | None = Query(
        default=None, description="Filter documents by their directory path"
    ),
    branch_id: str | None = Query(default=None, description="Filter by branch"),
    status: str | None = Query(default=None, description="Filter by status (approved, pending_review, rejected)"),
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> DocumentListResponse:
    # List documents. If branch_id is provided, filter by it.
    # Else, maybe list all accessible docs?
    
    raw_documents = metadata_service.list_documents(branch_id=branch_id, status=status)
    
    # Filter by directory if provided
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
    # Permission check should go here
    
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


@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(
    request: DocumentSearchRequest,
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> DocumentSearchResponse:
    # Basic search implementation
    raw_documents = metadata_service.list_documents()
    query_lower = request.query.lower()
    matching_documents = []

    for record in raw_documents:
        if (
            query_lower in str(record.get("filename", "")).lower()
            or query_lower in str(record.get("relative_path", "")).lower()
        ):
            matching_documents.append(DocumentListItem(**record))

    return DocumentSearchResponse(
        documents=matching_documents,
        total_count=len(matching_documents),
    )


@router.patch("/{document_id}/review", response_model=dict[str, Any])
async def review_document(
    document_id: UUID,
    review: DocumentReviewRequest,
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    # Check if user has permission to review (Admin/Owner)
    orgs = org_service.list_user_organizations(UUID(current_user.id))
    if not orgs:
        raise HTTPException(status_code=403, detail="User is not part of any organization")
    org_id = orgs[0].id

    roles = org_service.get_user_roles(UUID(current_user.id), org_id)
    is_authorized = False
    for role in roles:
        if role.name in ["OrgOwner", "Admin", "BranchManager"]:
            is_authorized = True
            break
    
    if not is_authorized:
        raise HTTPException(status_code=403, detail="Insufficient permissions to review documents")

    if review.document_id != document_id:
        raise HTTPException(status_code=400, detail="Document ID mismatch")

    success = metadata_service.update_document_status(
        str(document_id), 
        review.status,
        reviewed_by=current_user.id
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update document status")

    return {"id": str(document_id), "status": review.status}


@router.get("/permissions", response_model=dict[str, Any])
async def get_my_document_permissions(
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    # Determine org context
    orgs = org_service.list_user_organizations(UUID(current_user.id))
    if not orgs:
        # If no org, default to empty/limited
        # Or use default org if configured
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
        org_id = orgs[0].id # User's first org

    # Check roles
    roles = org_service.get_user_roles(UUID(current_user.id), org_id)
    role_names = {r.name for r in roles}
    
    is_superadmin = "OrgOwner" in role_names
    # Branch Manager or Admin typically has full access to their scope.
    # For simplicity, we treat them as having "full access" flag for UI, 
    # but backend might still enforce branch scoping elsewhere.
    is_admin_or_manager = is_superadmin or "Admin" in role_names or "BranchManager" in role_names
    
    # Calculate permissions
    # If superadmin, we can skip detailed calculation or return empty dict + full_access flag
    perms = {}
    if not is_superadmin:
        perms = org_service.get_user_permissions_summary(UUID(current_user.id), org_id)

    return {
        "is_superadmin": is_superadmin,
        "has_full_access": is_admin_or_manager,
        # Simple heuristic: if they have upload perm on any folder or are admin
        "can_upload_role": is_admin_or_manager or any(p.get("can_upload") for p in perms.values()),
        "document_permissions": perms
    }