from __future__ import annotations

import logging
from typing import Any
from pathlib import PurePosixPath

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from src.api.dependencies.auth import require_supabase_user
from src.core.config import get_settings
from src.models.schemas import (
    DocumentIngestResponse,
    DocumentListItem,
    DocumentListResponse,
)
from src.services.document_ingestor import DocumentIngestor, NoValidDocumentsError
from src.services.metadata import MetadataService
from src.services.pdf_processor import PDFProcessor
from src.services.storage import StorageService
from src.services.vectorstore import VectorStoreService

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(require_supabase_user)],
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


@router.post("/upload", response_model=DocumentIngestResponse)
async def upload_document(
    files: list[UploadFile] | None = File(default=None),
    file: UploadFile | None = File(default=None),
    relative_paths: list[str] | None = Form(default=None),
) -> DocumentIngestResponse:
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
            normalized = _normalize_relative_path(provided_path)
            sanitized_paths.append(normalized)

        for upload, normalized in zip(uploads, sanitized_paths, strict=False):
            if normalized:
                upload.filename = normalized

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


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    directory: str | None = Query(
        default=None, description="Filter documents by their directory path"
    ),
) -> DocumentListResponse:
    directory_filter = _normalize_relative_path(directory) if directory else None

    documents = [DocumentListItem(**item) for item in metadata_service.list_documents()]

    if directory_filter is not None:
        documents = [
            document
            for document in documents
            if _directory_matches_filter(_normalize_relative_path(document.directory), directory_filter)
        ]

    return DocumentListResponse(documents=documents)


@router.delete("", response_model=dict[str, Any])
async def delete_document(
    relative_path: str = Query(..., description="Relative path of the document to delete"),
) -> dict[str, Any]:
    if not relative_path:
        raise HTTPException(status_code=400, detail="relative_path must be provided")

    document = await run_in_threadpool(metadata_service.get_document, relative_path)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

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
) -> dict[str, Any]:
    normalized_directory = _normalize_relative_path(directory)
    documents = [DocumentListItem(**item) for item in metadata_service.list_documents()]

    target_paths = [
        document.relative_path
        for document in documents
        if _directory_matches_filter(
            _normalize_relative_path(document.directory),
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


def _normalize_relative_path(path: str | None) -> str:
    if not path:
        return ""
    parts = [
        segment
        for segment in PurePosixPath(path).parts
        if segment not in {"", ".", ".."}
    ]
    return "/".join(parts)


def _directory_matches_filter(document_directory: str, directory_filter: str) -> bool:
    if directory_filter == "":
        return document_directory == ""
    if document_directory == directory_filter:
        return True
    return document_directory.startswith(f"{directory_filter}/")
