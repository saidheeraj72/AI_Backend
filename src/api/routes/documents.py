from __future__ import annotations

import logging
from fastapi import APIRouter, File, HTTPException, UploadFile

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

router = APIRouter(prefix="/documents", tags=["documents"])

settings = get_settings()
logger = logging.getLogger("ai_backend.documents")
pdf_processor = PDFProcessor(logger=logger)
vector_service = VectorStoreService(settings, logger=logger)
storage_service = StorageService(settings.pdf_directory, logger=logger)
metadata_service = MetadataService(settings.documents_metadata_path, logger=logger)
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
) -> DocumentIngestResponse:
    uploads: list[UploadFile] = []
    if files:
        uploads.extend(files)
    if file is not None:
        uploads.append(file)

    if not uploads:
        raise HTTPException(status_code=400, detail="At least one file must be provided")

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
async def list_documents() -> DocumentListResponse:
    documents = [DocumentListItem(**item) for item in metadata_service.list_documents()]
    return DocumentListResponse(documents=documents)
