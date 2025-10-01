from __future__ import annotations

import logging
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List, Sequence, Tuple

from fastapi import UploadFile

from src.core.config import Settings
from src.models.schemas import (
    DocumentIngestResponse,
    DocumentUploadError,
    DocumentUploadResult,
)
from src.services.metadata import MetadataService
from src.services.pdf_processor import PDFProcessor
from src.services.storage import StorageService
from src.services.vectorstore import VectorStoreService


class DocumentIngestorError(Exception):
    """Base exception raised during document ingestion."""


class NoValidDocumentsError(DocumentIngestorError):
    """Raised when no uploaded documents could be ingested successfully."""

    def __init__(self, failures: List[DocumentUploadError]) -> None:
        super().__init__("No valid PDF files were ingested")
        self.failures = failures


class DocumentIngestor:
    """Coordinate PDF ingestion from uploads into storage, metadata, and vector store."""

    def __init__(
        self,
        *,
        settings: Settings,
        storage_service: StorageService,
        pdf_processor: PDFProcessor,
        vector_service: VectorStoreService,
        metadata_service: MetadataService,
        logger: logging.Logger,
    ) -> None:
        self.settings = settings
        self.storage_service = storage_service
        self.pdf_processor = pdf_processor
        self.vector_service = vector_service
        self.metadata_service = metadata_service
        self.logger = logger

    async def ingest_uploads(self, uploads: Sequence[UploadFile]) -> DocumentIngestResponse:
        successes: List[DocumentUploadResult] = []
        failures: List[DocumentUploadError] = []
        total_chunks = 0

        for upload in uploads:
            try:
                file_successes, file_failures, file_chunks = await self._ingest_upload(upload)
                successes.extend(file_successes)
                failures.extend(file_failures)
                total_chunks += file_chunks
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.exception("Unexpected error while ingesting upload %s", upload.filename)
                failures.append(
                    DocumentUploadError(
                        filename=upload.filename or "unknown",
                        reason=f"Unexpected error: {exc}",
                    )
                )

        if not successes:
            raise NoValidDocumentsError(failures)

        return DocumentIngestResponse(
            successes=successes,
            failures=failures,
            total_chunks_indexed=total_chunks,
        )

    async def _ingest_upload(
        self, upload: UploadFile
    ) -> Tuple[List[DocumentUploadResult], List[DocumentUploadError], int]:
        filename = upload.filename or ""
        if not filename:
            await upload.close()
            return (
                [],
                [DocumentUploadError(filename="unknown", reason="Missing filename in upload")],
                0,
            )

        if filename.lower().endswith(".zip"):
            return await self._process_archive_upload(upload)

        result, error = await self._process_single_pdf(upload)
        successes = [result] if result else []
        failures = [error] if error else []
        chunks = result.chunks_indexed if result else 0
        return successes, failures, chunks

    async def _process_single_pdf(
        self, upload: UploadFile
    ) -> Tuple[DocumentUploadResult | None, DocumentUploadError | None]:
        try:
            saved_path = await self.storage_service.save_pdf(upload, relative_path=upload.filename)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Failed to store uploaded PDF %s", upload.filename)
            return None, DocumentUploadError(
                filename=upload.filename or "unknown",
                reason=f"Failed to store PDF: {exc}",
            )

        return self._ingest_saved_pdf(saved_path, upload.filename or saved_path.name)

    async def _process_archive_upload(
        self, upload: UploadFile
    ) -> Tuple[List[DocumentUploadResult], List[DocumentUploadError], int]:
        try:
            archive_bytes = await upload.read()
        finally:
            await upload.close()

        successes: List[DocumentUploadResult] = []
        failures: List[DocumentUploadError] = []
        chunks = 0

        try:
            with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue

                    member_name = member.filename
                    if not member_name.lower().endswith(".pdf"):
                        failures.append(
                            DocumentUploadError(
                                filename=member_name,
                                reason="Skipped non-PDF entry inside archive",
                            )
                        )
                        continue

                    try:
                        with archive.open(member) as member_file:
                            pdf_data = member_file.read()
                        saved_path = self.storage_service.save_pdf_bytes(
                            pdf_data, relative_path=member_name
                        )
                    except Exception as exc:  # pragma: no cover - defensive logging
                        self.logger.exception(
                            "Failed to extract PDF %s from archive %s", member_name, upload.filename
                        )
                        failures.append(
                            DocumentUploadError(
                                filename=member_name,
                                reason=f"Failed to extract PDF from archive: {exc}",
                            )
                        )
                        continue

                    result, error = self._ingest_saved_pdf(saved_path, member_name)
                    if result:
                        successes.append(result)
                        chunks += result.chunks_indexed
                    if error:
                        failures.append(error)
        except zipfile.BadZipFile as exc:
            self.logger.exception("Invalid ZIP archive uploaded: %s", upload.filename)
            failures.append(
                DocumentUploadError(
                    filename=upload.filename or "archive",
                    reason=f"Invalid ZIP archive: {exc}",
                )
            )

        return successes, failures, chunks

    def _ingest_saved_pdf(
        self, saved_path: Path, source_label: str
    ) -> Tuple[DocumentUploadResult | None, DocumentUploadError | None]:
        document = self.pdf_processor.process_pdf(saved_path)
        if document is None:
            return None, DocumentUploadError(
                filename=source_label,
                reason="Could not extract content from PDF",
            )

        try:
            relative_path = saved_path.relative_to(self.settings.pdf_directory)
        except ValueError:  # pragma: no cover - defensive logging
            relative_path = saved_path

        document.metadata["source"] = relative_path.as_posix()

        try:
            chunks_indexed = self.vector_service.index_documents([document])
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Failed to index PDF %s", saved_path)
            return None, DocumentUploadError(
                filename=source_label,
                reason=f"Indexing failed: {exc}",
            )

        if chunks_indexed == 0:
            return None, DocumentUploadError(
                filename=source_label,
                reason="No content was indexed from the PDF",
            )

        self.metadata_service.record_document(relative_path, chunks_indexed)

        directory = relative_path.parent.as_posix() if relative_path.parent.as_posix() != "." else ""
        result = DocumentUploadResult(
            filename=relative_path.name,
            relative_path=relative_path.as_posix(),
            directory=directory,
            chunks_indexed=chunks_indexed,
            message="PDF processed and indexed successfully",
        )

        return result, None
