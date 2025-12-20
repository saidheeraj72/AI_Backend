from __future__ import annotations

import logging
import zipfile
import json
import asyncio
from io import BytesIO
from pathlib import Path
from typing import List, Sequence, Tuple, AsyncGenerator

from fastapi import UploadFile

from src.core.config import Settings
from src.models.schemas import (
    DocumentIngestResponse,
    DocumentUploadError,
    DocumentUploadResult,
)
from src.services.metadata import MetadataService
from src.services.pdf_processor import PDFProcessor
from src.services.storage import StorageService, StoredDocument
from src.services.vectorstore import VectorStoreService


class DocumentIngestorError(Exception):
    """Base exception raised during document ingestion."""


class NoValidDocumentsError(DocumentIngestorError):
    """Raised when no uploaded documents could be ingested successfully."""

    def __init__(self, failures: List[DocumentUploadError]) -> None:
        super().__init__("No valid PDF files were ingested")
        self.failures = failures


from src.services.socket_manager import manager

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

    async def ingest_paths_background(
        self,
        paths: List[str],
        branch_ids: List[str],
        branch_name: str | None = None,
        *,
        org_id: str | None = None,
        created_by: str | None = None,
        description: str | None = None,
        user_id: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        """
        Background task to ingest files and broadcast progress via WebSocket.
        """
        if not user_id:
            self.logger.warning("No user_id provided for background ingestion; cannot send notifications.")
            return

        for path_str in paths:
            filename = Path(path_str).name
            await manager.send_personal_message(json.dumps({"type": "start", "file": filename}), user_id)
            
            local_path = None
            try:
                # 1. Download/Prepare File
                await manager.send_personal_message(json.dumps({"type": "progress", "file": filename, "step": "downloading", "message": "Retrieving file..."}), user_id)
                
                local_path = await asyncio.to_thread(self.storage_service.download_to_temp, path_str)
                relative_path = Path(path_str)

                # 2. Process PDF (Extract Text)
                await manager.send_personal_message(json.dumps({"type": "progress", "file": filename, "step": "processing", "message": "Extracting text..."}), user_id)
                document = await asyncio.to_thread(self.pdf_processor.process_pdf, local_path)
                
                if document is None:
                    await manager.send_personal_message(json.dumps({"type": "error", "file": filename, "error": "Could not extract content"}), user_id)
                    self.storage_service.cleanup_local_path(local_path, allow_dir_cleanup=False)
                    continue

                document.metadata["source"] = relative_path.as_posix()
                if user_id:
                    document.metadata["user_id"] = user_id
                if chat_id:
                    document.metadata["chat_id"] = chat_id

                pinecone_index_to_use = None
                if chat_id:
                    pinecone_index_to_use = self.vector_service._pinecone_chat_index

                # 3. Index Vectors
                await manager.send_personal_message(json.dumps({"type": "progress", "file": filename, "step": "indexing", "message": "Generating embeddings..."}), user_id)
                chunks_indexed = await asyncio.to_thread(
                    self.vector_service.index_documents, 
                    [document], 
                    pinecone_index_override=pinecone_index_to_use
                )
                
                if chunks_indexed == 0:
                    await manager.send_personal_message(json.dumps({"type": "error", "file": filename, "error": "No content indexed"}), user_id)
                    self.storage_service.cleanup_local_path(local_path, allow_dir_cleanup=False)
                    continue

                # 4. Metadata Record
                await manager.send_personal_message(json.dumps({"type": "progress", "file": filename, "step": "recording", "message": "Saving metadata..."}), user_id)
                await asyncio.to_thread(
                    self.metadata_service.record_document,
                    relative_path,
                    chunks_indexed,
                    branch_ids=branch_ids,
                    branch_name=branch_name,
                    org_id=org_id,
                    created_by=created_by,
                    description=description,
                )

                # 5. Success
                result = {
                    "filename": relative_path.name,
                    "relative_path": relative_path.as_posix(),
                    "chunks_indexed": chunks_indexed,
                    "branch_ids": branch_ids
                }
                await manager.send_personal_message(json.dumps({"type": "complete", "file": filename, "result": result}), user_id)
                
                self.storage_service.cleanup_local_path(local_path, allow_dir_cleanup=False)

            except Exception as e:
                self.logger.exception("Error processing file path %s", path_str)
                await manager.send_personal_message(json.dumps({"type": "error", "file": filename, "error": str(e)}), user_id)
                if local_path:
                    self.storage_service.cleanup_local_path(local_path, allow_dir_cleanup=False)

    async def ingest_uploads_stream(
        self,
        uploads: Sequence[UploadFile],
        branch_ids: List[str],
        branch_name: str | None = None,
        *,
        org_id: str | None = None,
        created_by: str | None = None,
        description: str | None = None,
        user_id: str | None = None,  # New parameter
        chat_id: str | None = None,  # New parameter
    ) -> AsyncGenerator[str, None]:
        """
        Ingest uploads and yield progress events as JSON strings.
        """
        for upload in uploads:
            filename = upload.filename or "unknown"
            yield json.dumps({"type": "start", "file": filename}) + "\n"
            
            try:
                # 1. Save File
                yield json.dumps({"type": "progress", "file": filename, "step": "saving", "message": "Saving file..."}) + "\n"
                
                stored = await self.storage_service.save_pdf(upload, relative_path=filename)
                relative_path = stored.relative_path
                
                try:
                    # 2. Process PDF (Extract Text)
                    yield json.dumps({"type": "progress", "file": filename, "step": "processing", "message": "Extracting text..."}) + "\n"
                    document = self.pdf_processor.process_pdf(stored.local_path)
                    
                    if document is None:
                        yield json.dumps({"type": "error", "file": filename, "error": "Could not extract content"}) + "\n"
                        continue

                    document.metadata["source"] = relative_path.as_posix()
                    if user_id:
                        document.metadata["user_id"] = user_id
                    if chat_id:
                        document.metadata["chat_id"] = chat_id

                    # Determine which Pinecone index to use
                    pinecone_index_to_use = None
                    if chat_id:
                        pinecone_index_to_use = self.vector_service._pinecone_chat_index

                    # 3. Index Vectors
                    yield json.dumps({"type": "progress", "file": filename, "step": "indexing", "message": "Generating embeddings..."}) + "\n"
                    chunks_indexed = self.vector_service.index_documents([document], pinecone_index_override=pinecone_index_to_use)
                    
                    if chunks_indexed == 0:
                        yield json.dumps({"type": "error", "file": filename, "error": "No content indexed"}) + "\n"
                        continue

                    # 4. Metadata Record
                    yield json.dumps({"type": "progress", "file": filename, "step": "recording", "message": "Saving metadata..."}) + "\n"
                    self.metadata_service.record_document(
                        relative_path,
                        chunks_indexed,
                        branch_ids=branch_ids,
                        branch_name=branch_name,
                        org_id=org_id,
                        created_by=created_by,
                        description=description,
                    )

                    # 5. Success
                    result = {
                        "filename": relative_path.name,
                        "relative_path": relative_path.as_posix(),
                        "chunks_indexed": chunks_indexed,
                        "branch_ids": branch_ids
                    }
                    yield json.dumps({"type": "complete", "file": filename, "result": result}) + "\n"

                except Exception as e:
                    self.logger.exception("Error processing file %s", filename)
                    yield json.dumps({"type": "error", "file": filename, "error": str(e)}) + "\n"
                finally:
                    if stored.should_cleanup:
                        self.storage_service.cleanup_local_path(stored.local_path)

            except Exception as e:
                self.logger.exception("Error saving file %s", filename)
                yield json.dumps({"type": "error", "file": filename, "error": str(e)}) + "\n"

    async def ingest_uploads(
        self,
        uploads: Sequence[UploadFile],
        branch_ids: List[str],
        branch_name: str | None = None, 
        *,
        org_id: str | None = None,
        created_by: str | None = None,
        description: str | None = None,
        user_id: str | None = None, # New parameter
        chat_id: str | None = None, # New parameter
    ) -> DocumentIngestResponse:
        """
        Legacy non-streaming method (wraps the stream or implements logic directly).
        For backward compatibility if needed, or we can just deprecate.
        Actually, we can reuse `ingest_uploads_stream` but just collect results.
        """
        successes: List[DocumentUploadResult] = []
        failures: List[DocumentUploadError] = []
        total_chunks = 0

        async for line in self.ingest_uploads_stream(
            uploads, 
            branch_ids, 
            branch_name, 
            org_id=org_id, 
            created_by=created_by, 
            description=description,
            user_id=user_id, # Pass new parameter
            chat_id=chat_id, # Pass new parameter
        ):
            event = json.loads(line)
            if event["type"] == "complete":
                res = event["result"]
                successes.append(DocumentUploadResult(
                    filename=res["filename"],
                    relative_path=res["relative_path"],
                    directory=Path(res["relative_path"]).parent.as_posix(),
                    chunks_indexed=res["chunks_indexed"],
                    message="Success",
                    branch_ids=[branch_ids] if isinstance(branch_ids, str) else branch_ids, # Simplification
                    description=description
                ))
                total_chunks += res["chunks_indexed"]
            elif event["type"] == "error":
                failures.append(DocumentUploadError(
                    filename=event["file"],
                    reason=event["error"]
                ))

        if not successes and failures:
             raise NoValidDocumentsError(failures)

        return DocumentIngestResponse(
            successes=successes,
            failures=failures,
            total_chunks_indexed=total_chunks,
        )

    # Keeping _ingest_saved_pdf etc as internal helpers if needed, but `ingest_uploads_stream` inlined most logic for clarity of yielding.
    # Actually, `ingest_uploads_stream` above reimplements logic. This is cleaner for yielding.
    # I will keep the old methods if they are used elsewhere (they aren't really).