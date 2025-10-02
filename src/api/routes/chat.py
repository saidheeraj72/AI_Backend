from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from src.core.config import get_settings
from src.models.schemas import ChatResponse, RAGChatRequest, RAGChatResponse
from src.services.llm import LLMService
from src.services.metadata import MetadataService
from src.services.rag import RAGChatService
from src.services.vectorstore import VectorStoreService

router = APIRouter(prefix="/chat", tags=["chat"])
rag_router = APIRouter(prefix="/rag", tags=["rag"])

settings = get_settings()
logger = logging.getLogger("ai_backend.chat")
llm_service = LLMService(settings, logger=logger)
metadata_service = MetadataService(settings.documents_metadata_path, logger=logger)
vector_service = VectorStoreService(settings, logger=logger)
rag_service = RAGChatService(
    settings=settings,
    vector_service=vector_service,
    metadata_service=metadata_service,
    llm_service=llm_service,
    logger=logger,
)

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}


@router.post("", response_model=ChatResponse)
async def chat(
    model_key: str = Form(...),
    prompt: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
) -> ChatResponse:
    image_bytes: Optional[bytes] = None
    image_mime: Optional[str] = None

    if image is not None:
        image_mime = image.content_type or ""
        if image_mime not in ALLOWED_IMAGE_TYPES:
            await image.close()
            raise HTTPException(status_code=400, detail="Unsupported image type")

        image_bytes = await image.read()
        await image.close()

    try:
        result = await run_in_threadpool(
            llm_service.chat,
            model_key,
            prompt=prompt,
            image_bytes=image_bytes,
            image_mime_type=image_mime,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Chat completion failed: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to generate response") from exc

    return ChatResponse(**result)


async def _handle_rag_chat(request: RAGChatRequest) -> RAGChatResponse:
    selected_documents = None if request.use_all else request.document_paths

    try:
        result = await run_in_threadpool(
            rag_service.chat_with_documents,
            model_key=request.model_key,
            question=request.question,
            document_paths=selected_documents,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("RAG chat failed: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to generate response") from exc

    return RAGChatResponse(**result)


@router.post("/rag", response_model=RAGChatResponse)
async def rag_chat(request: RAGChatRequest) -> RAGChatResponse:
    return await _handle_rag_chat(request)


@rag_router.post("", response_model=RAGChatResponse)
async def rag_chat_root(request: RAGChatRequest) -> RAGChatResponse:
    return await _handle_rag_chat(request)
