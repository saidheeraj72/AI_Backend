from __future__ import annotations

import logging
from typing import Optional, Union
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from src.core.config import get_settings
from src.models.schemas import (
    ChatHistoryResponse,
    ChatHistoryScope,
    ChatResponse,
    ChatSessionListResponse,
    RAGChatRequest,
    RAGChatResponse,
)
from src.services.chat_history import ChatHistoryService
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
chat_history_service = ChatHistoryService(settings, logger=logger)

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}


@router.post("", response_model=ChatResponse)
async def chat(
    model_key: str = Form(...),
    prompt: Optional[str] = Form(None),
    user_id: str = Form(...),
    chat_session_id: Optional[str] = Form(None),
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

    session_id = chat_session_id or str(uuid4())

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

    await run_in_threadpool(
        chat_history_service.save_chat,
        chat_id=session_id,
        user_id=user_id,
        model_key=model_key,
        model_name=result.get("model_name", ""),
        user_input=prompt,
        response=result.get("response", ""),
        usage=result.get("usage"),
        has_image=image_bytes is not None,
        image_mime_type=image_mime,
        interaction_type="standard",
    )

    result["chat_id"] = session_id

    return ChatResponse(**result)


@router.get(
    "/history",
    response_model=Union[ChatHistoryResponse, ChatSessionListResponse],
)
async def chat_history(
    user_id: str = Query(..., description="Supabase auth identifier for the requesting user"),
    scope: ChatHistoryScope = Query(
        ChatHistoryScope.session,
        description="Return a single session ('session') or all sessions for the user ('sessions')",
    ),
    chat_session_id: Optional[str] = Query(
        None, description="Conversation identifier to retrieve when scope=session"
    ),
    limit: Optional[int] = Query(None, ge=1, le=500, description="Optional limit for session messages"),
) -> Union[ChatHistoryResponse, ChatSessionListResponse]:
    if not chat_history_service.enabled:
        raise HTTPException(status_code=503, detail="Chat history storage is not configured")

    if scope is ChatHistoryScope.sessions:
        try:
            sessions = await run_in_threadpool(
                chat_history_service.list_user_sessions,
                user_id=user_id,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to fetch chat sessions: %s", exc)
            raise HTTPException(status_code=502, detail="Failed to fetch chat sessions") from exc

        return ChatSessionListResponse(user_id=user_id, sessions=sessions)

    if not chat_session_id:
        raise HTTPException(
            status_code=400,
            detail="chat_session_id is required when scope is 'session'",
        )

    try:
        records = await run_in_threadpool(
            chat_history_service.get_chat_history,
            chat_id=chat_session_id,
            user_id=user_id,
            limit=limit,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to fetch chat history: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to fetch chat history") from exc

    return ChatHistoryResponse(chat_id=chat_session_id, messages=records)


@router.delete("/history")
async def delete_chat_history(
    user_id: str = Query(..., description="Supabase auth identifier for the requesting user"),
    chat_session_id: str = Query(..., description="Conversation identifier to delete"),
) -> dict[str, Union[str, int]]:
    if not chat_history_service.enabled:
        raise HTTPException(status_code=503, detail="Chat history storage is not configured")

    try:
        deleted_count = await run_in_threadpool(
            chat_history_service.delete_chat_session,
            chat_id=chat_session_id,
            user_id=user_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to delete chat session: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to delete chat session") from exc

    return {"chat_id": chat_session_id, "deleted_messages": deleted_count}


async def _handle_rag_chat(request: RAGChatRequest) -> RAGChatResponse:
    selected_documents = None if request.use_all else request.document_paths
    session_id = request.chat_session_id or str(uuid4())

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

    await run_in_threadpool(
        chat_history_service.save_chat,
        chat_id=session_id,
        user_id=request.user_id,
        model_key=result.get("model_key", request.model_key),
        model_name=result.get("model_name", ""),
        user_input=result.get("question", request.question),
        response=result.get("response", ""),
        usage=result.get("usage"),
        has_image=False,
        image_mime_type=None,
        interaction_type="rag",
        metadata={
            "document_paths": selected_documents,
            "use_all": request.use_all,
            "top_k": request.top_k,
            "sources": result.get("sources", []),
        },
    )

    result["chat_id"] = session_id

    return RAGChatResponse(**result)


@router.post("/rag", response_model=RAGChatResponse)
async def rag_chat(request: RAGChatRequest) -> RAGChatResponse:
    return await _handle_rag_chat(request)


@rag_router.post("", response_model=RAGChatResponse)
async def rag_chat_root(request: RAGChatRequest) -> RAGChatResponse:
    return await _handle_rag_chat(request)
