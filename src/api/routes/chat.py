from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Union
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from src.api.dependencies.auth import SupabaseUser, require_supabase_user
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
from src.services.document_permissions import ACCESS_RANK, DocumentPermission, DocumentPermissionsService
from src.services.vectorstore import VectorStoreService
from src.services.websearch import WebSearchService
from src.services.settings_manager import SettingsManager
from src.services.folder_permissions import FolderPermissionsService, expand_folder_permissions_to_documents
from src.utils.paths import normalize_relative_path

router = APIRouter(prefix="/chat", tags=["chat"])
rag_router = APIRouter(prefix="/rag", tags=["rag"])

settings = get_settings()
logger = logging.getLogger("ai_backend.chat")
llm_service = LLMService(settings, logger=logger)
metadata_service = MetadataService(
    settings.documents_metadata_path,
    logger=logger,
    supabase_url=settings.supabase_url,
    supabase_key=settings.supabase_key,
    supabase_table=settings.supabase_documents_table,
    default_org_id=settings.default_org_id,
)
vector_service = VectorStoreService(settings, logger=logger)
rag_service = RAGChatService(
    settings=settings,
    vector_service=vector_service,
    metadata_service=metadata_service,
    llm_service=llm_service,
    logger=logger,
)
chat_history_service = ChatHistoryService(settings, logger=logger)
websearch_service = WebSearchService(settings.serper_api_key, logger=logger)
settings_manager = SettingsManager(settings=settings, logger=logger)
doc_permissions_service = DocumentPermissionsService(settings=settings, logger=logger)
folder_permissions_service = FolderPermissionsService(settings=settings, logger=logger)

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}


def _ensure_user_access(current_user: SupabaseUser, requested_user_id: str) -> None:
    if current_user.id != requested_user_id:
        raise HTTPException(status_code=403, detail="Authenticated user mismatch")


def _user_doc_acl(
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


def _collect_accessible_documents(profile, user_id: str) -> list[dict[str, Any]]:
    documents = metadata_service.list_documents(org_id=profile.org_id)
    doc_acl = _user_doc_acl(profile, user_id, documents=documents)
    accessible: list[dict[str, Any]] = []
    is_org_owner = profile.normalized_role == "OrgOwner"
    is_branch_manager = profile.normalized_role == "BranchManager"

    for record in documents:
        doc_id = str(record.get("id") or record.get("relative_path") or "")
        if not doc_id:
            continue
        branch_id = record.get("branch_id")

        if profile.normalized_role == "superadmin" or is_org_owner:
            accessible.append(record)
            continue
        if is_branch_manager and branch_id and branch_id in profile.branch_ids:
            # BranchManager gets automatic access to all documents in their branch
            accessible.append(record)
            continue
        if doc_id in doc_acl and doc_acl[doc_id].can_view:
            accessible.append(record)
            continue
    return accessible


def _user_has_permission(profile, code: str) -> bool:
    return bool(profile.has_full_access or (hasattr(profile, "permission_codes") and code in profile.permission_codes))


@router.post("", response_model=ChatResponse)
async def chat(
    model_key: str = Form(...),
    prompt: Optional[str] = Form(None),
    user_id: str = Form(...),
    chat_session_id: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    websearch: bool = Form(False),
    current_user: SupabaseUser = Depends(require_supabase_user),
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

    if websearch and not prompt:
        raise HTTPException(status_code=400, detail="websearch requires a text prompt")

    _ensure_user_access(current_user, user_id)

    # Get user's org_id for chat history
    profile = settings_manager.get_user_access_profile(user_id)
    user_org_id = profile.org_id or settings.default_org_id

    augmented_prompt = prompt
    if websearch and prompt:
        try:
            search_summary = await run_in_threadpool(
                websearch_service.search_and_summarize,
                prompt,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Web search failed: %s", exc)
            search_summary = None

        if search_summary:
            augmented_prompt = (
                "Use the following web search summary to answer the question.\n"
                f"{search_summary}\n\n"
                "User question: "
                f"{prompt}"
            )

    try:
        result = await run_in_threadpool(
            llm_service.chat,
            model_key,
            prompt=augmented_prompt,
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
        org_id=user_org_id,
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
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> Union[ChatHistoryResponse, ChatSessionListResponse]:
    if not chat_history_service.enabled:
        raise HTTPException(status_code=503, detail="Chat history storage is not configured")

    _ensure_user_access(current_user, user_id)

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
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Union[str, int]]:
    if not chat_history_service.enabled:
        raise HTTPException(status_code=503, detail="Chat history storage is not configured")

    _ensure_user_access(current_user, user_id)

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
    session_id = request.chat_session_id or str(uuid4())

    profile = settings_manager.get_user_access_profile(request.user_id)
    accessible_documents = _collect_accessible_documents(profile, request.user_id)

    if not accessible_documents:
        raise HTTPException(status_code=403, detail="You do not have access to any documents.")

    if request.use_all:
        all_paths = [
            str(record.get("relative_path"))
            for record in accessible_documents
            if record.get("relative_path")
        ]
        if not all_paths:
            raise HTTPException(status_code=403, detail="You do not have access to any documents.")
        selected_documents = list(dict.fromkeys(all_paths))
    else:
        requested_paths = request.document_paths or []
        if not requested_paths:
            raise HTTPException(status_code=400, detail="Select at least one document or enable use_all")

        normalized_allowed = {normalize_relative_path(doc.get("relative_path") or ""): doc for doc in accessible_documents}
        allowed_paths: list[str] = []
        denied_paths: list[str] = []
        for entry in requested_paths:
            normalized = normalize_relative_path(entry)
            if normalized in normalized_allowed:
                allowed_paths.append(normalized)
            else:
                denied_paths.append(entry)

        if denied_paths:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to use the selected documents: " + ", ".join(sorted(set(denied_paths))[:5]),
            )

        if not allowed_paths:
            raise HTTPException(status_code=400, detail="No valid documents found for the selected folders.")

        selected_documents = list(dict.fromkeys(allowed_paths))

    try:
        result = await run_in_threadpool(
            rag_service.chat_with_documents,
            model_key=request.model_key,
            question=request.question,
            document_paths=selected_documents,
            top_k=request.top_k,
            document_records=accessible_documents,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("RAG chat failed: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to generate response") from exc

    # Get user's org_id for chat history
    user_org_id = profile.org_id or settings.default_org_id

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
        org_id=user_org_id,
    )

    result["chat_id"] = session_id

    return RAGChatResponse(**result)


@router.post("/rag", response_model=RAGChatResponse)
async def rag_chat(
    request: RAGChatRequest,
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> RAGChatResponse:
    _ensure_user_access(current_user, request.user_id)
    return await _handle_rag_chat(request)


@rag_router.post("", response_model=RAGChatResponse)
async def rag_chat_root(
    request: RAGChatRequest,
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> RAGChatResponse:
    _ensure_user_access(current_user, request.user_id)
    return await _handle_rag_chat(request)
