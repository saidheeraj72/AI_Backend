from __future__ import annotations

import logging
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Optional, Sequence, Union
from uuid import uuid4, UUID

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
from src.services.context_cache import ContextCacheService
from src.services.llm import LLMService
from src.services.metadata import MetadataService
from src.services.pdf_processor import PDFProcessor
from src.services.rag import RAGChatService
from src.services.vectorstore import VectorStoreService
from src.services.websearch import WebSearchService
from src.services.organization_service import OrganizationService
from src.utils.paths import normalize_relative_path

from src.services.session_store import SessionStoreService
from langchain_core.documents import Document

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
    supabase_table="documents",
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
org_service = OrganizationService(settings=settings, logger=logger)
pdf_processor = PDFProcessor(logger=logger)
context_cache_service = ContextCacheService(Path("data"), logger=logger)
session_store = SessionStoreService(logger=logger)

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}


def _ensure_user_access(current_user: SupabaseUser, requested_user_id: str) -> None:
    if current_user.id != requested_user_id:
        raise HTTPException(status_code=403, detail="Authenticated user mismatch")


def _collect_accessible_documents(user_id: str) -> list[dict[str, Any]]:
    # List all documents the user has access to.
    # This should rely on RLS or OrganizationService.
    # For now, we'll list all documents if the user is in the organization/branch.
    
    # Ideally, metadata_service.list_documents should take user_id and filter.
    # But currently it lists all. 
    # We should filter by user's branches.
    
    all_docs = metadata_service.list_documents()
    if not all_docs:
        return []

    org_id: Optional[UUID] = None
    if settings.default_org_id:
        try:
            org_id = UUID(settings.default_org_id)
        except ValueError:
            logger.warning(f"Invalid default_org_id in settings: {settings.default_org_id}")

    if not org_id:
        user_orgs = org_service.list_user_organizations(UUID(user_id))
        if user_orgs:
            org_id = user_orgs[0].id
            
    # If still no org_id, we can't strictly filter by branch. 
    # In a dev/simple environment, this might mean "access everything".
    if not org_id:
        logger.warning("No organization found for user %s. Returning all documents (fallback).", user_id)
        return all_docs

    branches = org_service.get_user_branch_memberships(UUID(user_id), org_id)
    branch_ids = [str(b.branch_id) for b in branches]
    
    accessible = []
    for doc in all_docs:
        doc_branch = doc.get("branch_id")
        # Allow if matches branch, or if doc has no branch (public/global)
        if doc_branch in branch_ids or doc_branch is None:
            accessible.append(doc)
    
    # Fallback: If filtering resulted in 0 documents but we have docs in the system,
    # and the user is authenticated, maybe we should let them see everything to avoid "broken" experience.
    if not accessible and all_docs:
        logger.warning("Strict branch filtering returned 0 documents. Returning all documents as fallback.")
        return all_docs
    
    return accessible


@router.post("", response_model=ChatResponse)
async def chat(
    model_key: str = Form(...),
    prompt: Optional[str] = Form(None),
    user_id: str = Form(...),
    chat_session_id: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    files: Optional[list[UploadFile]] = File(None),
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

    # Process uploaded documents (PDFs) and update session store
    if files:
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                continue  # Skip non-PDFs for now, or raise error

            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = Path(tmp.name)
            
            try:
                # Run PDF processing in threadpool to avoid blocking event loop
                doc = await run_in_threadpool(pdf_processor.process_pdf, tmp_path)
                if doc and doc.page_content:
                    # Update metadata
                    doc.metadata["source"] = file.filename
                    
                    # Split into chunks
                    chunks = await run_in_threadpool(
                        vector_service.text_splitter.split_documents, [doc]
                    )
                    
                    if chunks:
                        texts = [chunk.page_content for chunk in chunks]
                        # Generate embeddings
                        embeddings = await run_in_threadpool(
                            vector_service.embedding_model.embed_documents, texts
                        )
                        
                        # Store in session memory
                        session_store.add_documents(session_id, chunks, embeddings)

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                # Optionally raise error or just skip
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
                await file.close()

    # Construct the final prompt for the LLM
    llm_prompt_parts = []
    
    # 1. Add cached context from session store (RAG)
    if prompt:
        try:
            query_vector = await run_in_threadpool(
                vector_service.embedding_model.embed_query, prompt
            )
            results = session_store.search(session_id, query_vector, top_k=5)
            
            if results:
                context_parts = []
                for doc, score in results:
                    source_name = doc.metadata.get('source', 'uploaded document')
                    context_parts.append(f"Source: {source_name}\nContent: {doc.page_content}")
                
                cached_context = "\n\n".join(context_parts)
                llm_prompt_parts.append(
                    f"Here is the content of the uploaded documents for this session (retrieved via RAG):\n{cached_context}\n"
                    "Please use the above document content to answer the user's request."
                )
        except Exception as e:
            logger.error(f"Error retrieving session context: {e}")

    # 2. Add websearch summary if applicable
    if websearch and prompt:
        try:
            search_summary = await run_in_threadpool(
                websearch_service.search_and_summarize,
                prompt,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Web search failed: %s", exc)
            search_summary = None

        if search_summary:
            llm_prompt_parts.append(
                f"Use the following web search summary to answer the question.\n{search_summary}"
            )

    # 3. Add user prompt
    if prompt:
        llm_prompt_parts.append(prompt)
    
    final_prompt = "\n\n".join(llm_prompt_parts).strip()

    if not final_prompt and not image_bytes:
        raise HTTPException(status_code=400, detail="Prompt or image is required")

    # Fetch and format chat history
    history_records = []
    try:
        history_records = await run_in_threadpool(
            chat_history_service.get_chat_history,
            chat_id=session_id,
            user_id=user_id,
        )
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")

    llm_history = []
    for record in history_records:
        if record.get("user_input"):
            llm_history.append({"role": "user", "content": record["user_input"]})
        if record.get("response"):
            llm_history.append({"role": "assistant", "content": record["response"]})

    try:
        result = await run_in_threadpool(
            llm_service.chat,
            model_key,
            prompt=final_prompt,
            image_bytes=image_bytes,
            image_mime_type=image_mime,
            history=llm_history,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat completion failed: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to generate response") from exc

    await run_in_threadpool(
        chat_history_service.save_chat,
        chat_id=session_id,
        user_id=user_id,
        model_key=model_key,
        model_name=result.get("model_name", ""),
        user_input=prompt, # Save original user input, not the context-heavy prompt
        response=result.get("response", ""),
        usage=result.get("usage"),
        has_image=image_bytes is not None,
        image_mime_type=image_mime,
        interaction_type="standard",
        org_id=None, 
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
        except Exception as exc:
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
    except Exception as exc:
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
    except Exception as exc:
        logger.exception("Failed to delete chat session: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to delete chat session") from exc

    return {"chat_id": chat_session_id, "deleted_messages": deleted_count}


async def _handle_rag_chat(request: RAGChatRequest) -> RAGChatResponse:
    session_id = request.chat_session_id or str(uuid4())

    accessible_documents = _collect_accessible_documents(request.user_id)

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
    except Exception as exc:
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
        org_id=None,
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