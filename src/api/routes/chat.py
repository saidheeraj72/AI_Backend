from __future__ import annotations

import logging
import shutil
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Optional, Sequence, Union
from uuid import uuid4, UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

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
from src.services.llm import LLMService, LLM_MODELS
from src.services.metadata import MetadataService
from src.services.pdf_processor import PDFProcessor
from src.services.rag import RAGChatService
from src.services.vectorstore import VectorStoreService
from src.services.websearch import WebSearchService
from src.services.organization_service import OrganizationService
from src.utils.paths import normalize_relative_path

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


@router.post("", response_class=StreamingResponse)
async def chat(
    model_key: str = Form(...),
    prompt: Optional[str] = Form(None),
    user_id: str = Form(...),
    chat_session_id: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    files: Optional[list[UploadFile]] = File(None),
    websearch: bool = Form(False),
    current_user: SupabaseUser = Depends(require_supabase_user),
) -> StreamingResponse:
    # Validate user access immediately
    _ensure_user_access(current_user, user_id)

    # Process image if present
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

    async def event_stream():
        try:
            yield json.dumps({"type": "status", "message": "Initializing chat..."}) + "\n"

            # 1. Process uploaded documents
            if files:
                yield json.dumps({"type": "status", "message": f"Processing {len(files)} uploaded file(s)..."}) + "\n"
                
                for file in files:
                    if not file.filename.lower().endswith(".pdf"):
                        continue
                    
                    yield json.dumps({"type": "status", "message": f"Reading {file.filename}..."}) + "\n"
                    
                    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        shutil.copyfileobj(file.file, tmp)
                        tmp_path = Path(tmp.name)
                    
                    try:
                        # 1. Extract Text
                        yield json.dumps({"type": "status", "message": f"Extracting text from {file.filename}..."}) + "\n"
                        text_content = await run_in_threadpool(pdf_processor.extract_text, tmp_path)
                        
                        # 2. Extract Tables (with progress)
                        num_pages = await run_in_threadpool(pdf_processor.get_page_count, tmp_path)
                        table_content = ""
                        
                        if num_pages > 0:
                            batch_size = 5
                            for start in range(1, num_pages + 1, batch_size):
                                end = min(start + batch_size - 1, num_pages)
                                pages_range = f"{start}-{end}"
                                
                                yield json.dumps({"type": "status", "message": f"Analyzing tables ({start}-{end}/{num_pages})..."}) + "\n"
                                
                                chunk_table = await run_in_threadpool(
                                    pdf_processor.extract_tables_batch, 
                                    tmp_path, 
                                    pages_range
                                )
                                table_content += chunk_table

                        combined_content = f"{text_content}{table_content}".strip()
                        
                        if combined_content:
                            doc = Document(page_content=combined_content)
                            doc.metadata["source"] = file.filename
                            doc.metadata["user_id"] = user_id
                            doc.metadata["chat_id"] = session_id
                            
                            yield json.dumps({"type": "status", "message": f"Indexing {file.filename}..."}) + "\n"
                            
                            await run_in_threadpool(
                                vector_service.index_documents, 
                                [doc], 
                                pinecone_index_override=vector_service._pinecone_chat_index
                            )
                        else:
                             logger.warning(f"No content extracted from {file.filename}")

                    except Exception as e:
                        logger.error(f"Error processing file {file.filename}: {e}")
                        yield json.dumps({"type": "status", "message": f"Error processing {file.filename}: {str(e)}"}) + "\n"
                    finally:
                        if tmp_path.exists():
                            tmp_path.unlink()
                        await file.close()

            # 2. Build Prompt (RAG / Websearch)
            llm_prompt_parts = []
            
            if prompt:
                yield json.dumps({"type": "status", "message": "Retrieving context..."}) + "\n"
                try:
                    results = await run_in_threadpool(
                        vector_service.search,
                        query=prompt,
                        user_id=user_id,
                        chat_id=session_id,
                        top_k=5,
                        pinecone_index_override=vector_service._pinecone_chat_index,
                    )
                    
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

            if websearch and prompt:
                yield json.dumps({"type": "status", "message": "Searching the web..."}) + "\n"
                try:
                    search_summary = await run_in_threadpool(
                        websearch_service.search_and_summarize,
                        prompt,
                    )
                    if search_summary:
                        llm_prompt_parts.append(
                            f"Use the following web search summary to answer the question.\n{search_summary}"
                        )
                except Exception as e:
                    logger.error(f"Web search failed: {e}")

            if prompt:
                llm_prompt_parts.append(prompt)
            
            final_prompt = "\n\n".join(llm_prompt_parts).strip()
            if not final_prompt and not image_bytes:
                 yield json.dumps({"type": "error", "message": "Prompt or image is required"}) + "\n"
                 return

            # 3. Fetch History
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

            # 4. Stream LLM Response
            yield json.dumps({"type": "status", "message": "Generating response..."}) + "\n"
            
            full_response = ""
            try:
                # Use the new streaming method
                stream_generator = llm_service.stream_chat(
                    model_key,
                    prompt=final_prompt,
                    image_bytes=image_bytes,
                    image_mime_type=image_mime,
                    history=llm_history,
                )
                
                for chunk in stream_generator:
                    full_response += chunk
                    yield json.dumps({"type": "content", "delta": chunk}) + "\n"

            except Exception as e:
                logger.exception("Chat generation failed: %s", e)
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
                return

            # 5. Save History
            yield json.dumps({"type": "status", "message": "Finalizing..."}) + "\n"
            try:
                await run_in_threadpool(
                    chat_history_service.save_chat,
                    chat_id=session_id,
                    user_id=user_id,
                    model_key=model_key,
                    model_name=LLM_MODELS.get(model_key, {}).get("name", "Unknown"),
                    user_input=prompt,
                    response=full_response,
                    usage=None, # Usage is harder to get in stream, maybe estimate or skip
                    has_image=image_bytes is not None,
                    image_mime_type=image_mime,
                    interaction_type="standard",
                    org_id=None, 
                )
            except Exception as e:
                logger.error(f"Failed to save history: {e}")

            # 6. Final Result
            yield json.dumps({
                "type": "result",
                "chat_id": session_id,
                "model_key": model_key,
                "response": full_response,
                "sources": [] # Standard chat usually doesn't return structured sources like RAG
            }) + "\n"

        except Exception as e:
            logger.exception("Unexpected error in chat stream")
            yield json.dumps({"type": "error", "message": f"Unexpected error: {str(e)}"}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


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
        # Delete associated vectors from Pinecone chat-session index
        vectors_removed = await run_in_threadpool(
            vector_service.remove_by_chat_id,
            chat_session_id,
        )
        logger.info(
            "Deleted chat session %s for user %s. Removed %s chat history records and %s vectors.",
            chat_session_id,
            user_id,
            deleted_count,
            vectors_removed,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to delete chat session: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to delete chat session") from exc

    return {"chat_id": chat_session_id, "deleted_messages": deleted_count, "deleted_vectors": vectors_removed}


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