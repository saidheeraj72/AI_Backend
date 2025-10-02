from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentUploadResult(BaseModel):
    filename: str
    relative_path: str
    directory: str
    chunks_indexed: int
    message: str


class DocumentUploadError(BaseModel):
    filename: str
    reason: str


class DocumentListItem(BaseModel):
    filename: str
    relative_path: str
    directory: str
    chunks_indexed: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem] = Field(default_factory=list)


class DocumentIngestResponse(BaseModel):
    successes: list[DocumentUploadResult] = Field(default_factory=list)
    failures: list[DocumentUploadError] = Field(default_factory=list)
    total_chunks_indexed: int


class ChatResponse(BaseModel):
    model_key: str
    model_name: str
    response: str
    usage: dict[str, int] | None = None
    chat_id: str | None = None


class RAGSource(BaseModel):
    filename: str
    download_url: str | None = None


class RAGChatResponse(ChatResponse):
    question: str
    sources: list[RAGSource] = Field(default_factory=list)


class RAGChatRequest(BaseModel):
    question: str
    model_key: str = Field(default="complex")
    document_paths: list[str] | None = Field(default=None, description="Relative paths of selected documents")
    use_all: bool = Field(default=False, description="Ignore document_paths and search across all documents")
    top_k: int = Field(default=4, ge=1, le=20)
    user_id: str = Field(..., description="Supabase auth identifier for the requesting user")
    chat_session_id: str | None = Field(None, description="Conversation identifier to group related messages")


class ChatHistoryMessage(BaseModel):
    chat_id: str
    user_id: str
    interaction_type: str
    model_key: str
    model_name: str
    user_input: str | None
    response: str
    usage: dict[str, Any] | None = None
    has_image: bool
    image_mime_type: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    chat_id: str
    messages: list[ChatHistoryMessage] = Field(default_factory=list)


class ChatSessionSummary(BaseModel):
    chat_id: str
    user_id: str
    message_count: int
    last_message: str | None = None
    last_user_input: str | None = None
    last_model_key: str | None = None
    last_interaction_type: str | None = None
    last_interaction_at: datetime


class ChatSessionListResponse(BaseModel):
    user_id: str
    sessions: list[ChatSessionSummary] = Field(default_factory=list)


class ChatHistoryScope(str, Enum):
    session = "session"
    sessions = "sessions"
