from __future__ import annotations

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


class RAGSource(BaseModel):
    document: str
    score: float
    snippet: str


class RAGChatResponse(ChatResponse):
    question: str
    sources: list[RAGSource] = Field(default_factory=list)


class RAGChatRequest(BaseModel):
    question: str
    model_key: str = Field(default="complex")
    document_paths: list[str] | None = Field(default=None, description="Relative paths of selected documents")
    use_all: bool = Field(default=False, description="Ignore document_paths and search across all documents")
    top_k: int = Field(default=4, ge=1, le=20)
