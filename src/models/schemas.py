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
