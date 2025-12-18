from __future__ import annotations

from datetime import date, datetime, time
from enum import Enum
from typing import Any, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, EmailStr

# --- Enums ---

class RoleType(str, Enum):
    ADMIN = "Admin"
    BRANCH_MANAGER = "Branch Manager"
    VIEWER = "Viewer"

class TeamProfileStatus(str, Enum):
    ACTIVE = "Active"
    ARCHIVED = "Archived"
    ON_LEAVE = "OnLeave"

class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

# --- Base Models ---

class OrganizationBase(BaseModel):
    name: str
    slug: str
    domain: Optional[str] = None

class OrganizationCreate(OrganizationBase):
    pass

class Organization(OrganizationBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class BranchBase(BaseModel):
    name: str
    code: str
    location: str
    timezone: Optional[str] = "UTC"

class BranchCreate(BranchBase):
    organization_id: UUID

class Branch(BranchBase):
    id: UUID
    organization_id: UUID
    created_at: datetime
    updated_at: datetime
    user_count: Optional[int] = 0

    class Config:
        from_attributes = True

class Role(BaseModel):
    id: UUID
    organization_id: Optional[UUID] = None
    name: str
    permissions: dict = Field(default_factory=dict)
    is_system_role: bool
    created_at: datetime

    class Config:
        from_attributes = True

class Profile(BaseModel):
    id: UUID
    email: EmailStr
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    expiry_date: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class OrganizationMember(BaseModel):
    id: UUID
    organization_id: UUID
    user_id: UUID
    role_id: Optional[UUID] = None
    joined_at: datetime

    class Config:
        from_attributes = True

class BranchMember(BaseModel):
    id: UUID
    branch_id: UUID
    user_id: UUID
    role_id: Optional[UUID] = None
    is_primary_branch: bool
    joined_at: datetime

    class Config:
        from_attributes = True

class Group(BaseModel):
    id: UUID
    organization_id: UUID
    branch_id: Optional[UUID] = None
    name: str
    description: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

# --- Team Profiles ---

class TeamSkill(BaseModel):
    id: UUID
    skill_name: str
    proficiency: Optional[str] = None

    class Config:
        from_attributes = True

class TeamQualification(BaseModel):
    id: UUID
    degree: str
    institution: Optional[str] = None
    year_completed: Optional[int] = None

    class Config:
        from_attributes = True

class TeamExperience(BaseModel):
    id: UUID
    company: str
    role: str
    years_duration: Optional[float] = None
    description: Optional[str] = None

    class Config:
        from_attributes = True

class TeamProfileBase(BaseModel):
    employee_code: Optional[str] = None
    job_title: Optional[str] = None
    department: Optional[str] = None
    weekly_hours: Optional[float] = 40.00
    overtime_allowed: Optional[bool] = False
    daily_timings_from: Optional[time] = None
    daily_timings_to: Optional[time] = None
    timezone: Optional[str] = None
    employment_start_date: Optional[date] = None
    employment_end_date: Optional[date] = None
    billing_rate: Optional[float] = None
    billing_currency: Optional[str] = "USD"
    profile_summary: Optional[str] = None
    is_active: bool = True
    status: Optional[str] = "Active"

class TeamProfile(TeamProfileBase):
    id: UUID
    organization_id: UUID
    reporting_manager_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime
    skills: List[TeamSkill] = Field(default_factory=list)
    qualifications: List[TeamQualification] = Field(default_factory=list)
    experience: List[TeamExperience] = Field(default_factory=list)

    class Config:
        from_attributes = True

# --- DMS ---

class Folder(BaseModel):
    id: UUID
    # branch_id removed, folders are now many-to-many. 
    # We might include a list of branch_ids if needed, but for basic "Folder" object it's just the entity.
    parent_id: Optional[UUID] = None
    name: str
    description: Optional[str] = None
    created_by: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class FolderCreate(BaseModel):
    branch_ids: List[UUID]
    parent_id: Optional[UUID] = None
    parent_path: Optional[str] = None
    name: str
    description: Optional[str] = None

class Document(BaseModel):
    id: UUID
    # branch_id removed
    folder_id: Optional[UUID] = None
    owner_id: Optional[UUID] = None
    title: str
    description: Optional[str] = None
    # status removed
    storage_path: str
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class FolderPermission(BaseModel):
    id: UUID
    folder_id: UUID
    user_id: Optional[UUID] = None
    group_id: Optional[UUID] = None
    can_view: bool = True
    can_upload: bool = False
    can_edit: bool = False
    can_delete: bool = False

    class Config:
        from_attributes = True

# --- Chat ---

class ChatSession(BaseModel):
    id: UUID
    user_id: UUID
    title: Optional[str] = "New Chat"
    model: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ChatMessage(BaseModel):
    id: UUID
    session_id: UUID
    role: ChatRole
    content: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime

    class Config:
        from_attributes = True


# --- Existing RAG/API Models (Updated where needed) ---

class DocumentUploadResult(BaseModel):
    filename: str
    relative_path: str
    directory: str
    chunks_indexed: int
    message: str
    branch_ids: Optional[List[UUID]] = None
    description: Optional[str] = None


class DocumentUploadError(BaseModel):
    filename: str
    reason: str


class DocumentListItem(BaseModel):
    id: Optional[UUID] = None
    folder_id: Optional[UUID] = None
    filename: str
    relative_path: str
    directory: str
    chunks_indexed: int
    created_at: Optional[datetime] = None
    # If listing for a specific branch context, we can include it.
    # If listing all, maybe list of branches? 
    # For now, let's keep it simple.
    description: Optional[str] = None
    owner_email: Optional[str] = None
    branch_name: Optional[str] = None

class FolderResponse(BaseModel):
    id: UUID
    parent_id: Optional[UUID] = None
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    branch_ids: Optional[List[UUID]] = None

class DocumentListResponse(BaseModel):
    documents: List[DocumentListItem] = Field(default_factory=list)



class DocumentIngestResponse(BaseModel):
    successes: List[DocumentUploadResult] = Field(default_factory=list)
    failures: List[DocumentUploadError] = Field(default_factory=list)
    total_chunks_indexed: int


class ChatResponse(BaseModel):
    model_key: str
    model_name: str
    response: str
    usage: Optional[dict[str, int]] = None
    chat_id: Optional[str] = None


class RAGSource(BaseModel):
    filename: str
    download_url: Optional[str] = None


class RAGChatResponse(ChatResponse):
    question: str
    sources: List[RAGSource] = Field(default_factory=list)


class RAGChatRequest(BaseModel):
    question: str
    model_key: str = Field(default="complex")
    document_paths: Optional[List[str]] = Field(default=None, description="Relative paths of selected documents")
    use_all: bool = Field(default=False, description="Ignore document_paths and search across all documents")
    top_k: int = Field(default=4, ge=1, le=20)
    user_id: str = Field(..., description="Supabase auth identifier for the requesting user")
    chat_session_id: Optional[str] = Field(None, description="Conversation identifier to group related messages")
    branch_id: Optional[UUID] = Field(None, description="Context branch ID")


class ChatHistoryMessage(BaseModel):
    chat_id: str
    user_id: str
    interaction_type: str
    model_key: str
    model_name: str
    user_input: Optional[str]
    response: str
    usage: Optional[dict[str, Any]] = None
    has_image: bool
    image_mime_type: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    chat_id: str
    messages: List[ChatHistoryMessage] = Field(default_factory=list)


class ChatSessionSummary(BaseModel):
    chat_id: str
    user_id: str
    message_count: int
    last_message: Optional[str] = None
    last_user_input: Optional[str] = None
    last_model_key: Optional[str] = None
    last_interaction_type: Optional[str] = None
    last_interaction_at: datetime


class ChatSessionListResponse(BaseModel):
    user_id: str
    sessions: List[ChatSessionSummary] = Field(default_factory=list)


class ChatHistoryScope(str, Enum):
    session = "session"
    sessions = "sessions"


class DocumentSearchRequest(BaseModel):
    query: str = Field(..., description="Search query to match against filenames and paths")


class DocumentSearchResponse(BaseModel):
    documents: List[DocumentListItem] = Field(default_factory=list)
    total_count: int = Field(description="Total number of matching documents")

class BatchProcessRequest(BaseModel):
    paths: List[str]
    branch_ids: List[str]
    description: Optional[str] = None
    chat_id: Optional[str] = None

class UploadUrlResponse(BaseModel):
    upload_url: str
    relative_path: str
    token: Optional[str] = None