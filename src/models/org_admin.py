from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

class BranchDTO(BaseModel):
    id: str
    name: str
    code: Optional[str] = None
    location: Optional[str] = None
    is_active: bool
    created_at: datetime
    # derived stats
    user_count: int = 0
    storage_usage: int = 0

class CreateBranchRequest(BaseModel):
    name: str
    code: Optional[str] = None
    location: Optional[str] = None

class OrgUserDTO(BaseModel):
    id: str
    email: Optional[EmailStr]
    full_name: Optional[str]
    avatar_url: Optional[str]
    role: str
    status: str
    joined_at: datetime
    branch_id: Optional[str] = None

class InviteUserRequest(BaseModel):
    email: EmailStr
    role: str = "staff" # org_admin, branch_manager, staff
    full_name: Optional[str] = None
    branch_id: Optional[str] = None
