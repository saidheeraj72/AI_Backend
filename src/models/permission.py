from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class PermissionDTO(BaseModel):
    id: str
    resource_type: str # folder, document
    resource_id: str
    resource_name: Optional[str] = None
    grantee_id: str
    permission: str # view, edit, create, delete, full_access
    granted_by: Optional[str]
    created_at: datetime

class GrantPermissionRequest(BaseModel):
    resource_type: str
    resource_id: str
    grantee_id: str
    permission: str
