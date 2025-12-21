from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ModuleDTO(BaseModel):
    id: str
    name: str
    slug: str
    description: Optional[str] = None
    is_active: bool

class OrganizationModuleDTO(BaseModel):
    id: str
    organization_id: str
    module_slug: str
    status: str  # active, requested, disabled
    requested_at: Optional[datetime] = None
    activated_at: Optional[datetime] = None
    updated_at: datetime
    
class UpdateModuleStatusRequest(BaseModel):
    module_slug: str
    status: str # active, disabled
