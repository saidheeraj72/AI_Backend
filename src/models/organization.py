from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class OrganizationDTO(BaseModel):
    id: str
    name: str
    slug: str
    domain: Optional[str] = None
    status: str
    is_organization: bool = False
    created_at: datetime
    
    # Limits
    user_limit: int = 5
    storage_limit_gb: int = 10
    
    # Billing
    subscription_plan: Optional[str] = "free"
    subscription_status: Optional[str] = "active"
    billing_cycle: Optional[str] = "monthly"
    next_billing_date: Optional[datetime] = None
    
    # Modules
    modules: Dict[str, str] = {} # e.g. {"rag_chat": "active"}

class OrganizationActionRequest(BaseModel):
    organization_id: str
    action: str  # 'approve' or 'reject'
