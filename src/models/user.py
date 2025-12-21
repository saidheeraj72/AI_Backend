from pydantic import BaseModel, EmailStr
from typing import Optional

class UserConfigResponse(BaseModel):
    user_id: str
    email: EmailStr
    is_superadmin: bool
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    
    # Organization Context
    organization_id: Optional[str] = None
    organization_name: Optional[str] = None
    role: Optional[str] = None
    
    # Billing & Modules (Personal)
    subscription_plan: Optional[str] = "free"
    modules: dict = {}
