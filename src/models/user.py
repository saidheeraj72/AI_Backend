from pydantic import BaseModel, EmailStr
from typing import Optional

class UserConfigResponse(BaseModel):
    user_id: str
    email: EmailStr
    is_superadmin: bool
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
