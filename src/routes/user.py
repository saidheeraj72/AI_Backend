from fastapi import APIRouter, Depends, HTTPException, status
from src.models.user import UserConfigResponse
from src.services.user_service import UserService
from src.core.dependencies import get_current_user # Assuming a dependency to get current user from token
from supabase import Client

# In a real app, you'd inject Supabase client or a user_id from auth system
# For now, we'll assume get_current_user provides a user_id or similar.

router = APIRouter()

@router.get("/user-config", response_model=UserConfigResponse)
async def get_user_config_endpoint(current_user: dict = Depends(get_current_user)):
    """
    Retrieves configuration for the current user, including superadmin status.
    """
    user_id = current_user.get("id") # Assuming 'id' is in the current_user dict
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials: User ID missing"
        )
    try:
        user_config = await UserService.get_user_config(user_id)
        return user_config
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
