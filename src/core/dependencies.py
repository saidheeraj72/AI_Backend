from fastapi import Header, HTTPException, status, Depends
from typing import Optional
from src.core.config import settings
from src.core.database import supabase

async def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """
    Dependency to get the current user.
    Validates the Bearer token using Supabase Auth and extracts user info.
    """
    
    # Check if Supabase client is initialized
    if not supabase:
        print("Warning: Supabase client not initialized. Cannot validate token.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection not available"
        )

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        token_type, access_token = authorization.split(" ")
        if token_type.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type, must be Bearer",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Validate token with Supabase
        user_response = supabase.auth.get_user(access_token)
        
        if not user_response or not user_response.user:
             raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        user = user_response.user
        
        return {"id": user.id, "email": user.email}
    
    except HTTPException:
        # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        print(f"Error in get_current_user: {e}")
        # Supabase client might raise exceptions for invalid tokens
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_org_admin(current_user: dict = Depends(get_current_user)) -> str:
    """
    Dependency to ensure the user is an Organization Admin.
    Returns the organization_id if authorized.
    """
    user_id = current_user.get("id")
    
    try:
        # Check organization_members for org_admin role
        # We assume a user can be admin of only ONE organization for now in this context,
        # or we return the first one found.
        response = supabase.table("organization_members")\
            .select("organization_id")\
            .eq("user_id", user_id)\
            .eq("role", "org_admin")\
            .maybe_single()\
            .execute()
            
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not an Organization Admin"
            )
            
        return response.data['organization_id']
        
    except Exception as e:
        print(f"Error checking org admin status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify organization privileges"
        )
