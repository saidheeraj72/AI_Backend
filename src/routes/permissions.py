from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from src.core.dependencies import get_current_org_admin, get_current_user
from src.services.permission_service import PermissionService
from src.models.permission import PermissionDTO, GrantPermissionRequest

router = APIRouter()

@router.get("/user/{user_id}", response_model=List[PermissionDTO])
async def get_user_permissions(
    user_id: str,
    org_id: str = Depends(get_current_org_admin)
):
    """
    Get all permissions for a specific user.
    """
    try:
        # TODO: Verify user belongs to org_id
        return await PermissionService.get_user_permissions(user_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/", status_code=status.HTTP_201_CREATED)
async def grant_permission(
    data: GrantPermissionRequest,
    current_user: dict = Depends(get_current_user),
    org_id: str = Depends(get_current_org_admin)
):
    """
    Grant a permission to a user.
    """
    try:
        # Service logic fetches org_id from resource. 
        # We could add a check here to verify resource.org_id == org_id
        await PermissionService.grant_permission(data, current_user['id'])
        return {"message": "Permission granted"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.delete("/{permission_id}", status_code=status.HTTP_200_OK)
async def revoke_permission(
    permission_id: str,
    org_id: str = Depends(get_current_org_admin)
):
    """
    Revoke a permission.
    """
    try:
        await PermissionService.revoke_permission(permission_id)
        return {"message": "Permission revoked"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
