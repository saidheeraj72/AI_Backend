from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from src.models.organization import OrganizationDTO, OrganizationActionRequest
from src.services.organization_service import OrganizationService
from src.core.dependencies import get_current_user # Assuming dependency for current user
from src.services.user_service import UserService # To check superadmin status

router = APIRouter()

async def get_superadmin_user(current_user: dict = Depends(get_current_user)):
    """
    Dependency to ensure the current user is a superadmin.
    """
    user_id = current_user.get("id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials: User ID missing"
        )
    
    try:
        user_config = await UserService.get_user_config(user_id)
        if not user_config.is_superadmin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access superadmin resources"
            )
        return user_config
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/pending-organizations", response_model=List[OrganizationDTO])
async def get_pending_organizations_endpoint(superadmin_user: dict = Depends(get_superadmin_user)):
    """
    Retrieves a list of organizations with 'pending_approval' status.
    Accessible only by superadmins.
    """
    try:
        pending_orgs = await OrganizationService.get_pending_organizations()
        return pending_orgs
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/organizations", response_model=List[OrganizationDTO])
async def get_all_organizations_endpoint(superadmin_user: dict = Depends(get_superadmin_user)):
    """
    Retrieves a list of all established organizations (active, suspended, etc.).
    Accessible only by superadmins.
    """
    try:
        orgs = await OrganizationService.get_all_organizations()
        return orgs
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/organizations/{organization_id}/action", status_code=status.HTTP_200_OK)
async def organization_action_endpoint(
    organization_id: str,
    action_request: OrganizationActionRequest,
    superadmin_user: dict = Depends(get_superadmin_user)
):
    """
    Accepts or rejects an organization.
    'action' can be 'approve' (sets status to 'active') or 'reject' (sets status to 'suspended').
    Accessible only by superadmins.
    """
    if action_request.organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization ID in path and body do not match."
        )
    if action_request.action not in ["approve", "reject"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid action. Must be 'approve' or 'reject'."
        )
    
    try:
        await OrganizationService.process_organization_action(organization_id, action_request.action)
        return {"message": f"Organization {organization_id} {action_request.action}d successfully."}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/organizations/{organization_id}/users", status_code=status.HTTP_200_OK)
async def get_organization_users_endpoint(
    organization_id: str,
    superadmin_user: dict = Depends(get_superadmin_user)
):
    """
    Retrieves users for a specific organization.
    Accessible only by superadmins.
    """
    try:
        users = await OrganizationService.get_organization_users(organization_id)
        return users
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

from src.models.module import UpdateModuleStatusRequest

@router.post("/organizations/{organization_id}/modules", status_code=status.HTTP_200_OK)
async def update_module_status_endpoint(
    organization_id: str,
    request: UpdateModuleStatusRequest,
    superadmin_user: dict = Depends(get_superadmin_user)
):
    """
    Updates the status of a module for an organization.
    Accessible only by superadmins.
    """
    try:
        await OrganizationService.update_module_status(organization_id, request.module_slug, request.status)
        return {"message": f"Module {request.module_slug} updated to {request.status}."}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
