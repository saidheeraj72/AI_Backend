from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from src.core.dependencies import get_current_org_admin
from src.services.org_admin_service import OrgAdminService
from src.models.org_admin import BranchDTO, CreateBranchRequest, OrgUserDTO, InviteUserRequest

router = APIRouter()

@router.get("/branches", response_model=List[BranchDTO])
async def get_branches(org_id: str = Depends(get_current_org_admin)):
    """
    List all branches for the current organization.
    """
    try:
        return await OrgAdminService.get_branches(org_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/branches", response_model=BranchDTO)
async def create_branch(
    branch_data: CreateBranchRequest,
    org_id: str = Depends(get_current_org_admin)
):
    """
    Create a new branch for the current organization.
    """
    try:
        return await OrgAdminService.create_branch(org_id, branch_data)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/users", response_model=List[OrgUserDTO])
async def get_users(org_id: str = Depends(get_current_org_admin)):
    """
    List all users in the current organization.
    """
    try:
        return await OrgAdminService.get_users(org_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/users", status_code=status.HTTP_201_CREATED)
async def invite_user(
    invite_data: InviteUserRequest,
    org_id: str = Depends(get_current_org_admin)
):
    """
    Add an existing user to the organization by email.
    """
    try:
        await OrgAdminService.invite_user(org_id, invite_data)
        return {"message": f"User {invite_data.email} added successfully."}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
