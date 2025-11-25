from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from src.api.dependencies.auth import require_supabase_user, SupabaseUser
from src.core.config import get_settings

router = APIRouter(prefix="/team-profiles", tags=["team-profiles"])
logger = logging.getLogger(__name__)


# ==================== Pydantic Models ====================

class BillingInfo(BaseModel):
    ratePerHour: float
    currency: str = "AUD"


class DailyTimings(BaseModel):
    from_time: str = "09:00"
    to_time: str = "17:00"

    class Config:
        # Allow using 'from' as field name
        fields = {'from_time': 'from', 'to_time': 'to'}


class Experience(BaseModel):
    years: int
    role: str
    company: str
    description: str


class TeamProfileResponse(BaseModel):
    id: str
    name: str
    title: str
    email: str
    employeeCode: Optional[str] = None
    profilePhoto: Optional[str] = None
    department: str
    reportingManager: Optional[str] = None
    branch: Optional[str] = None
    location: Optional[str] = None
    timezone: Optional[str] = None
    billing: BillingInfo
    weeklyHours: int = 40
    dailyTimings: DailyTimings
    overtimeAllowed: bool = False
    employmentStartDate: Optional[str] = None
    employmentEndDate: Optional[str] = None
    profileSummary: Optional[str] = None
    qualifications: list[str] = []
    skillsAndExpertise: list[str] = []
    experience: list[Experience] = []
    isActive: bool = True
    status: str = "Active"


class TeamProfileCreate(BaseModel):
    name: str
    title: str
    email: EmailStr
    employeeCode: Optional[str] = None
    profilePhoto: Optional[str] = None
    department: str
    reportingManager: Optional[str] = None
    branch: Optional[str] = None
    location: Optional[str] = None
    timezone: Optional[str] = None
    overtimeAllowed: bool = False
    employmentStartDate: Optional[str] = None
    employmentEndDate: Optional[str] = None
    billingRatePerHour: float
    billingCurrency: str = "AUD"
    weeklyHours: int = 40
    dailyTimingsFrom: str = "09:00"
    dailyTimingsTo: str = "17:00"
    profileSummary: Optional[str] = None
    qualifications: list[str] = []
    skillsAndExpertise: list[str] = []
    experience: list[Experience] = []
    isActive: bool = True


class TeamProfileUpdate(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    email: Optional[EmailStr] = None
    employeeCode: Optional[str] = None
    profilePhoto: Optional[str] = None
    department: Optional[str] = None
    reportingManager: Optional[str] = None
    branch: Optional[str] = None
    location: Optional[str] = None
    timezone: Optional[str] = None
    overtimeAllowed: Optional[bool] = None
    employmentStartDate: Optional[str] = None
    employmentEndDate: Optional[str] = None
    billingRatePerHour: Optional[float] = None
    billingCurrency: Optional[str] = None
    weeklyHours: Optional[int] = None
    dailyTimingsFrom: Optional[str] = None
    dailyTimingsTo: Optional[str] = None
    profileSummary: Optional[str] = None
    qualifications: Optional[list[str]] = None
    skillsAndExpertise: Optional[list[str]] = None
    experience: Optional[list[Experience]] = None
    isActive: Optional[bool] = None


# ==================== Helper Functions ====================

def _build_supabase_headers() -> dict[str, str]:
    """Build headers for Supabase requests."""
    settings = get_settings()
    api_key = settings.supabase_anon_key or settings.supabase_key
    return {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


async def _supabase_request(
    method: str, url: str, headers: dict[str, str], json: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Make a request to Supabase."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        if method.upper() == "GET":
            response = await client.get(url, headers=headers)
        elif method.upper() == "POST":
            response = await client.post(url, headers=headers, json=json)
        elif method.upper() == "PATCH":
            response = await client.patch(url, headers=headers, json=json)
        elif method.upper() == "DELETE":
            response = await client.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

    if response.status_code >= 400:
        logger.error(f"Supabase request failed: {response.status_code} - {response.text}")
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Database operation failed: {response.text}",
        )

    try:
        return response.json()
    except ValueError:
        return {}


def _transform_db_to_response(db_profile: dict[str, Any]) -> dict[str, Any]:
    """Transform database profile to API response format."""
    return {
        "id": db_profile["id"],
        "name": db_profile["name"],
        "title": db_profile["title"],
        "email": db_profile["email"],
        "employeeCode": db_profile.get("employee_code"),
        "profilePhoto": db_profile.get("profile_photo"),
        "department": db_profile["department"],
        "reportingManager": db_profile.get("reporting_manager"),
        "branch": db_profile.get("branch"),
        "location": db_profile.get("location"),
        "timezone": db_profile.get("timezone"),
        "billing": {
            "ratePerHour": float(db_profile.get("billing_rate_per_hour", 0)),
            "currency": db_profile.get("billing_currency", "AUD"),
        },
        "weeklyHours": db_profile.get("weekly_hours", 40),
        "dailyTimings": {
            "from": db_profile.get("daily_timings_from", "09:00:00")[:5],
            "to": db_profile.get("daily_timings_to", "17:00:00")[:5],
        },
        "overtimeAllowed": db_profile.get("overtime_allowed", False),
        "employmentStartDate": db_profile.get("employment_start_date"),
        "employmentEndDate": db_profile.get("employment_end_date"),
        "profileSummary": db_profile.get("profile_summary"),
        "qualifications": db_profile.get("qualifications", []),
        "skillsAndExpertise": db_profile.get("skills", []),
        "experience": db_profile.get("experience", []),
        "isActive": db_profile.get("is_active", True),
        "status": "Active" if db_profile.get("is_active", True) else "Archived",
    }


# ==================== API Endpoints ====================

@router.get("/", response_model=list[TeamProfileResponse])
async def get_team_profiles(
    is_active: Optional[bool] = None,
    user: SupabaseUser = Depends(require_supabase_user),
) -> list[dict[str, Any]]:
    """Get all team profiles, optionally filtered by active status."""
    settings = get_settings()
    headers = _build_supabase_headers()

    # Build query URL
    url = f"{settings.supabase_url}/rest/v1/rpc/get_team_profiles_with_details"

    # Build request body for RPC function
    rpc_body = {}
    if is_active is not None:
        rpc_body["p_is_active"] = is_active

    profiles_data = await _supabase_request("POST", url, headers, json=rpc_body)

    # Transform the data
    transformed_profiles = [_transform_db_to_response(profile) for profile in profiles_data]

    return transformed_profiles


@router.get("/{profile_id}", response_model=TeamProfileResponse)
async def get_team_profile(
    profile_id: str,
    user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    """Get a single team profile by ID."""
    settings = get_settings()
    headers = _build_supabase_headers()

    url = f"{settings.supabase_url}/rest/v1/rpc/get_team_profile_by_id"
    profile_data = await _supabase_request("POST", url, headers, json={"p_profile_id": profile_id})

    if not profile_data or len(profile_data) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Team profile with ID {profile_id} not found",
        )

    return _transform_db_to_response(profile_data[0])


@router.post("/", response_model=TeamProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_team_profile(
    profile: TeamProfileCreate,
    user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    """Create a new team profile."""
    settings = get_settings()
    headers = _build_supabase_headers()

    # Insert main profile
    profile_data = {
        "name": profile.name,
        "title": profile.title,
        "email": profile.email,
        "employee_code": profile.employeeCode,
        "profile_photo": profile.profilePhoto,
        "department": profile.department,
        "reporting_manager": profile.reportingManager,
        "branch": profile.branch,
        "location": profile.location,
        "timezone": profile.timezone,
        "overtime_allowed": profile.overtimeAllowed,
        "employment_start_date": profile.employmentStartDate,
        "employment_end_date": profile.employmentEndDate,
        "billing_rate_per_hour": profile.billingRatePerHour,
        "billing_currency": profile.billingCurrency,
        "weekly_hours": profile.weeklyHours,
        "daily_timings_from": profile.dailyTimingsFrom,
        "daily_timings_to": profile.dailyTimingsTo,
        "profile_summary": profile.profileSummary,
        "is_active": profile.isActive,
        "status": "Active" if profile.isActive else "Archived",
        "created_by": user.id,
    }

    url = f"{settings.supabase_url}/rest/v1/team_profiles"
    created_profile = await _supabase_request("POST", url, headers, json=profile_data)

    if not created_profile or len(created_profile) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create team profile",
        )

    profile_id = created_profile[0]["id"]

    # Insert qualifications
    if profile.qualifications:
        qualifications_data = [
            {
                "team_profile_id": profile_id,
                "qualification": qual,
                "sort_order": idx,
            }
            for idx, qual in enumerate(profile.qualifications)
        ]
        qual_url = f"{settings.supabase_url}/rest/v1/team_profile_qualifications"
        await _supabase_request("POST", qual_url, headers, json=qualifications_data)

    # Insert skills
    if profile.skillsAndExpertise:
        skills_data = [
            {
                "team_profile_id": profile_id,
                "skill": skill,
                "sort_order": idx,
            }
            for idx, skill in enumerate(profile.skillsAndExpertise)
        ]
        skills_url = f"{settings.supabase_url}/rest/v1/team_profile_skills"
        await _supabase_request("POST", skills_url, headers, json=skills_data)

    # Insert experience
    if profile.experience:
        experience_data = [
            {
                "team_profile_id": profile_id,
                "years": exp.years,
                "role": exp.role,
                "company": exp.company,
                "description": exp.description,
                "sort_order": idx,
            }
            for idx, exp in enumerate(profile.experience)
        ]
        exp_url = f"{settings.supabase_url}/rest/v1/team_profile_experience"
        await _supabase_request("POST", exp_url, headers, json=experience_data)

    # Fetch the complete profile
    return await get_team_profile(profile_id, user)


@router.patch("/{profile_id}", response_model=TeamProfileResponse)
async def update_team_profile(
    profile_id: str,
    profile_update: TeamProfileUpdate,
    user: SupabaseUser = Depends(require_supabase_user),
) -> dict[str, Any]:
    """Update a team profile."""
    settings = get_settings()
    headers = _build_supabase_headers()

    # Build update data for main profile
    update_data = {}
    if profile_update.name is not None:
        update_data["name"] = profile_update.name
    if profile_update.title is not None:
        update_data["title"] = profile_update.title
    if profile_update.email is not None:
        update_data["email"] = profile_update.email
    if profile_update.employeeCode is not None:
        update_data["employee_code"] = profile_update.employeeCode
    if profile_update.profilePhoto is not None:
        update_data["profile_photo"] = profile_update.profilePhoto
    if profile_update.department is not None:
        update_data["department"] = profile_update.department
    if profile_update.reportingManager is not None:
        update_data["reporting_manager"] = profile_update.reportingManager
    if profile_update.branch is not None:
        update_data["branch"] = profile_update.branch
    if profile_update.location is not None:
        update_data["location"] = profile_update.location
    if profile_update.timezone is not None:
        update_data["timezone"] = profile_update.timezone
    if profile_update.overtimeAllowed is not None:
        update_data["overtime_allowed"] = profile_update.overtimeAllowed
    if profile_update.employmentStartDate is not None:
        update_data["employment_start_date"] = profile_update.employmentStartDate
    if profile_update.employmentEndDate is not None:
        update_data["employment_end_date"] = profile_update.employmentEndDate
    if profile_update.billingRatePerHour is not None:
        update_data["billing_rate_per_hour"] = profile_update.billingRatePerHour
    if profile_update.billingCurrency is not None:
        update_data["billing_currency"] = profile_update.billingCurrency
    if profile_update.weeklyHours is not None:
        update_data["weekly_hours"] = profile_update.weeklyHours
    if profile_update.dailyTimingsFrom is not None:
        update_data["daily_timings_from"] = profile_update.dailyTimingsFrom
    if profile_update.dailyTimingsTo is not None:
        update_data["daily_timings_to"] = profile_update.dailyTimingsTo
    if profile_update.profileSummary is not None:
        update_data["profile_summary"] = profile_update.profileSummary
    if profile_update.isActive is not None:
        update_data["is_active"] = profile_update.isActive
        update_data["status"] = "Active" if profile_update.isActive else "Archived"

    update_data["updated_by"] = user.id

    # Update main profile
    if update_data:
        url = f"{settings.supabase_url}/rest/v1/team_profiles?id=eq.{profile_id}"
        await _supabase_request("PATCH", url, headers, json=update_data)

    # Update qualifications if provided
    if profile_update.qualifications is not None:
        # Delete existing qualifications
        del_url = f"{settings.supabase_url}/rest/v1/team_profile_qualifications?team_profile_id=eq.{profile_id}"
        await _supabase_request("DELETE", del_url, headers)

        # Insert new qualifications
        if profile_update.qualifications:
            qualifications_data = [
                {
                    "team_profile_id": profile_id,
                    "qualification": qual,
                    "sort_order": idx,
                }
                for idx, qual in enumerate(profile_update.qualifications)
            ]
            qual_url = f"{settings.supabase_url}/rest/v1/team_profile_qualifications"
            await _supabase_request("POST", qual_url, headers, json=qualifications_data)

    # Update skills if provided
    if profile_update.skillsAndExpertise is not None:
        # Delete existing skills
        del_url = f"{settings.supabase_url}/rest/v1/team_profile_skills?team_profile_id=eq.{profile_id}"
        await _supabase_request("DELETE", del_url, headers)

        # Insert new skills
        if profile_update.skillsAndExpertise:
            skills_data = [
                {
                    "team_profile_id": profile_id,
                    "skill": skill,
                    "sort_order": idx,
                }
                for idx, skill in enumerate(profile_update.skillsAndExpertise)
            ]
            skills_url = f"{settings.supabase_url}/rest/v1/team_profile_skills"
            await _supabase_request("POST", skills_url, headers, json=skills_data)

    # Update experience if provided
    if profile_update.experience is not None:
        # Delete existing experience
        del_url = f"{settings.supabase_url}/rest/v1/team_profile_experience?team_profile_id=eq.{profile_id}"
        await _supabase_request("DELETE", del_url, headers)

        # Insert new experience
        if profile_update.experience:
            experience_data = [
                {
                    "team_profile_id": profile_id,
                    "years": exp.years,
                    "role": exp.role,
                    "company": exp.company,
                    "description": exp.description,
                    "sort_order": idx,
                }
                for idx, exp in enumerate(profile_update.experience)
            ]
            exp_url = f"{settings.supabase_url}/rest/v1/team_profile_experience"
            await _supabase_request("POST", exp_url, headers, json=experience_data)

    # Fetch the updated profile
    return await get_team_profile(profile_id, user)


@router.delete("/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_team_profile(
    profile_id: str,
    user: SupabaseUser = Depends(require_supabase_user),
) -> None:
    """Delete a team profile (soft delete by setting is_active to False)."""
    settings = get_settings()
    headers = _build_supabase_headers()

    # Soft delete by setting is_active to False
    update_data = {
        "is_active": False,
        "status": "Archived",
        "updated_by": user.id,
    }

    url = f"{settings.supabase_url}/rest/v1/team_profiles?id=eq.{profile_id}"
    await _supabase_request("PATCH", url, headers, json=update_data)
