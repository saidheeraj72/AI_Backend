from __future__ import annotations

import logging
from typing import Any, Optional, List
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from src.api.dependencies.auth import require_supabase_user, SupabaseUser
from src.core.config import get_settings
from src.services.organization_service import OrganizationService
from src.models.schemas import (
    TeamProfile as DbTeamProfile,
    TeamSkill,
    TeamQualification,
    TeamExperience
)

router = APIRouter(prefix="/team-profiles", tags=["team-profiles"])
logger = logging.getLogger(__name__)

# --- Frontend-Compatible Models ---

class BillingInfo(BaseModel):
    ratePerHour: float
    currency: str = "AUD"

class DailyTimings(BaseModel):
    from_time: str = Field(..., alias="from")
    to_time: str = Field(..., alias="to")

    class Config:
        populate_by_name = True

class ExperienceItem(BaseModel):
    years: float
    role: str
    company: str
    description: Optional[str] = None

class TeamProfileResponse(BaseModel):
    id: str
    name: Optional[str] = None # In Profile, but we might need to fetch it
    title: Optional[str] = Field(None, alias="job_title") # Map job_title to title
    email: Optional[str] = None
    employeeCode: Optional[str] = None
    profilePhoto: Optional[str] = None
    department: Optional[str] = None
    reportingManager: Optional[str] = None
    branch: Optional[str] = None # We have branch_id, need name?
    location: Optional[str] = None # Derived from branch?
    timezone: Optional[str] = None
    billing: BillingInfo
    weeklyHours: float = 40
    dailyTimings: DailyTimings
    overtimeAllowed: bool = False
    employmentStartDate: Optional[str] = None
    employmentEndDate: Optional[str] = None
    profileSummary: Optional[str] = None
    qualifications: List[str] = []
    skillsAndExpertise: List[str] = []
    experience: List[ExperienceItem] = []
    isActive: bool = True
    status: str = "Active"

    class Config:
        populate_by_name = True

class TeamProfileCreateRequest(BaseModel):
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
    weeklyHours: float = 40
    dailyTimingsFrom: str = "09:00"
    dailyTimingsTo: str = "17:00"
    profileSummary: Optional[str] = None
    qualifications: list[str] = []
    skillsAndExpertise: list[str] = []
    experience: list[dict] = []
    isActive: bool = True


# ==================== Helper Functions ====================

def _build_supabase_headers() -> dict[str, str]:
    settings = get_settings()
    api_key = settings.supabase_anon_key or settings.supabase_key
    return {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation" 
    }

async def _supabase_request(
    method: str, url: str, headers: dict[str, str], json: Optional[dict[str, Any]] = None
) -> Any:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.request(method, url, headers=headers, json=json)

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

def _map_db_to_response(db_profile: dict) -> TeamProfileResponse:
    # Map snake_case DB fields to camelCase frontend model
    
    skills = [s["skill_name"] for s in db_profile.get("skills", [])]
    quals = [q["degree"] for q in db_profile.get("qualifications", [])]
    exps = [
        ExperienceItem(
            years=e.get("years_duration", 0),
            role=e.get("role", ""),
            company=e.get("company", ""),
            description=e.get("description", "")
        ) 
        for e in db_profile.get("experience", [])
    ]

    return TeamProfileResponse(
        id=db_profile["id"],
        name=db_profile.get("full_name") or "Unknown", # Should join with profiles
        title=db_profile.get("job_title"),
        email=db_profile.get("email"), # Should join with profiles
        employeeCode=db_profile.get("employee_code"),
        # profilePhoto=...
        department=db_profile.get("department"),
        # reportingManager=...
        # branch=...
        timezone=db_profile.get("timezone"),
        billing=BillingInfo(
            ratePerHour=db_profile.get("billing_rate", 0) or 0,
            currency=db_profile.get("billing_currency", "USD")
        ),
        weeklyHours=db_profile.get("weekly_hours", 40),
        dailyTimings=DailyTimings(
            from_time=str(db_profile.get("daily_timings_from", "09:00")),
            to_time=str(db_profile.get("daily_timings_to", "17:00"))
        ),
        overtimeAllowed=db_profile.get("overtime_allowed", False),
        employmentStartDate=str(db_profile.get("employment_start_date")) if db_profile.get("employment_start_date") else None,
        employmentEndDate=str(db_profile.get("employment_end_date")) if db_profile.get("employment_end_date") else None,
        profileSummary=db_profile.get("profile_summary"),
        qualifications=quals,
        skillsAndExpertise=skills,
        experience=exps,
        isActive=db_profile.get("is_active", True),
        status=db_profile.get("status", "Active")
    )

# ==================== API Endpoints ====================

@router.get("/", response_model=list[TeamProfileResponse])
async def get_team_profiles(
    is_active: Optional[bool] = None,
    user: SupabaseUser = Depends(require_supabase_user),
) -> list[TeamProfileResponse]:
    settings = get_settings()
    headers = _build_supabase_headers()

    # We need to join with profiles table to get name/email
    # select=*,skills:team_skills(*),qualifications:team_qualifications(*),experience:team_experience(*),profiles(full_name,email,avatar_url)
    query = "*,skills:team_skills(*),qualifications:team_qualifications(*),experience:team_experience(*),profiles(full_name,email,avatar_url)"
    
    url = f"{settings.supabase_url}/rest/v1/team_profiles?select={query}"
    if is_active is not None:
        url += f"&is_active=eq.{str(is_active).lower()}"

    data = await _supabase_request("GET", url, headers)
    
    # Flatten the 'profiles' join
    for item in data:
        if item.get("profiles"):
            item["full_name"] = item["profiles"].get("full_name")
            item["email"] = item["profiles"].get("email")
            item["avatar_url"] = item["profiles"].get("avatar_url")
    
    return [_map_db_to_response(item) for item in data]

@router.get("/{profile_id}", response_model=TeamProfileResponse)
async def get_team_profile(
    profile_id: str,
    user: SupabaseUser = Depends(require_supabase_user),
) -> TeamProfileResponse:
    settings = get_settings()
    headers = _build_supabase_headers()

    query = "*,skills:team_skills(*),qualifications:team_qualifications(*),experience:team_experience(*),profiles(full_name,email,avatar_url)"
    url = f"{settings.supabase_url}/rest/v1/team_profiles?id=eq.{profile_id}&select={query}"

    data = await _supabase_request("GET", url, headers)
    if not data:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    item = data[0]
    if item.get("profiles"):
        item["full_name"] = item["profiles"].get("full_name")
        item["email"] = item["profiles"].get("email")
    
    return _map_db_to_response(item)

@router.post("/", response_model=TeamProfileResponse)
async def create_team_profile(
    request: TeamProfileCreateRequest,
    user: SupabaseUser = Depends(require_supabase_user),
) -> TeamProfileResponse:
    settings = get_settings()
    headers = _build_supabase_headers()
    
    # Resolve Organization ID
    org_id = settings.default_org_id
    # Validate if org_id is a valid UUID string if present
    if org_id:
        try:
            UUID(org_id)
        except ValueError:
            logger.warning(f"Invalid default_org_id in settings: {org_id}")
            org_id = None
            
    if not org_id:
        # Fallback: Fetch from user's organizations
        # Note: mixing sync service call in async route (following project pattern)
        org_service = OrganizationService(settings)
        user_orgs = org_service.list_user_organizations(UUID(user.id))
        if user_orgs:
            org_id = str(user_orgs[0].id)
            
    if not org_id:
        raise HTTPException(status_code=400, detail="Organization ID is required to create a profile")
    
    # Map request to DB fields
    payload = {
        "id": user.id, # Enforce own profile for now, or we need ID in request
        "organization_id": org_id, 
        "job_title": request.title,
        "department": request.department,
        "employee_code": request.employeeCode,
        "weekly_hours": request.weeklyHours,
        "overtime_allowed": request.overtimeAllowed,
        "daily_timings_from": request.dailyTimingsFrom,
        "daily_timings_to": request.dailyTimingsTo,
        "timezone": request.timezone,
        "billing_rate": request.billingRatePerHour,
        "billing_currency": request.billingCurrency,
        "profile_summary": request.profileSummary,
        "employment_start_date": request.employmentStartDate,
        "employment_end_date": request.employmentEndDate,
        "is_active": request.isActive,
        "status": "Active" if request.isActive else "Archived"
    }
    
    url = f"{settings.supabase_url}/rest/v1/team_profiles"
    created = await _supabase_request("POST", url, headers, json=payload)
    if not created:
         raise HTTPException(status_code=400, detail="Failed to create profile")
    
    pid = created[0]["id"]
    
    # Skills
    if request.skillsAndExpertise:
        s_data = [{"profile_id": pid, "skill_name": s} for s in request.skillsAndExpertise]
        await _supabase_request("POST", f"{settings.supabase_url}/rest/v1/team_skills", headers, json=s_data)

    # Qualifications
    if request.qualifications:
        q_data = [{"profile_id": pid, "degree": q} for q in request.qualifications]
        await _supabase_request("POST", f"{settings.supabase_url}/rest/v1/team_qualifications", headers, json=q_data)
        
    # Experience
    if request.experience:
        e_data = []
        for e in request.experience:
             e_data.append({
                 "profile_id": pid,
                 "company": e.get("company"),
                 "role": e.get("role"),
                 "years_duration": e.get("years"),
                 "description": e.get("description")
             })
        await _supabase_request("POST", f"{settings.supabase_url}/rest/v1/team_experience", headers, json=e_data)

    return await get_team_profile(pid, user)
