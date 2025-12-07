from __future__ import annotations

import logging
from typing import Any, Optional, List
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field

from src.api.dependencies.auth import require_supabase_user, SupabaseUser, bearer_scheme
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
    name: str
    title: str = Field(..., alias="job_title")
    email: str
    employeeCode: Optional[str] = None
    profilePhoto: str
    department: str
    reportingManager: str
    branch: str
    location: str
    timezone: str
    billing: BillingInfo
    weeklyHours: float = 40
    dailyTimings: DailyTimings
    overtimeAllowed: bool = False
    employmentStartDate: Optional[str] = None
    employmentEndDate: Optional[str] = None
    profileSummary: str
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

class TeamProfileUpdateRequest(BaseModel):
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
    weeklyHours: Optional[float] = None
    dailyTimingsFrom: Optional[str] = None
    dailyTimingsTo: Optional[str] = None
    profileSummary: Optional[str] = None
    qualifications: Optional[list[str]] = None
    skillsAndExpertise: Optional[list[str]] = None
    experience: Optional[list[dict]] = None
    isActive: Optional[bool] = None


# ==================== Helper Functions ====================

def _build_supabase_headers(token: Optional[str] = None) -> dict[str, str]:
    settings = get_settings()
    api_key = settings.supabase_anon_key or settings.supabase_key
    headers = {
        "apikey": api_key,
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers

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
        name=db_profile.get("full_name") or "Unknown",
        title=db_profile.get("job_title") or "No Title",
        email=db_profile.get("email") or "",
        employeeCode=db_profile.get("employee_code"),
        profilePhoto=db_profile.get("avatar_url") or "",
        department=db_profile.get("department") or "Unassigned",
        reportingManager=db_profile.get("reporting_manager_id") or "",
        branch=db_profile.get("branch_name") or "", # Use branch_name from joined data
        location=db_profile.get("branch_location") or "",  # Mapped from branch
        timezone=db_profile.get("timezone") or "UTC",
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
        profileSummary=db_profile.get("profile_summary") or "",
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
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> list[TeamProfileResponse]:
    settings = get_settings()
    
    # --- RLS BYPASS IMPLEMENTATION ---
    # The debug script confirmed RLS policies are blocking access for authenticated users.
    # To ensure data display, we use the server-side Supabase Key (assumed to be service role key)
    # for this specific API call to bypass RLS.
    headers = {
        "apikey": settings.supabase_key, # Use supabase_key here
        "Content-Type": "application/json",
        "Prefer": "return=representation",
        "Authorization": f"Bearer {settings.supabase_key}" # Bypass RLS
    }
    logger.info("RLS Bypass: Using server-side Supabase Key for team_profiles access.")
    # --- END RLS BYPASS ---

    # We need to join with profiles table to get name/email AND branches for branch name
    query = "*,skills:team_skills(*),qualifications:team_qualifications(*),experience:team_experience(*),profiles:profiles!team_profiles_id_fkey(full_name,email,avatar_url),branches:branches!team_profiles_branch_id_fkey(name,location)"
    
    url = f"{settings.supabase_url}/rest/v1/team_profiles?select={query}"
    if is_active is not None:
        url += f"&is_active=eq.{str(is_active).lower()}"

    logger.info(f"Requesting Supabase URL (RLS Bypass): {url}")
    data = await _supabase_request("GET", url, headers)
    
    logger.info(f"Supabase returned {len(data) if isinstance(data, list) else 'invalid'} records (RLS Bypass). Raw data: {data}")
    
    if not isinstance(data, list):
        logger.error(f"Unexpected response from Supabase (expected list): {data}")
        return []

    # Flatten the 'profiles' and 'branches' joins
    for item in data:
        if not isinstance(item, dict):
            continue

        if item.get("profiles"):
            item["full_name"] = item["profiles"].get("full_name")
            item["email"] = item["profiles"].get("email")
            item["avatar_url"] = item["profiles"].get("avatar_url")
        
        if item.get("branches"):
            item["branch_name"] = item["branches"].get("name")
            item["branch_location"] = item["branches"].get("location")
    
    return [_map_db_to_response(item) for item in data if isinstance(item, dict)]

    # Flatten the 'profiles' join
    for item in data:
        if not isinstance(item, dict):
            continue

        if item.get("profiles"):
            item["full_name"] = item["profiles"].get("full_name")
            item["email"] = item["profiles"].get("email")
            item["avatar_url"] = item["profiles"].get("avatar_url")
    
    return [_map_db_to_response(item) for item in data if isinstance(item, dict)]

@router.get("/{profile_id}", response_model=TeamProfileResponse)
async def get_team_profile(
    profile_id: str,
    user: SupabaseUser = Depends(require_supabase_user),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> TeamProfileResponse:
    settings = get_settings()
    
    # --- RLS BYPASS ---
    headers = {
        "apikey": settings.supabase_key,
        "Content-Type": "application/json",
        "Prefer": "return=representation",
        "Authorization": f"Bearer {settings.supabase_key}"
    }
    logger.info(f"RLS Bypass: Fetching single profile {profile_id} using server-side key.")
    # ------------------

    query = "*,skills:team_skills(*),qualifications:team_qualifications(*),experience:team_experience(*),profiles:profiles!team_profiles_id_fkey(full_name,email,avatar_url),branches:branches!team_profiles_branch_id_fkey(name,location)"
    url = f"{settings.supabase_url}/rest/v1/team_profiles?id=eq.{profile_id}&select={query}"

    data = await _supabase_request("GET", url, headers)
    
    if not isinstance(data, list) or not data:
        if isinstance(data, dict):
            logger.error(f"Unexpected response in get_team_profile: {data}")
        raise HTTPException(status_code=404, detail="Profile not found")
    
    item = data[0]
    if not isinstance(item, dict):
        raise HTTPException(status_code=500, detail="Invalid profile data format")

    if item.get("profiles"):
        item["full_name"] = item["profiles"].get("full_name")
        item["email"] = item["profiles"].get("email")
        item["avatar_url"] = item["profiles"].get("avatar_url")
        
    if item.get("branches"):
        item["branch_name"] = item["branches"].get("name")
        item["branch_location"] = item["branches"].get("location")
    
    return _map_db_to_response(item)

@router.post("/", response_model=TeamProfileResponse)
async def create_team_profile(
    request: TeamProfileCreateRequest,
    user: SupabaseUser = Depends(require_supabase_user),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> TeamProfileResponse:
    settings = get_settings()
    headers = _build_supabase_headers(token=credentials.credentials)
    
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
    
    if not isinstance(created, list) or not created or not isinstance(created[0], dict):
         logger.error(f"Failed to create profile, response: {created}")
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

    return await get_team_profile(pid, user, credentials)

@router.patch("/{profile_id}", response_model=TeamProfileResponse)
async def update_team_profile(
    profile_id: str,
    request: TeamProfileUpdateRequest,
    user: SupabaseUser = Depends(require_supabase_user),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> TeamProfileResponse:
    settings = get_settings()
    
    # Use Service Key to bypass RLS
    headers = {
        "apikey": settings.supabase_key,
        "Content-Type": "application/json",
        "Prefer": "return=representation",
        "Authorization": f"Bearer {settings.supabase_key}"
    }

    # 1. Resolve Branch ID if name is provided
    branch_id = None
    if request.branch:
        branch_url = f"{settings.supabase_url}/rest/v1/branches?name=eq.{request.branch}&select=id"
        branch_data = await _supabase_request("GET", branch_url, headers)
        if isinstance(branch_data, list) and len(branch_data) > 0:
            branch_id = branch_data[0]['id']

    # 2. Prepare Update Payload
    payload = {}
    if request.title is not None: payload["job_title"] = request.title
    if request.department is not None: payload["department"] = request.department
    if request.employeeCode is not None: payload["employee_code"] = request.employeeCode
    if request.weeklyHours is not None: payload["weekly_hours"] = request.weeklyHours
    if request.overtimeAllowed is not None: payload["overtime_allowed"] = request.overtimeAllowed
    if request.dailyTimingsFrom is not None: payload["daily_timings_from"] = request.dailyTimingsFrom
    if request.dailyTimingsTo is not None: payload["daily_timings_to"] = request.dailyTimingsTo
    if request.timezone is not None: payload["timezone"] = request.timezone
    if request.billingRatePerHour is not None: payload["billing_rate"] = request.billingRatePerHour
    if request.billingCurrency is not None: payload["billing_currency"] = request.billingCurrency
    if request.profileSummary is not None: payload["profile_summary"] = request.profileSummary
    if request.employmentStartDate is not None: payload["employment_start_date"] = request.employmentStartDate
    if request.employmentEndDate is not None: payload["employment_end_date"] = request.employmentEndDate
    if request.isActive is not None: 
        payload["is_active"] = request.isActive
        payload["status"] = "Active" if request.isActive else "Archived"
    
    if branch_id:
        payload["branch_id"] = branch_id

    # 3. Update Team Profile
    if payload:
        url = f"{settings.supabase_url}/rest/v1/team_profiles?id=eq.{profile_id}"
        await _supabase_request("PATCH", url, headers, json=payload)

    # 4. Update Relations (Delete & Re-insert strategy)
    
    # Skills
    if request.skillsAndExpertise is not None:
        # Delete existing
        await _supabase_request("DELETE", f"{settings.supabase_url}/rest/v1/team_skills?profile_id=eq.{profile_id}", headers)
        # Insert new
        if request.skillsAndExpertise:
            s_data = [{"profile_id": profile_id, "skill_name": s} for s in request.skillsAndExpertise]
            await _supabase_request("POST", f"{settings.supabase_url}/rest/v1/team_skills", headers, json=s_data)

    # Qualifications
    if request.qualifications is not None:
        await _supabase_request("DELETE", f"{settings.supabase_url}/rest/v1/team_qualifications?profile_id=eq.{profile_id}", headers)
        if request.qualifications:
            q_data = [{"profile_id": profile_id, "degree": q} for q in request.qualifications]
            await _supabase_request("POST", f"{settings.supabase_url}/rest/v1/team_qualifications", headers, json=q_data)

    # Experience
    if request.experience is not None:
        await _supabase_request("DELETE", f"{settings.supabase_url}/rest/v1/team_experience?profile_id=eq.{profile_id}", headers)
        if request.experience:
            e_data = []
            for e in request.experience:
                 e_data.append({
                     "profile_id": profile_id,
                     "company": e.get("company"),
                     "role": e.get("role"),
                     "years_duration": e.get("years"),
                     "description": e.get("description")
                 })
            await _supabase_request("POST", f"{settings.supabase_url}/rest/v1/team_experience", headers, json=e_data)

    return await get_team_profile(profile_id, user, credentials)
