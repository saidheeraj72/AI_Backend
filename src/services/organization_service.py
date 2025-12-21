from src.core.database import supabase
from src.models.organization import OrganizationDTO
from typing import List

class OrganizationService:
    @staticmethod
    async def get_pending_organizations() -> List[OrganizationDTO]:
        try:
            # Query for pending organizations
            response = supabase.table("organizations").select("*").eq("status", "pending_approval").execute()
            return [OrganizationDTO(**item) for item in response.data]
        except Exception as e:
            print(f"Error fetching pending organizations: {e}")
            raise e

    @staticmethod
    async def get_all_organizations() -> List[OrganizationDTO]:
        try:
            # Query for all organizations, excluding pending ones, AND filtering for is_organization=True
            # as requested: "in tenants it shows only true"
            response = supabase.table("organizations").select("*").neq("status", "pending_approval").eq("is_organization", True).execute()
            return [OrganizationDTO(**item) for item in response.data]
        except Exception as e:
            print(f"Error fetching organizations: {e}")
            raise e

    @staticmethod
    async def get_organization_users(org_id: str):
        try:
            # Join organization_members with profiles
            response = supabase.table("organization_members").select("user_id, role, status, profiles(email, full_name, avatar_url)").eq("organization_id", org_id).execute()
            
            users = []
            for item in response.data:
                profile = item.get('profiles') or {}
                users.append({
                    "id": item['user_id'],
                    "email": profile.get('email'),
                    "full_name": profile.get('full_name'),
                    "avatar_url": profile.get('avatar_url'),
                    "role": item['role'],
                    "status": item['status']
                })
            return users
        except Exception as e:
            print(f"Error fetching organization users: {e}")
            raise e

    @staticmethod
    async def update_module_status(org_id: str, module_slug: str, status: str):
        try:
            # We need to update the specific key in the jsonb column.
            # Postgres jsonb_set is useful here, or we can fetch, modify, and update.
            # Fetch current modules
            response = supabase.table("organizations").select("modules").eq("id", org_id).single().execute()
            if not response.data:
                raise ValueError("Organization not found")
            
            current_modules = response.data.get("modules") or {}
            current_modules[module_slug] = status
            
            # Update back
            update_response = supabase.table("organizations").update({"modules": current_modules}).eq("id", org_id).execute()
            return update_response.data
        except Exception as e:
            print(f"Error updating module status: {e}")
            raise e

    @staticmethod
    async def process_organization_action(org_id: str, action: str):
        try:
            # Map action to status
            # 'approve' -> 'active'
            # 'reject' -> 'suspended' (or 'archived', using suspended for now)
            new_status = 'active' if action == 'approve' else 'suspended'
            
            # If action is invalid, we could raise error, but Pydantic model constrains it if we used Enum.
            # Here we assume validation happened at controller/route level or basic logic.
            if action not in ['approve', 'reject']:
                 raise ValueError("Invalid action. Must be 'approve' or 'reject'.")

            response = supabase.table("organizations").update({"status": new_status}).eq("id", org_id).execute()
            return response.data
        except Exception as e:
            print(f"Error updating organization status: {e}")
            raise e
