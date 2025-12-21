from src.core.database import supabase
from src.models.user import UserConfigResponse

class UserService:
    @staticmethod
    async def get_user_config(user_id: str) -> UserConfigResponse:
        if not supabase:
             raise ValueError("Database client not initialized")

        # Fetch profile from Supabase
        try:
            response = supabase.table("profiles").select("*").eq("id", user_id).single().execute()
            if not response.data:
                raise ValueError("User profile not found")
            
            data = response.data
            
            # Check if user is in superadmins table by email
            is_superadmin = False
            try:
                # Query superadmins table using the email from the profile
                print(f"DEBUG: Checking superadmin status for email: {data['email']}")
                sa_response = supabase.table("superadmins").select("id").eq("email", data['email']).maybe_single().execute()
                print(f"DEBUG: Superadmin table response: {sa_response.data}")
                if sa_response.data:
                    is_superadmin = True
            except Exception as e:
                print(f"Error checking superadmin status: {e}")
                # Default to False if check fails
                is_superadmin = False

            # Fetch Organization Context (First active membership)
            org_id = None
            org_name = None
            org_role = None
            try:
                org_response = supabase.table("organization_members")\
                    .select("role, organization_id, organizations(name)")\
                    .eq("user_id", user_id)\
                    .maybe_single()\
                    .execute()
                
                if org_response.data:
                    org_member = org_response.data
                    org_id = org_member.get('organization_id')
                    org_role = org_member.get('role')
                    # Access joined organization data
                    org_data = org_member.get('organizations')
                    if org_data:
                        org_name = org_data.get('name')
            except Exception as e:
                print(f"Error fetching organization context: {e}")

            return UserConfigResponse(
                user_id=data['id'],
                email=data['email'],
                is_superadmin=is_superadmin,
                full_name=data.get('full_name'),
                avatar_url=data.get('avatar_url'),
                organization_id=org_id,
                organization_name=org_name,
                role=org_role
            )
        except Exception as e:
            print(f"Error fetching user config: {e}")
            raise e
