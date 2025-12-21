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

            return UserConfigResponse(
                user_id=data['id'],
                email=data['email'],
                is_superadmin=is_superadmin,
                full_name=data.get('full_name'),
                avatar_url=data.get('avatar_url')
            )
        except Exception as e:
            print(f"Error fetching user config: {e}")
            raise e
