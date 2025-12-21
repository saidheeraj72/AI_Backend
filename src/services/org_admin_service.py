from src.core.database import supabase
from src.models.org_admin import BranchDTO, CreateBranchRequest, OrgUserDTO, InviteUserRequest
from typing import List, Optional
from datetime import datetime

class OrgAdminService:
    @staticmethod
    async def get_branches(org_id: str) -> List[BranchDTO]:
        try:
            response = supabase.table("branches").select("*").eq("organization_id", org_id).execute()
            # For now, stats are 0
            return [BranchDTO(**item, user_count=0, storage_usage=0) for item in response.data]
        except Exception as e:
            print(f"Error fetching branches: {e}")
            raise e

    @staticmethod
    async def create_branch(org_id: str, data: CreateBranchRequest) -> BranchDTO:
        try:
            new_branch = {
                "organization_id": org_id,
                "name": data.name,
                "code": data.code,
                "location": data.location,
                "is_active": True
            }
            response = supabase.table("branches").insert(new_branch).execute()
            if not response.data:
                raise ValueError("Failed to create branch")
            
            return BranchDTO(**response.data[0], user_count=0, storage_usage=0)
        except Exception as e:
            print(f"Error creating branch: {e}")
            raise e

    @staticmethod
    async def get_users(org_id: str) -> List[OrgUserDTO]:
        try:
            # Join with profiles to get details
            response = supabase.table("organization_members")\
                .select("user_id, role, status, joined_at, profiles(email, full_name, avatar_url)")\
                .eq("organization_id", org_id)\
                .execute()
            
            users = []
            for item in response.data:
                profile = item.get('profiles') or {}
                users.append(OrgUserDTO(
                    id=item['user_id'],
                    email=profile.get('email'),
                    full_name=profile.get('full_name'),
                    avatar_url=profile.get('avatar_url'),
                    role=item['role'],
                    status=item['status'],
                    joined_at=item['joined_at']
                ))
            return users
        except Exception as e:
            print(f"Error fetching users: {e}")
            raise e

    @staticmethod
    async def search_resources(org_id: str, query: str):
        try:
            resources = []
            # Search folders
            folders = supabase.table("folders")\
                .select("id, name")\
                .eq("organization_id", org_id)\
                .ilike("name", f"%{query}%")\
                .limit(10)\
                .execute()
            
            for f in folders.data:
                resources.append({"id": f['id'], "name": f['name'], "type": "folder"})

            # Search documents
            docs = supabase.table("documents")\
                .select("id, title")\
                .eq("organization_id", org_id)\
                .ilike("title", f"%{query}%")\
                .limit(10)\
                .execute()
                
            for d in docs.data:
                resources.append({"id": d['id'], "name": d['title'], "type": "document"})
                
            return resources
        except Exception as e:
            print(f"Error searching resources: {e}")
            raise e

    @staticmethod
    async def get_folder_contents(org_id: str, folder_id: Optional[str] = None):
        try:
            items = []
            
            # Fetch Subfolders
            query = supabase.table("folders").select("id, name").eq("organization_id", org_id)
            if folder_id:
                query = query.eq("parent_id", folder_id)
            else:
                query = query.is_("parent_id", "null")
            
            folders = query.execute()
            for f in folders.data:
                items.append({"id": f['id'], "name": f['name'], "type": "folder"})

            # Fetch Documents (only if inside a folder, or root docs if applicable)
            doc_query = supabase.table("documents").select("id, title").eq("organization_id", org_id)
            if folder_id:
                doc_query = doc_query.eq("folder_id", folder_id)
            else:
                doc_query = doc_query.is_("folder_id", "null")
                
            docs = doc_query.execute()
            for d in docs.data:
                items.append({"id": d['id'], "name": d['title'], "type": "document"})
                
            return items
        except Exception as e:
            print(f"Error fetching folder contents: {e}")
            raise e

    @staticmethod
    async def invite_user(org_id: str, data: InviteUserRequest):
        # This is a complex flow involving Auth. 
        # For now, we'll try to find an existing profile by email and link it.
        try:
            # 1. Check if user exists in profiles
            profile_response = supabase.table("profiles").select("id").eq("email", data.email).maybe_single().execute()
            
            if not profile_response.data:
                raise ValueError(f"User with email {data.email} not found. Please ask them to sign up first.")
            
            user_id = profile_response.data['id']
            
            # 2. Add to organization_members
            member_data = {
                "organization_id": org_id,
                "user_id": user_id,
                "role": data.role,
                "status": "active" # Auto-activate for now
            }
            
            response = supabase.table("organization_members").insert(member_data).execute()
            return response.data
        except Exception as e:
            print(f"Error inviting user: {e}")
            raise e
