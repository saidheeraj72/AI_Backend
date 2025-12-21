from src.core.database import supabase
from src.models.permission import PermissionDTO, GrantPermissionRequest
from typing import List

class PermissionService:
    @staticmethod
    async def get_user_permissions(user_id: str) -> List[PermissionDTO]:
        try:
            # Fetch ACL entries for the user
            # We might want to join with folders/documents to get names, but Supabase join syntax is specific.
            # We'll fetch ACL first.
            response = supabase.table("access_control_list").select("*").eq("grantee_id", user_id).execute()
            
            permissions = []
            for item in response.data:
                # Fetch resource name manually for now (optimization: can utilize join if resource_type is consistent or separate queries)
                resource_name = "Unknown Resource"
                if item['resource_type'] == 'folder':
                    res = supabase.table("folders").select("name").eq("id", item['resource_id']).maybe_single().execute()
                    if res.data: resource_name = res.data['name']
                elif item['resource_type'] == 'document':
                    res = supabase.table("documents").select("title").eq("id", item['resource_id']).maybe_single().execute()
                    if res.data: resource_name = res.data['title']

                permissions.append(PermissionDTO(
                    id=item['id'],
                    resource_type=item['resource_type'],
                    resource_id=item['resource_id'],
                    resource_name=resource_name,
                    grantee_id=item['grantee_id'],
                    permission=item['permission'],
                    granted_by=item['granted_by'],
                    created_at=item['created_at']
                ))
            return permissions
        except Exception as e:
            print(f"Error fetching permissions: {e}")
            raise e

    @staticmethod
    async def grant_permission(data: GrantPermissionRequest, granted_by: str):
        try:
            # Check if entry exists to update or insert
            existing = supabase.table("access_control_list")\
                .select("id")\
                .eq("resource_type", data.resource_type)\
                .eq("resource_id", data.resource_id)\
                .eq("grantee_id", data.grantee_id)\
                .eq("permission", data.permission)\
                .maybe_single().execute()
            
            if existing.data:
                return existing.data # Already exists

            new_acl = {
                "organization_id": "unknown", # Needs to be fetched from resource or context. 
                # Ideally service should take org_id context.
                "resource_type": data.resource_type,
                "resource_id": data.resource_id,
                "grantee_type": "user",
                "grantee_id": data.grantee_id,
                "permission": data.permission,
                "granted_by": granted_by
            }
            
            # Fetch Org ID from the resource
            if data.resource_type == 'folder':
                res = supabase.table("folders").select("organization_id").eq("id", data.resource_id).single().execute()
                if res.data: new_acl['organization_id'] = res.data['organization_id']
            elif data.resource_type == 'document':
                res = supabase.table("documents").select("organization_id").eq("id", data.resource_id).single().execute()
                if res.data: new_acl['organization_id'] = res.data['organization_id']

            response = supabase.table("access_control_list").insert(new_acl).execute()
            return response.data
        except Exception as e:
            print(f"Error granting permission: {e}")
            raise e

    @staticmethod
    async def revoke_permission(permission_id: str):
        try:
            supabase.table("access_control_list").delete().eq("id", permission_id).execute()
        except Exception as e:
            print(f"Error revoking permission: {e}")
            raise e
