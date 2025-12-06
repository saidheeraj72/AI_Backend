from __future__ import annotations

import logging
from typing import Any, Optional, List, Dict
from uuid import UUID

from supabase import Client, create_client

from src.core.config import Settings
from src.models.schemas import (
    Organization,
    Branch,
    Role,
    OrganizationMember,
    BranchMember,
    Group,
    Profile
)

class OrganizationService:
    def __init__(self, settings: Settings, *, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self._settings = settings
        self._client: Optional[Client] = None
        
        if settings.supabase_url and settings.supabase_key:
            try:
                self._client = create_client(settings.supabase_url, settings.supabase_key)
            except Exception as exc:
                self.logger.error("Failed to initialise Supabase client: %s", exc)
        else:
            self.logger.warning("Supabase credentials missing")

    def _require_client(self) -> Client:
        if self._client is None:
            raise RuntimeError("Supabase client is not configured")
        return self._client

    def get_organization(self, org_id: UUID) -> Optional[Organization]:
        client = self._require_client()
        try:
            response = client.table("organizations").select("*").eq("id", str(org_id)).single().execute()
            if response.data:
                return Organization(**response.data)
        except Exception as exc:
            self.logger.error(f"Error fetching organization {org_id}: {exc}")
        return None

    def get_organization_by_slug(self, slug: str) -> Optional[Organization]:
        client = self._require_client()
        try:
            response = client.table("organizations").select("*").eq("slug", slug).single().execute()
            if response.data:
                return Organization(**response.data)
        except Exception as exc:
            self.logger.error(f"Error fetching organization by slug {slug}: {exc}")
        return None

    def list_user_organizations(self, user_id: UUID) -> List[Organization]:
        client = self._require_client()
        try:
            # Join organization_members with organizations
            response = client.table("organization_members").select("organization_id, organizations(*)").eq("user_id", str(user_id)).execute()
            orgs = []
            for record in response.data or []:
                if record.get("organizations"):
                    orgs.append(Organization(**record["organizations"]))
            return orgs
        except Exception as exc:
            self.logger.error(f"Error listing user organizations: {exc}")
            return []

    def list_branches(self, org_id: UUID) -> List[Branch]:
        client = self._require_client()
        try:
            # 1. Fetch branches
            response = client.table("branches").select("*").eq("organization_id", str(org_id)).execute()
            branches_data = response.data or []
            
            if not branches_data:
                return []
                
            # 2. Fetch member counts for these branches
            # We can use .select(count='exact') with head=True if we did one by one, but that's N queries.
            # Instead, we fetch all branch_members for this org's branches and count in memory
            # or use a view/RPC. For now, fetching branch_id from branch_members is efficient enough for typical usage.
            branch_ids = [b['id'] for b in branches_data]
            
            members_response = client.table("branch_members").select("branch_id").in_("branch_id", branch_ids).execute()
            members_data = members_response.data or []
            
            counts = {}
            for m in members_data:
                bid = m['branch_id']
                counts[bid] = counts.get(bid, 0) + 1
                
            result = []
            for b in branches_data:
                b['user_count'] = counts.get(b['id'], 0)
                result.append(Branch(**b))
                
            return result
        except Exception as exc:
            self.logger.error(f"Error listing branches for org {org_id}: {exc}")
            return []

    def get_user_branch_memberships(self, user_id: UUID, org_id: UUID) -> List[BranchMember]:
        client = self._require_client()
        try:
            branches = self.list_branches(org_id)
            branch_ids = [str(b.id) for b in branches]
            
            if not branch_ids:
                return []

            response = client.table("branch_members").select("*").eq("user_id", str(user_id)).in_("branch_id", branch_ids).execute()
            return [BranchMember(**item) for item in response.data or []]
        except Exception as exc:
            self.logger.error(f"Error fetching branch memberships: {exc}")
            return []

    def get_user_roles(self, user_id: UUID, org_id: UUID) -> List[Role]:
        client = self._require_client()
        roles = []
        try:
            # Org level role
            org_member_response = client.table("organization_members").select("role_id, roles(*)").eq("organization_id", str(org_id)).eq("user_id", str(user_id)).maybe_single().execute()
            if org_member_response.data and org_member_response.data.get("roles"):
                roles.append(Role(**org_member_response.data["roles"]))

            # Branch level roles
            branch_members = self.get_user_branch_memberships(user_id, org_id)
            branch_role_ids = [str(bm.role_id) for bm in branch_members if bm.role_id]
            
            if branch_role_ids:
                roles_response = client.table("roles").select("*").in_("id", branch_role_ids).execute()
                roles.extend([Role(**item) for item in roles_response.data or []])
                
            return roles
        except Exception as exc:
            self.logger.error(f"Error fetching user roles: {exc}")
            return []

    def create_organization(self, name: str, slug: str, domain: Optional[str] = None) -> Organization:
        client = self._require_client()
        payload = {"name": name, "slug": slug, "domain": domain}
        response = client.table("organizations").insert(payload).execute()
        return Organization(**response.data[0])

    def add_user_to_organization(self, org_id: UUID, user_id: UUID, role_id: Optional[UUID] = None) -> OrganizationMember:
        client = self._require_client()
        payload = {
            "organization_id": str(org_id),
            "user_id": str(user_id),
            "role_id": str(role_id) if role_id else None
        }
        response = client.table("organization_members").insert(payload).execute()
        return OrganizationMember(**response.data[0])
    
    def create_branch(self, org_id: UUID, name: str, code: str, location: str, timezone: str = "UTC") -> Branch:
        client = self._require_client()
        payload = {
            "organization_id": str(org_id),
            "name": name,
            "code": code,
            "location": location,
            "timezone": timezone
        }
        response = client.table("branches").insert(payload).execute()
        return Branch(**response.data[0])

    def add_user_to_branch(self, branch_id: UUID, user_id: UUID, role_id: Optional[UUID] = None, is_primary: bool = False) -> BranchMember:
        client = self._require_client()
        payload = {
            "branch_id": str(branch_id),
            "user_id": str(user_id),
            "role_id": str(role_id) if role_id else None,
            "is_primary_branch": is_primary
        }
        response = client.table("branch_members").insert(payload).execute()
        return BranchMember(**response.data[0])

    def list_roles(self, org_id: Optional[UUID] = None) -> List[Role]:
        client = self._require_client()
        query = client.table("roles").select("*")
        if org_id:
            # Schema defines organization_id as NOT NULL.
            # We explicitly fetch roles for this org. 
            # If global roles (NULL org_id) are introduced later despite schema, 
            # we can re-introduce logic or separate query.
            query = query.eq("organization_id", str(org_id))
        else:
            # Fallback or system admin view?
            pass
            
        response = query.execute()
        return [Role(**item) for item in response.data or []]

    def create_role(self, org_id: UUID, name: str, permissions: Dict[str, bool]) -> Role:
        client = self._require_client()
        payload = {
            "organization_id": str(org_id),
            "name": name,
            "permissions": permissions,
            "is_system_role": False
        }
        response = client.table("roles").insert(payload).execute()
        return Role(**response.data[0])

    def delete_role(self, role_id: UUID, org_id: UUID) -> None:
        client = self._require_client()
        # Only allow deleting roles belonging to this org (and not system roles ideally, handled by RLS or check)
        client.table("roles").delete().eq("id", str(role_id)).eq("organization_id", str(org_id)).execute()

    def update_role(self, role_id: UUID, org_id: UUID, name: str, permissions: Dict[str, bool]) -> Optional[Role]:
        client = self._require_client()
        payload = {
            "name": name,
            "permissions": permissions
        }
        response = client.table("roles").update(payload).eq("id", str(role_id)).eq("organization_id", str(org_id)).execute()
        if response.data:
            return Role(**response.data[0])
        return None
    
    def update_branch(self, branch_id: UUID, org_id: UUID, name: str, code: str, location: str, timezone: str) -> Optional[Branch]:
        client = self._require_client()
        payload = {
            "name": name,
            "code": code,
            "location": location,
            "timezone": timezone
        }
        response = client.table("branches").update(payload).eq("id", str(branch_id)).eq("organization_id", str(org_id)).execute()
        if response.data:
            return Branch(**response.data[0])
        return None
        
    def list_group_members(self, org_id: UUID) -> List[dict]:
        client = self._require_client()
        # Get all groups in org
        groups = client.table("groups").select("id").eq("organization_id", str(org_id)).execute()
        group_ids = [g['id'] for g in groups.data or []]
        if not group_ids:
            return []
            
        response = client.table("group_members").select("*, profiles(email)").in_("group_id", group_ids).execute()
        return response.data or []
        
    def list_group_folder_permissions(self, org_id: UUID) -> List[Dict[str, Any]]:
        client = self._require_client()
        try:
             # FIX: Use not.is.null for group_id check
             response = client.table("folder_permissions").select("*, folders!inner(branch_id, branches!inner(organization_id))").not_.is_("group_id", "null").eq("folders.branches.organization_id", str(org_id)).execute()
             return response.data or []
        except Exception as exc:
             self.logger.error(f"Error listing group folder permissions: {exc}")
             return []

    def create_group(self, org_id: UUID, name: str, branch_id: Optional[UUID] = None) -> Group:
        client = self._require_client()
        payload = {
            "organization_id": str(org_id),
            "name": name,
            "branch_id": str(branch_id) if branch_id else None
        }
        response = client.table("groups").insert(payload).execute()
        return Group(**response.data[0])

    def add_user_to_groups(self, user_id: UUID, group_ids: List[UUID], added_by: Optional[UUID] = None) -> None:
        client = self._require_client()
        payload = []
        for gid in group_ids:
            payload.append({
                "group_id": str(gid),
                "user_id": str(user_id),
            })
        if payload:
            client.table("group_members").upsert(payload, on_conflict="group_id, user_id").execute()

    def remove_user_from_group(self, user_id: UUID, group_id: UUID) -> None:
        client = self._require_client()
        client.table("group_members").delete().eq("user_id", str(user_id)).eq("group_id", str(group_id)).execute()
        
    def get_profile(self, user_id: UUID) -> Optional[Profile]:
        client = self._require_client()
        try:
            response = client.table("profiles").select("*").eq("id", str(user_id)).single().execute()
            if response.data:
                return Profile(**response.data)
        except Exception as exc:
            self.logger.warning(f"Profile not found for {user_id}: {exc}")
        return None

    # --- New Methods for Admin Page ---

    def list_users_paginated(self, org_id: UUID, offset: int = 0, limit: int = 10, search: str = "") -> Dict[str, Any]:
        client = self._require_client()
        try:
            # 1. Fetch Profiles (Paginated)
            query = client.table("profiles").select("*", count="exact")
            
            if search:
                query = query.ilike("email", f"%{search}%")
            
            query = query.range(offset, offset + limit - 1)
            query = query.order("email") 
            
            response = query.execute()
            profiles = response.data or []
            total = response.count
            
            if not profiles:
                 return {"users": [], "total": 0}

            profile_ids = [p['id'] for p in profiles]

            # 2. Fetch Membership info for these profiles in the current Org
            members_query = client.table("organization_members")\
                .select("user_id, role_id, roles(name)")\
                .eq("organization_id", str(org_id))\
                .in_("user_id", profile_ids)\
                .execute()
                
            member_map = {m['user_id']: m for m in (members_query.data or [])}
            
            # 3. Fetch Branch info (only for those who are members)
            # We fetch all branches for the org once to map names efficiently
            all_branches = self.list_branches(org_id)
            branch_lookup = {str(b.id): b.name for b in all_branches}

            users = []
            for p in profiles:
                uid = p['id']
                member = member_map.get(uid)
                
                role_name = "None"
                role_id = None
                user_branch_ids = []
                user_branch_names = []

                if member:
                    role_data = member.get("roles")
                    role_name = role_data.get("name") if role_data else "Viewer"
                    role_id = member.get("role_id")
                    
                    # Get branches
                    memberships = self.get_user_branch_memberships(UUID(uid), org_id)
                    user_branch_ids = [str(m.branch_id) for m in memberships]
                    user_branch_names = [branch_lookup.get(bid, "Unknown") for bid in user_branch_ids]
                
                users.append({
                    "id": uid,
                    "user_id": uid,
                    "email": p.get("email"),
                    "full_name": p.get("full_name"),
                    "role": role_name,
                    "role_id": role_id,
                    "branch_ids": user_branch_ids,
                    "branch_names": user_branch_names
                })

            return {"users": users, "total": total}

        except Exception as exc:
            self.logger.error(f"Error listing paginated users: {exc}")
            return {"users": [], "total": 0}

    def list_groups_paginated(self, org_id: UUID, offset: int = 0, limit: int = 10, search: str = "") -> Dict[str, Any]:
        client = self._require_client()
        try:
            query = client.table("groups").select("*", count="exact").eq("organization_id", str(org_id))
            
            if search:
                query = query.ilike("name", f"%{search}%")
                
            query = query.range(offset, offset + limit - 1)
            response = query.execute()
            
            return {"groups": response.data or [], "total": response.count}
        except Exception as exc:
            self.logger.error(f"Error listing paginated groups: {exc}")
            return {"groups": [], "total": 0}

    def get_folder_tree(self, org_id: UUID) -> List[Dict[str, Any]]:
        client = self._require_client()
        try:
             # Get all folders for the org (via branches)
             # Join branches to filter by org
             response = client.table("folders").select("*, branches!inner(organization_id)").eq("branches.organization_id", str(org_id)).execute()
             
             folders = response.data or []
             # Construct tree logic is typically frontend or complex backend.
             # Here we just return the flat list of folders, frontend builds tree or we build it.
             # Frontend `buildTreeFromFlatPaths` expects paths or tree nodes.
             # Our folders have `parent_id`. We can build a tree.
             
             # For simplicity, let's return the flat list and let frontend handle it or return a simplified structure if requested.
             # The frontend endpoint `/settings/folders/tree` expects `folders` (flat) and `folder_tree` (nested).
             # Let's return flat folders, but we need to construct "path" for them if they don't have it.
             # Constructing paths requires traversing parent_id.
             
             folder_map = {f["id"]: f for f in folders}
             
             for f in folders:
                 path_parts = [f["name"]]
                 current = f
                 while current.get("parent_id"):
                     parent = folder_map.get(current["parent_id"])
                     if parent:
                         path_parts.insert(0, parent["name"])
                         current = parent
                     else:
                         break
                 f["path"] = "/".join(path_parts)
                 
             return folders
        except Exception as exc:
            self.logger.error(f"Error fetching folder tree: {exc}")
            return []

    def update_group_folder_permissions(self, group_id: UUID, folder_permissions: List[Dict[str, Any]], org_id: UUID) -> None:
        """
        folder_permissions: List of dicts, e.g.
        [
            {"path": "docs/HR", "permissions": {"can_view": True, "can_upload": False, ...}}
        ]
        """
        client = self._require_client()
        
        # 1. Clear existing permissions for this group
        client.table("folder_permissions").delete().eq("group_id", str(group_id)).execute()
        
        if not folder_permissions:
            return

        # 2. Resolve paths to IDs
        all_folders = self.get_folder_tree(org_id)
        folder_map = {f.get("path"): f["id"] for f in all_folders}
        
        permissions_to_insert = []
        for item in folder_permissions:
            path = item.get("path")
            perms = item.get("permissions", {})
            
            # Check both exact path and stripped (frontend might send either)
            fid = folder_map.get(path) or folder_map.get(str(path).strip("/"))
            
            if fid:
                permissions_to_insert.append({
                    "folder_id": fid,
                    "group_id": str(group_id),
                    "can_view": perms.get("can_view", False),
                    "can_upload": perms.get("can_upload", False),
                    "can_edit": perms.get("can_edit", False),
                    "can_delete": perms.get("can_delete", False),
                })
        
        if permissions_to_insert:
            client.table("folder_permissions").insert(permissions_to_insert).execute()
            
    def get_direct_user_folder_permissions(self, user_id: UUID, org_id: UUID) -> List[Dict[str, Any]]:
        client = self._require_client()
        
        # 1. Get all folders for path mapping
        all_folders = self.get_folder_tree(org_id)
        folder_id_to_path = {f["id"]: f.get("path", "") for f in all_folders}
        org_folder_ids = list(folder_id_to_path.keys())
        
        if not org_folder_ids:
            return []
            
        # 2. Fetch direct user permissions
        perms = client.table("folder_permissions")\
            .select("*")\
            .eq("user_id", str(user_id))\
            .in_("folder_id", org_folder_ids)\
            .execute().data or []
            
        result = []
        for p in perms:
            path = folder_id_to_path.get(p["folder_id"])
            if path:
                result.append({
                    "document_path": path,
                    "has_access": p.get("can_view", False),
                    "can_view": p.get("can_view", False),
                    "can_upload": p.get("can_upload", False),
                    "can_edit": p.get("can_edit", False),
                    "can_delete": p.get("can_delete", False),
                })
        return result

    def get_role_by_name(self, name: str, org_id: Optional[UUID] = None) -> Optional[Role]:
        client = self._require_client()
        query = client.table("roles").select("*").eq("name", name)
        if org_id:
            query = query.eq("organization_id", str(org_id))
        # else: implies strictly global/system role search if we supported it
             
        response = query.limit(1).execute()
        if response.data:
            return Role(**response.data[0])
        return None

    def update_user_role(self, user_id: UUID, org_id: UUID, role_name: str) -> None:
        client = self._require_client()
        role = self.get_role_by_name(role_name, org_id)
        if not role:
            raise ValueError(f"Role {role_name} not found")
            
        # Upsert organization_members (Add if not exists, Update if exists)
        payload = {
            "organization_id": str(org_id),
            "user_id": str(user_id),
            "role_id": str(role.id)
        }
        client.table("organization_members").upsert(payload, on_conflict="organization_id, user_id").execute()

    def update_user_branch_memberships(self, user_id: UUID, org_id: UUID, branch_ids: List[UUID]) -> None:
        client = self._require_client()
        
        # 1. Get all branches for this org to ensure we only touch relevant branches
        org_branches = self.list_branches(org_id)
        valid_branch_ids = {b.id for b in org_branches}
        
        target_branch_ids = {bid for bid in branch_ids if bid in valid_branch_ids}
        
        # 2. Get current memberships
        current_memberships = self.get_user_branch_memberships(user_id, org_id)
        current_branch_ids = {m.branch_id for m in current_memberships}
        
        # 3. Calculate diff
        to_add = target_branch_ids - current_branch_ids
        to_remove = current_branch_ids - target_branch_ids
        
        # 4. Remove
        if to_remove:
            client.table("branch_members").delete()\
                .eq("user_id", str(user_id))\
                .in_("branch_id", [str(bid) for bid in to_remove])\
                .execute()
                
        # 5. Add
        if to_add:
            payload = [
                {"branch_id": str(bid), "user_id": str(user_id)}
                for bid in to_add
            ]
            client.table("branch_members").insert(payload).execute()

    def get_user_permissions_summary(self, user_id: UUID, org_id: UUID) -> Dict[str, Any]:
        """
        Aggregates permissions from direct assignment and group memberships.
        Returns a dict keyed by folder path.
        """
        client = self._require_client()
        
        # 1. Get all folders with paths
        all_folders = self.get_folder_tree(org_id)
        # Map ID to Path
        folder_id_to_path = {f["id"]: f.get("path", "") for f in all_folders}

        # 2. Get user's groups
        groups_resp = client.table("group_members").select("group_id").eq("user_id", str(user_id)).execute()
        group_ids = [g['group_id'] for g in groups_resp.data or []]
        
        # 3. Fetch permissions
        # Condition: (user_id == X) OR (group_id IN [...])
        query = client.table("folder_permissions").select("*")
        
        # Supabase-py/Postgrest doesn't support complex OR (user_id=X OR group_id=Y) easily in one chained call
        # without raw filter string. 
        # Alternatively, two queries.
        
        # Query 1: Direct user permissions
        user_perms = query.eq("user_id", str(user_id)).execute().data or []
        
        # Query 2: Group permissions (if any groups)
        group_perms = []
        if group_ids:
            group_perms = client.table("folder_permissions").select("*").in_("group_id", group_ids).execute().data or []
            
        all_perms = user_perms + group_perms
        
        # 4. Merge permissions (User override? Or Union? typically Union of allow)
        # We'll take the most permissive.
        merged = {}
        
        for p in all_perms:
            fid = p.get("folder_id")
            path = folder_id_to_path.get(fid)
            
            if not path:
                continue
                
            if path not in merged:
                merged[path] = {
                    "can_view": False,
                    "can_upload": False,
                    "can_edit": False,
                    "can_delete": False
                }
            
            # Union logic (OR)
            merged[path]["can_view"] = merged[path]["can_view"] or p.get("can_view", False)
            merged[path]["can_upload"] = merged[path]["can_upload"] or p.get("can_upload", False)
            merged[path]["can_edit"] = merged[path]["can_edit"] or p.get("can_edit", False)
            merged[path]["can_delete"] = merged[path]["can_delete"] or p.get("can_delete", False)
            
        return merged

    def update_user_document_permissions(self, user_id: UUID, permissions: List[Dict[str, Any]], org_id: UUID) -> None:
        """
        Updates document-level permissions for a user.
        permissions: List of dicts containing path and boolean flags.
        """
        client = self._require_client()
        
        # 1. Resolve paths to Document IDs
        # Note: frontend sends "document_path" which usually maps to 'directory' + 'filename' or just 'storage_path'.
        # The 'documents' table has 'storage_path'.
        # We assume frontend sends paths that match 'storage_path' or directory.
        # The schema says `document_permissions` links to `documents.id`.
        
        # Fetch all documents for the org to map paths. 
        # Optimized: fetch only needed paths if list is short, but usually we need ID lookup.
        
        # Get all docs for org
        docs_resp = client.table("documents").select("id, storage_path, branch_id, branches!inner(organization_id)").eq("branches.organization_id", str(org_id)).execute()
        docs = docs_resp.data or []
        
        # Map storage_path -> id
        # Also handle cases where path might be just a folder path? 
        # "document_permissions" usually implies files. 
        # But the UI "FolderPermissionNode" suggests we might be setting permissions on folders acting as containers?
        # Schema has both `folder_permissions` and `document_permissions`.
        # If the UI is setting permissions on *folders* for a specific *user*, we should use `folder_permissions` with `user_id` set.
        # If the UI is setting permissions on *files*, we use `document_permissions`.
        
        # Looking at the UI code: `DocumentPermissionsFullView` uses `folderTree`.
        # And `handleSaveDocumentPermissions` sends `document_path`.
        # If these are folders, we should be updating `folder_permissions` table where `user_id` is set (instead of `group_id`).
        
        # Let's assume the intent is Folder Permissions for a specific User (Personal overrides).
        # This matches `folder_permissions` schema: (folder_id, user_id OR group_id).
        
        # So we reuse similar logic to group folder permissions but for user_id.
        
        # 1. Clear existing user folder permissions for this org's folders
        # We need to delete permissions where user_id = X and folder_id IN (org folders)
        
        # Get all folder IDs for org
        all_folders = self.get_folder_tree(org_id)
        org_folder_ids = [f["id"] for f in all_folders]
        folder_map = {f.get("path"): f["id"] for f in all_folders}
        
        if not org_folder_ids:
            return

        # Delete existing
        client.table("folder_permissions")\
            .delete()\
            .eq("user_id", str(user_id))\
            .in_("folder_id", org_folder_ids)\
            .execute()
            
        # 2. Insert new
        to_insert = []
        for p in permissions:
            path = p.get("document_path") or p.get("path")
            if not path:
                continue
                
            fid = folder_map.get(path) or folder_map.get(path.strip("/"))
            if fid:
                to_insert.append({
                    "folder_id": fid,
                    "user_id": str(user_id),
                    "can_view": p.get("has_access", False) or p.get("can_view", False),
                    # Add other granular flags if UI supports them for user-specific overrides
                    # UI DocumentPermissionsFullView currently just sends "has_access" (view). 
                    # If we want granular, we'd need to update UI. For now assuming "has_access" -> view.
                    # If the schema allows others, we default them to false or match logic.
                    "can_upload": p.get("can_upload", False),
                    "can_edit": p.get("can_edit", False),
                    "can_delete": p.get("can_delete", False),
                })
        
        if to_insert:
            client.table("folder_permissions").insert(to_insert).execute()