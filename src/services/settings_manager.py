from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import httpx
from supabase import Client, create_client

from src.core.config import Settings


@dataclass(frozen=True)
class RoleDefinition:
    id: str
    name: str
    scope: str
    permissions: frozenset[str]


@dataclass(frozen=True)
class AssignmentRecord:
    id: str
    role: RoleDefinition
    branch_id: Optional[str]
    group_id: Optional[str]
    is_primary: bool


@dataclass(frozen=True)
class UserAccessProfile:
    user_id: str
    org_id: Optional[str]
    role: str
    normalized_role: str
    branch_id: Optional[str]
    branch_name: Optional[str]
    branch_ids: frozenset[str]
    branch_names: frozenset[str]
    direct_group_ids: frozenset[str]
    accessible_group_ids: frozenset[str]
    folder_paths: frozenset[str]
    has_full_access: bool
    permission_codes: frozenset[str]
    assignments: tuple[AssignmentRecord, ...]


class SettingsManager:
    """Wrapper around Supabase settings tables aligned with the new RBAC schema."""

    def __init__(self, settings: Settings, *, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self._settings = settings
        self._client: Optional[Client] = None
        self._org_id_cache: Optional[str] = None
        self._org_domain_cache: dict[str, str] = {}

        if settings.supabase_url and settings.supabase_key:
            try:
                self._client = create_client(settings.supabase_url, settings.supabase_key)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to initialise Supabase client for settings manager: %s", exc)
        else:
            self.logger.warning("Supabase credentials missing; settings manager disabled")

    def _require_client(self) -> Client:
        if self._client is None:
            raise RuntimeError("Supabase client is not configured")
        return self._client

    def _get_default_org_id(self, *, client: Optional[Client] = None) -> str:
        if self._org_id_cache:
            return self._org_id_cache

        configured = self._settings.default_org_id
        if configured:
            self._org_id_cache = configured
            return configured

        client = client or self._require_client()
        try:
            response = client.table("organizations").select("id").limit(1).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError("Failed to resolve organization identifier") from exc

        records = response.data or []
        if not records or not records[0].get("id"):
            raise RuntimeError("No organizations are configured. Please seed the organizations table.")

        org_id = str(records[0]["id"])
        self._org_id_cache = org_id
        return org_id

    def _fetch_roles(self, role_ids: Sequence[str], *, client: Client) -> dict[str, RoleDefinition]:
        if not role_ids:
            return {}

        sanitized = sorted({role_id for role_id in role_ids if role_id})
        if not sanitized:
            return {}

        try:
            response = (
                client.table("roles")
                .select("id, name, scope")
                .in_("id", sanitized)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to fetch roles: %s", exc)
            return {}

        role_map: dict[str, RoleDefinition] = {}
        permission_map = self._fetch_role_permissions(sanitized, client=client)

        for record in response.data or []:
            role_id = str(record.get("id"))
            if not role_id:
                continue
            role_map[role_id] = RoleDefinition(
                id=role_id,
                name=str(record.get("name") or "user"),
                scope=str(record.get("scope") or "organization"),
                permissions=frozenset(permission_map.get(role_id, set())),
            )

        return role_map

    def _fetch_role_permissions(self, role_ids: Sequence[str], *, client: Client) -> dict[str, set[str]]:
        if not role_ids:
            return {}

        sanitized = sorted({role_id for role_id in role_ids if role_id})
        if not sanitized:
            return {}

        try:
            response = (
                client.table("role_permissions")
                .select("role_id, permission_id")
                .in_("role_id", sanitized)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to fetch role permission mappings: %s", exc)
            return {}

        permission_ids = {str(record.get("permission_id")) for record in response.data or [] if record.get("permission_id")}
        permission_lookup = self._fetch_permission_codes(permission_ids, client=client)

        result: dict[str, set[str]] = {role_id: set() for role_id in sanitized}
        for record in response.data or []:
            role_id = str(record.get("role_id")) if record.get("role_id") else None
            permission_id = str(record.get("permission_id")) if record.get("permission_id") else None
            if not role_id or not permission_id:
                continue
            code = permission_lookup.get(permission_id)
            if code:
                result.setdefault(role_id, set()).add(code)

        return result

    def _fetch_permission_codes(self, permission_ids: set[str], *, client: Client) -> dict[str, str]:
        if not permission_ids:
            return {}

        try:
            response = (
                client.table("permissions")
                .select("id, code")
                .in_("id", sorted(permission_ids))
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to fetch permission codes: %s", exc)
            return {}

        return {
            str(record.get("id")): str(record.get("code"))
            for record in response.data or []
            if record.get("id") and record.get("code")
        }

    def _fetch_branch_names(self, branch_ids: set[str], *, client: Client) -> dict[str, str]:
        if not branch_ids:
            return {}

        try:
            response = (
                client.table("branches")
                .select("id, name")
                .in_("id", sorted(branch_ids))
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to fetch branch names: %s", exc)
            return {}

        return {
            str(record.get("id")): str(record.get("name"))
            for record in response.data or []
            if record.get("id") and record.get("name")
        }

    def _fetch_group_memberships(self, user_id: str, *, client: Client) -> set[str]:
        try:
            response = (
                client.table("group_members")
                .select("group_id")
                .eq("user_id", user_id)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to fetch group memberships: %s", exc)
            return set()

        return {
            str(record.get("group_id"))
            for record in response.data or []
            if record.get("group_id")
        }

    def _build_assignments(
        self,
        records: list[dict[str, Any]],
        *,
        client: Client,
    ) -> tuple[AssignmentRecord, ...]:
        if not records:
            return tuple()

        role_ids = [str(record.get("role_id")) for record in records if record.get("role_id")]
        roles = self._fetch_roles(role_ids, client=client)

        assignments: list[AssignmentRecord] = []
        for record in records:
            role_id = str(record.get("role_id")) if record.get("role_id") else None
            if not role_id:
                continue
            role = roles.get(role_id)
            if role is None:
                continue
            assignments.append(
                AssignmentRecord(
                    id=str(record.get("id")),
                    role=role,
                    branch_id=str(record.get("branch_id")) if record.get("branch_id") else None,
                    group_id=str(record.get("group_id")) if record.get("group_id") else None,
                    is_primary=bool(record.get("is_primary_branch")),
                )
            )

        return tuple(assignments)

    def _normalize_role(self, assignments: Sequence[AssignmentRecord]) -> str:
        # Return the actual role name without normalization
        # If user has multiple roles, return the highest priority one
        if not assignments:
            return "Viewer"

        # Priority order: OrgOwner > BranchManager > Contributor > Viewer
        role_priority = {
            "OrgOwner": 4,
            "BranchManager": 3,
            "Contributor": 2,
            "Viewer": 1,
        }

        highest_role = max(
            assignments,
            key=lambda a: role_priority.get(a.role.name, 0)
        )
        return highest_role.role.name

    def _build_user_profile(self, user_id: str, *, client: Client) -> UserAccessProfile:
        org_id = self._resolve_org_for_user(user_id, client=client)
        try:
            response = (
                client.table("user_assignments")
                .select("id, role_id, branch_id, group_id, is_primary_branch")
                .eq("user_id", user_id)
                .eq("org_id", org_id)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to load user assignments: %s", exc)
            assignments: tuple[AssignmentRecord, ...] = tuple()
        else:
            assignments = self._build_assignments(response.data or [], client=client)

        branch_ids = {assignment.branch_id for assignment in assignments if assignment.branch_id}
        branch_names_map = self._fetch_branch_names({branch_id for branch_id in branch_ids if branch_id}, client=client)
        primary_assignment = next((assignment for assignment in assignments if assignment.is_primary), assignments[0] if assignments else None)
        primary_branch_id = primary_assignment.branch_id if primary_assignment else None
        primary_branch_name = branch_names_map.get(primary_branch_id) if primary_branch_id else None

        group_ids = self._fetch_group_memberships(user_id, client=client)
        accessible_groups = set(group_ids)

        permission_codes = frozenset({code for assignment in assignments for code in assignment.role.permissions})
        normalized_role = self._normalize_role(assignments)

        return UserAccessProfile(
            user_id=user_id,
            org_id=org_id,
            role=assignments[0].role.name if assignments else "Viewer",
            normalized_role=normalized_role,
            branch_id=primary_branch_id,
            branch_name=primary_branch_name,
            branch_ids=frozenset(branch_ids),
            branch_names=frozenset(branch_names_map.get(branch_id, "") for branch_id in branch_ids if branch_id),
            direct_group_ids=frozenset(group_ids),
            accessible_group_ids=frozenset(accessible_groups),
            folder_paths=frozenset(),
            has_full_access=(normalized_role == "OrgOwner" or "ORG_MANAGE" in permission_codes),
            permission_codes=permission_codes,
            assignments=assignments,
        )

    def get_user_access_profile(self, user_id: str) -> UserAccessProfile:
        client = self._require_client()
        try:
            return self._build_user_profile(user_id, client=client)
        except Exception as exc:
            self.logger.error("Failed to build access profile for %s: %s", user_id, exc)
            return UserAccessProfile(
                user_id=user_id,
                org_id=None,
                role="user",
                normalized_role="user",
                branch_id=None,
                branch_name=None,
                branch_ids=frozenset(),
                branch_names=frozenset(),
                direct_group_ids=frozenset(),
                accessible_group_ids=frozenset(),
                folder_paths=frozenset(),
                has_full_access=False,
                permission_codes=frozenset(),
                assignments=tuple(),
            )

    def get_user_role(self, user_id: str, *, client: Optional[Client] = None) -> str:
        client = client or self._require_client()
        profile = self._build_user_profile(user_id, client=client)
        return profile.normalized_role

    def get_user_branch_assignments(self, user_id: str, *, client: Optional[Client] = None) -> tuple[list[str], list[str]]:
        client = client or self._require_client()
        profile = self._build_user_profile(user_id, client=client)
        branch_ids = list(profile.branch_ids)
        branch_names = list(profile.branch_names)
        return branch_ids, branch_names

    def list_roles(self) -> list[dict[str, Any]]:
        """List all platform-level roles (org_id is null)"""
        client = self._require_client()
        try:
            response = (
                client.table("roles")
                .select("id, name, description, scope")
                .is_("org_id", "null")
                .order("name")
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError("Failed to list roles") from exc
        return response.data or []

    def list_branches(self, *, org_id: Optional[str] = None) -> list[dict[str, Any]]:
        client = self._require_client()
        resolved_org = org_id or self._get_default_org_id(client=client)
        try:
            response = (
                client.table("branches")
                .select("id, name, created_at, code")
                .eq("org_id", resolved_org)
                .order("name")
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError("Failed to list branches") from exc
        return response.data or []

    def create_branch(self, *, name: str, org_id: Optional[str] = None) -> str:
        client = self._require_client()
        resolved_org = org_id or self._get_default_org_id(client=client)
        payload = {
            "name": name,
            "org_id": resolved_org,
        }
        try:
            response = client.table("branches").insert(payload).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError("Failed to create branch") from exc
        record = (response.data or [None])[0]
        if not record or not record.get("id"):
            raise RuntimeError("Supabase did not return a branch identifier")
        return str(record["id"])

    def list_groups_paginated(self, *, offset: int, limit: int, search: str, requesting_user_id: Optional[str]) -> dict[str, Any]:
        client = self._require_client()
        if requesting_user_id:
            org_id = self._resolve_org_for_user(requesting_user_id, client=client)
        else:
            org_id = self._get_default_org_id(client=client)

        query = (
            client.table("groups")
            .select("id, name, created_at, description")
            .eq("org_id", org_id)
            .order("name")
        )

        if search:
            query = query.ilike("name", f"%{search}%")

        try:
            response = query.range(offset, offset + limit - 1).execute()
            count_response = (
                client.table("groups")
                .select("id", count="exact")
                .eq("org_id", org_id)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError("Failed to list groups") from exc

        return {
            "groups": response.data or [],
            "total": count_response.count or 0,
            "offset": offset,
            "limit": limit,
        }

    def list_users_paginated(
        self,
        *,
        offset: int,
        limit: int,
        search: str,
        requesting_user_id: Optional[str],
    ) -> dict[str, Any]:
        client = self._require_client()
        if requesting_user_id:
            org_id = self._resolve_org_for_user(requesting_user_id, client=client)
        else:
            org_id = self._get_default_org_id(client=client)

        try:
            response = (
                client.table("user_assignments")
                .select("id, user_id, role_id, branch_id")
                .eq("org_id", org_id)
                .order("created_at", desc=True)
                .range(offset, offset + limit - 1)
                .execute()
            )
            count_response = (
                client.table("user_assignments")
                .select("id", count="exact")
                .eq("org_id", org_id)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError("Failed to list users") from exc

        ordered_user_ids: list[str] = []
        for record in response.data or []:
            user_id = record.get("user_id")
            if not user_id:
                continue
            normalized_user = str(user_id)
            if normalized_user not in ordered_user_ids:
                ordered_user_ids.append(normalized_user)

        assignments_by_user: dict[str, list[dict[str, Any]]] = {}
        if ordered_user_ids:
            try:
                all_assignments_response = (
                    client.table("user_assignments")
                    .select("id, user_id, role_id, branch_id")
                    .eq("org_id", org_id)
                    .in_("user_id", sorted(ordered_user_ids))
                    .execute()
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                raise RuntimeError("Failed to load user assignments") from exc
            for record in all_assignments_response.data or []:
                user_id = record.get("user_id")
                if not user_id:
                    continue
                assignments_by_user.setdefault(str(user_id), []).append(record)

        role_ids = {str(record.get("role_id")) for record_list in assignments_by_user.values() for record in record_list if record.get("role_id")}
        roles = self._fetch_roles(sorted(role_ids), client=client)
        branch_ids = {str(record.get("branch_id")) for record_list in assignments_by_user.values() for record in record_list if record.get("branch_id")}
        branch_names = self._fetch_branch_names(branch_ids, client=client)

        email_lookup = self._fetch_emails_by_user_ids(client, set(assignments_by_user.keys()))

        users: list[dict[str, Any]] = []
        for user_id in ordered_user_ids:
            records = assignments_by_user.get(user_id, [])
            role = roles.get(str(records[0].get("role_id"))) if records else None
            branch_list = [branch_names.get(str(record.get("branch_id"))) for record in records if record.get("branch_id")]
            branch_ids_list = [str(record.get("branch_id")) for record in records if record.get("branch_id")]
            users.append(
                {
                    "user_id": user_id,
                    "role": role.name if role else "user",
                    "email": email_lookup.get(user_id),
                    "branch_ids": branch_ids_list,
                    "branch_names": [name for name in branch_list if name],
                }
            )

        if search:
            needle = search.lower()
            users = [
                user
                for user in users
                if needle in (user.get("email") or "").lower()
                or needle in (user.get("role") or "").lower()
            ]

        return {
            "users": users,
            "total": count_response.count or 0,
            "offset": offset,
            "limit": limit,
        }

    def list_group_members(self, requesting_user_id: Optional[str] = None, *, org_id: Optional[str] = None) -> list[dict[str, Any]]:
        client = self._require_client()
        resolved_org = org_id
        if not resolved_org and requesting_user_id:
            try:
                resolved_org = self._resolve_org_for_user(requesting_user_id, client=client)
            except Exception:
                resolved_org = None

        try:
            response = client.table("group_members").select("id, group_id, user_id").execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError("Failed to list group members") from exc

        user_ids = {str(record.get("user_id")) for record in response.data or [] if record.get("user_id")}
        emails = self._fetch_emails_by_user_ids(client, user_ids)

        allowed_group_ids: Optional[Set[str]] = None
        if resolved_org:
            try:
                group_resp = client.table("groups").select("id").eq("org_id", resolved_org).execute()
                allowed_group_ids = {str(entry.get("id")) for entry in group_resp.data or [] if entry.get("id")}
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.warning("Failed to resolve groups for org %s: %s", resolved_org, exc)

        members: list[dict[str, Any]] = []
        for record in response.data or []:
            user_id = record.get("user_id")
            group_id = record.get("group_id")
            if not user_id or not group_id:
                continue
            if allowed_group_ids is not None and group_id not in allowed_group_ids:
                continue
            members.append(
                {
                    "id": record.get("id"),
                    "group_id": group_id,
                    "user_id": user_id,
                    "email": emails.get(str(user_id)),
                }
            )
        return members

    def create_group(self, *, name: str, created_by: Optional[str], branch_id: Optional[str] = None, org_id: Optional[str] = None) -> str:
        client = self._require_client()
        resolved_org = org_id or self._get_default_org_id(client=client)

        payload = {
            "name": name,
            "org_id": resolved_org,
        }
        if created_by:
            payload["created_by"] = created_by
        if branch_id:
            payload["branch_id"] = branch_id

        try:
            response = client.table("groups").insert(payload).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError("Failed to create group") from exc

        record = (response.data or [None])[0]
        if not record or not record.get("id"):
            raise RuntimeError("Supabase did not return a group identifier")
        return str(record["id"])

    def add_user_to_groups(self, *, user_id: str, group_ids: Sequence[str], added_by: Optional[str]) -> None:
        client = self._require_client()
        payload = []
        for group_id in group_ids:
            payload.append(
                {
                    "group_id": group_id,
                    "user_id": user_id,
                    "added_by": added_by,
                }
            )
        if not payload:
            return
        try:
            client.table("group_members").upsert(payload, on_conflict="group_id,user_id").execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError("Failed to add user to groups") from exc

    def remove_user_from_group(self, *, user_id: str, group_id: str) -> None:
        client = self._require_client()
        try:
            client.table("group_members").delete().eq("user_id", user_id).eq("group_id", group_id).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError("Failed to remove user from group") from exc

    def _find_role_id_by_name(self, role_name: Optional[str], *, client: Client, org_id: str) -> Optional[str]:
        if not role_name:
            return None

        def _query(target_org: Optional[str]):
            query = client.table("roles").select("id").ilike("name", role_name).limit(1)
            if target_org is None:
                query = query.is_("org_id", None)
            else:
                query = query.eq("org_id", target_org)
            return query.execute()

        try:
            response = _query(org_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to lookup role %s: %s", role_name, exc)
            return None

        records = response.data or []
        if not records:
            try:
                response = _query(None)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to lookup global role %s: %s", role_name, exc)
                return None
            records = response.data or []
            if not records:
                return None
        role_id = records[0].get("id")
        return str(role_id) if role_id else None

    def _ensure_user_profile(self, user_id: str, *, org_id: str, client: Client) -> None:
        payload = {
            "user_id": user_id,
            "org_id": org_id,
        }
        try:
            client.table("user_profiles").upsert(payload, on_conflict="user_id").execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Failed to upsert user profile for %s: %s", user_id, exc)

    def _find_user_id_by_email(self, email: str) -> Optional[str]:
        supabase_url = self._settings.supabase_url
        supabase_key = self._settings.supabase_key
        if not supabase_url or not supabase_key:
            return None

        url = f"{supabase_url}/auth/v1/admin/users"
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
        }
        params = {"email": email}

        try:
            response = httpx.get(url, headers=headers, params=params, timeout=10.0)
        except httpx.HTTPError as exc:  # pragma: no cover - defensive logging
            self.logger.error("Admin API request failed for email %s: %s", email, exc)
            return None

        if response.status_code != 200:
            return None

        data = response.json()
        users = data.get("users") if isinstance(data, dict) else None
        if not isinstance(users, list):
            return None
        for entry in users:
            entry_email = entry.get("email")
            if entry_email and entry_email.lower() == email.lower():
                return str(entry.get("id"))
        return None

    def _fetch_emails_by_user_ids(self, client: Client, user_ids: set[str]) -> dict[str, Optional[str]]:
        emails: dict[str, Optional[str]] = {}
        for user_id in user_ids:
            email = self._fetch_email_via_admin_api(user_id)
            if email is not None:
                emails[user_id] = email
        return emails

    def _extract_email_domain(self, email: Optional[str]) -> Optional[str]:
        if not email or "@" not in email:
            return None
        return email.split("@", 1)[1].lower().strip()

    def _get_or_create_org_for_domain(self, domain: str, *, client: Client) -> Optional[str]:
        if not domain:
            return None
        cached = self._org_domain_cache.get(domain)
        if cached:
            return cached

        try:
            response = (
                client.table("organizations")
                .select("id, metadata")
                .contains("metadata", {"domain": domain})
                .limit(1)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to lookup organization for domain %s: %s", domain, exc)
            return None

        records = response.data or []
        if not records:
            payload = {
                "name": domain,
                "metadata": {"domain": domain},
            }
            try:
                response = client.table("organizations").insert(payload).execute()
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to create organization for domain %s: %s", domain, exc)
                return None
            records = response.data or []
            if not records:
                return None

        org_id = records[0].get("id")
        if not org_id:
            return None
        resolved = str(org_id)
        self._org_domain_cache[domain] = resolved
        return resolved

    def _organization_has_assignments(self, org_id: str, *, client: Client) -> bool:
        try:
            response = (
                client.table("user_assignments")
                .select("id")
                .eq("org_id", org_id)
                .limit(1)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Failed to inspect assignments for org %s: %s", org_id, exc)
            return False
        return bool(response.data)

    def _assign_default_role_for_user(self, user_id: str, org_id: str, *, client: Client) -> None:
        try:
            response = (
                client.table("user_assignments")
                .select("id")
                .eq("user_id", user_id)
                .eq("org_id", org_id)
                .limit(1)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Failed to check existing assignments for %s: %s", user_id, exc)
            return

        if response.data:
            return

        role_name = "OrgOwner" if not self._organization_has_assignments(org_id, client=client) else "Viewer"
        role_id = self._find_role_id_by_name(role_name, client=client, org_id=org_id)
        if not role_id:
            self.logger.warning("Role %s not found when assigning %s to org %s", role_name, user_id, org_id)
            return

        payload = {
            "user_id": user_id,
            "org_id": org_id,
            "role_id": role_id,
            "is_primary_branch": True,
        }
        try:
            client.table("user_assignments").insert(payload).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Failed to assign default role %s to %s: %s", role_name, user_id, exc)

    def _resolve_org_for_user(self, user_id: str, *, client: Client) -> str:
        try:
            response = (
                client.table("user_profiles")
                .select("org_id")
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Failed to lookup user profile for %s: %s", user_id, exc)
            response = None

        records = response.data if response else []
        if records:
            org_id = records[0].get("org_id")
            if org_id:
                return str(org_id)

        email = self._fetch_email_via_admin_api(user_id)
        domain = self._extract_email_domain(email)
        org_id = self._get_or_create_org_for_domain(domain, client=client)
        if not org_id:
            org_id = self._get_default_org_id(client=client)

        self._ensure_user_profile(user_id, org_id=org_id, client=client)
        self._assign_default_role_for_user(user_id, org_id, client=client)
        return org_id

    def _fetch_email_via_admin_api(self, user_id: str) -> Optional[str]:
        supabase_url = self._settings.supabase_url
        supabase_key = self._settings.supabase_key
        if not supabase_url or not supabase_key:
            return None

        url = f"{supabase_url}/auth/v1/admin/users/{user_id}"
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
        }
        try:
            response = httpx.get(url, headers=headers, timeout=10.0)
        except httpx.HTTPError as exc:  # pragma: no cover - defensive logging
            self.logger.error("Admin API request failed for %s: %s", user_id, exc)
            return None

        if response.status_code == 200:
            payload = response.json()
            return payload.get("email")
        return None

    def create_user_assignments(
        self,
        *,
        email: str,
        role: Optional[str],
        group_ids: Sequence[str],
        folder_paths: Sequence[str],
        acting_user_id: Optional[str],
        branch_ids: Optional[Sequence[str]] = None,
        branch_names: Optional[Sequence[str]] = None,
        org_id: Optional[str] = None,
    ) -> dict[str, Any]:
        client = self._require_client()
        resolved_org = org_id
        if not resolved_org and acting_user_id:
            try:
                resolved_org = self._resolve_org_for_user(acting_user_id, client=client)
            except Exception:
                resolved_org = None
        if not resolved_org:
            resolved_org = self._get_default_org_id(client=client)
        user_id = self._find_user_id_by_email(email)
        if not user_id:
            raise RuntimeError("Could not find a Supabase user with that email")

        role_id = self._find_role_id_by_name(role, client=client, org_id=resolved_org)
        if not role_id:
            raise RuntimeError("Role not found")

        self._ensure_user_profile(user_id, org_id=resolved_org, client=client)

        try:
            delete_query = (
                client.table("user_assignments")
                .delete()
                .eq("user_id", user_id)
                .eq("org_id", resolved_org)
            )
            if branch_ids:
                delete_query = delete_query.in_("branch_id", branch_ids)
            delete_query.execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Failed to clear existing assignments for %s: %s", user_id, exc)

        try:
            primary_check = (
                client.table("user_assignments")
                .select("id")
                .eq("user_id", user_id)
                .eq("org_id", resolved_org)
                .execute()
            )
        except Exception as exc:
            self.logger.warning("Failed to check existing assignments for %s: %s", user_id, exc)
            primary_check = None
        has_primary = bool(primary_check and primary_check.data and len(primary_check.data) > 0)

        payload = []
        target_branch_ids = list(branch_ids or [])
        self.logger.info(
            "Creating user assignments for %s with %d branches: %s",
            user_id,
            len(target_branch_ids),
            target_branch_ids,
        )
        if not target_branch_ids:
            payload.append(
                {
                    "user_id": user_id,
                    "org_id": resolved_org,
                    "role_id": role_id,
                    "is_primary_branch": not has_primary,
                }
            )
        else:
            for index, branch_id in enumerate(target_branch_ids):
                payload.append(
                    {
                        "user_id": user_id,
                        "org_id": resolved_org,
                        "role_id": role_id,
                        "branch_id": branch_id,
                        "is_primary_branch": index == 0 and not has_primary,
                    }
                )

        self.logger.info("Inserting %d user_assignment records: %s", len(payload), payload)
        if payload:
            try:
                result = client.table("user_assignments").insert(payload).execute()
                self.logger.info("Successfully created %d user_assignment records", len(result.data or []))
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to insert user assignments: %s", exc)
                raise RuntimeError("Failed to create role assignments") from exc

        self.add_user_to_groups(user_id=user_id, group_ids=group_ids, added_by=acting_user_id)
        return {"user_id": user_id}

    def delete_user(self, *, target_user_id: str, acting_user_id: Optional[str]) -> None:
        client = self._require_client()
        if acting_user_id:
            try:
                org_id = self._resolve_org_for_user(acting_user_id, client=client)
            except Exception:
                org_id = self._get_default_org_id(client=client)
        else:
            org_id = self._get_default_org_id(client=client)

        try:
            client.table("user_assignments").delete().eq("user_id", target_user_id).eq("org_id", org_id).execute()
            client.table("group_members").delete().eq("user_id", target_user_id).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError("Failed to remove user") from exc
