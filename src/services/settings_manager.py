from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Set, FrozenSet

import httpx
from supabase import Client, create_client

from src.core.config import Settings


@dataclass(frozen=True)
class UserAccessProfile:
    """Summary of group and folder access for a Supabase-authenticated user."""

    user_id: str
    role: str
    normalized_role: str
    direct_group_ids: FrozenSet[str]
    accessible_group_ids: FrozenSet[str]
    folder_paths: FrozenSet[str]
    has_full_access: bool


class SettingsManager:
    """Utility wrapper around Supabase tables needed for the Settings UI."""

    def __init__(self, settings: Settings, *, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self._settings = settings
        self._client: Optional[Client] = None

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

    def get_user_role(self, user_id: str, *, client: Optional[Client] = None) -> str:
        client = client or self._require_client()
        try:
            response = (
                client.table("user_roles")
                .select("role")
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to fetch user role for %s: %s", user_id, exc)
            raise RuntimeError("Failed to fetch user role") from exc

        records = response.data or []
        if not records:
            return "user"
        role = records[0].get("role")
        return str(role) if role else "user"

    def list_groups(self) -> list[dict[str, Any]]:
        client = self._require_client()
        try:
            response = client.table("groups").select("id, name, created_at").order("name").execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to list groups: %s", exc)
            raise RuntimeError("Failed to list groups") from exc
        return response.data or []

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
            try:
                payload = response.json()
            except ValueError:  # pragma: no cover - defensive logging
                self.logger.error("Admin API returned invalid JSON for %s", user_id)
                return None

            if isinstance(payload, dict):
                email = payload.get("email")
                if email:
                    return str(email)
                user = payload.get("user")
                if isinstance(user, dict):
                    user_email = user.get("email")
                    return str(user_email) if user_email else None
        elif response.status_code != 404:
            self.logger.error(
                "Admin API failed for %s: %s %s",
                user_id,
                response.status_code,
                response.text,
            )
        return None

    def _fetch_emails_by_user_ids(self, client: Client, user_ids: set[str]) -> dict[str, Optional[str]]:
        if not user_ids:
            return {}
        try:
            response = (
                client.schema("auth")
                .table("users")
                .select("id, email")
                .in_("id", list(user_ids))
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to fetch user emails for %d users: %s", len(user_ids), exc)
            response = None

        emails: dict[str, Optional[str]] = {}
        if response is not None:
            for record in response.data or []:
                profile_id = record.get("id")
                if not profile_id:
                    continue
                email = record.get("email")
                emails[str(profile_id)] = str(email) if email else None

        missing_ids = [user_id for user_id in user_ids if user_id not in emails]
        for user_id in missing_ids:
            email = self._fetch_email_via_admin_api(user_id)
            if email is not None:
                emails[user_id] = email
        return emails

    def get_user_access_profile(self, user_id: str) -> UserAccessProfile:
        """Return group and folder access information for the supplied user."""

        try:
            client = self._require_client()
        except RuntimeError:
            return UserAccessProfile(
                user_id=user_id,
                role="superadmin",
                normalized_role="superadmin",
                direct_group_ids=frozenset(),
                accessible_group_ids=frozenset(),
                folder_paths=frozenset(),
                has_full_access=True,
            )

        try:
            role = self.get_user_role(user_id, client=client)
        except RuntimeError:
            role = "user"

        normalized_role = str(role or "user").strip().lower().replace("-", "_").replace(" ", "_")
        has_full_access = normalized_role in {"admin", "superadmin"}

        if has_full_access:
            return UserAccessProfile(
                user_id=user_id,
                role=str(role or "admin"),
                normalized_role=normalized_role or "admin",
                direct_group_ids=frozenset(),
                accessible_group_ids=frozenset(),
                folder_paths=frozenset(),
                has_full_access=True,
            )

        direct_group_ids: Set[str] = set()
        try:
            response = (
                client.table("group_members")
                .select("group_id")
                .eq("user_id", user_id)
                .execute()
            )
            for record in response.data or []:
                group_id = record.get("group_id")
                if group_id:
                    direct_group_ids.add(str(group_id))
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to fetch group memberships for %s: %s", user_id, exc)

        accessible_group_ids: Set[str] = set(direct_group_ids)

        adjacency: dict[str, Set[str]] = {}
        try:
            response = (
                client.table("group_access")
                .select("group_id, can_access_group_id")
                .execute()
            )
            for record in response.data or []:
                source = record.get("group_id")
                target = record.get("can_access_group_id")
                if not source or not target:
                    continue
                adjacency.setdefault(str(source), set()).add(str(target))
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to fetch group access rules: %s", exc)

        queue: list[str] = list(accessible_group_ids)
        while queue:
            current = queue.pop()
            for neighbor in adjacency.get(current, set()):
                if neighbor not in accessible_group_ids:
                    accessible_group_ids.add(neighbor)
                    queue.append(neighbor)

        folder_paths: Set[str] = set()
        if accessible_group_ids:
            try:
                response = (
                    client.table("group_folder_permissions")
                    .select("group_id, folder_path")
                    .in_("group_id", list(accessible_group_ids))
                    .execute()
                )
                for record in response.data or []:
                    folder_path = record.get("folder_path")
                    if isinstance(folder_path, str):
                        trimmed = folder_path.strip()
                        folder_paths.add(trimmed)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to fetch folder permissions for %s: %s", user_id, exc)

        return UserAccessProfile(
            user_id=user_id,
            role=str(role or "user"),
            normalized_role=normalized_role or "user",
            direct_group_ids=frozenset(direct_group_ids),
            accessible_group_ids=frozenset(accessible_group_ids),
            folder_paths=frozenset(folder_paths),
            has_full_access=False,
        )

    def list_group_members(self) -> list[dict[str, Any]]:
        client = self._require_client()
        try:
            response = (
                client.table("group_members")
                .select("id, group_id, user_id")
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to list group members: %s", exc)
            raise RuntimeError("Failed to list group members") from exc

        records = response.data or []
        user_ids = {str(record.get("user_id")) for record in records if record.get("user_id")}
        email_lookup = self._fetch_emails_by_user_ids(client, user_ids)

        members: list[dict[str, Any]] = []
        for record in records:
            user_id = record.get("user_id")
            members.append(
                {
                    "id": record.get("id"),
                    "group_id": record.get("group_id"),
                    "user_id": user_id,
                    "email": email_lookup.get(str(user_id)),
                }
            )
        return members

    def list_group_access(self) -> list[dict[str, Any]]:
        client = self._require_client()
        try:
            response = (
                client.table("group_access")
                .select("id, group_id, can_access_group_id, groups!group_access_can_access_group_id_fkey(name)")
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to list group access rules: %s", exc)
            raise RuntimeError("Failed to list group access rules") from exc

        access_entries: list[dict[str, Any]] = []
        for record in response.data or []:
            access_group = record.get("groups")
            access_entries.append(
                {
                    "id": record.get("id"),
                    "group_id": record.get("group_id"),
                    "can_access_group_id": record.get("can_access_group_id"),
                    "can_access_group_name": access_group.get("name") if isinstance(access_group, dict) else None,
                }
            )
        return access_entries

    def list_group_folder_permissions(self) -> list[dict[str, Any]]:
        client = self._require_client()
        try:
            response = (
                client.table("group_folder_permissions")
                .select("id, group_id, folder_path, granted_by, granted_at")
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to list group folder permissions: %s", exc)
            raise RuntimeError("Failed to list group folder permissions") from exc
        return response.data or []

    def list_all_users(self) -> list[dict[str, Any]]:
        client = self._require_client()
        try:
            response = (
                client.table("user_roles")
                .select("id, user_id, role")
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to list user roles: %s", exc)
            raise RuntimeError("Failed to list users") from exc

        records = response.data or []
        user_ids = {str(record.get("user_id")) for record in records if record.get("user_id")}
        email_lookup = self._fetch_emails_by_user_ids(client, user_ids)

        users: list[dict[str, Any]] = []
        for record in records:
            user_id = record.get("user_id")
            email = email_lookup.get(str(user_id))
            users.append(
                {
                    "id": record.get("id"),
                    "user_id": user_id,
                    "role": record.get("role"),
                    "email": email,
                }
            )
        return users

    def _find_user_id_by_email(self, email: str) -> Optional[str]:
        client = self._require_client()
        normalized_email = email.lower()
        try:
            response = (
                client.schema("auth")
                .table("users")
                .select("id")
                .eq("email", normalized_email)
                .limit(1)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to look up user in auth.users for %s: %s", email, exc)
            response = None

        if response is not None:
            records = response.data or []
            if records:
                user_id = records[0].get("id")
                if user_id:
                    return str(user_id)

        # Fallback: attempt to use the admin API when enabled
        if self._settings.supabase_url and self._settings.supabase_key:
            url = f"{self._settings.supabase_url}/auth/v1/admin/users"
            headers = {
                "apikey": self._settings.supabase_key,
                "Authorization": f"Bearer {self._settings.supabase_key}",
                "Content-Type": "application/json",
            }
            try:
                admin_response = httpx.get(
                    url,
                    headers=headers,
                    params={"email": normalized_email},
                    timeout=10.0,
                )
                if admin_response.status_code == 200:
                    payload = admin_response.json()
                    if isinstance(payload, dict):
                        users = payload.get("users")
                        if isinstance(users, list) and users:
                            user_id = users[0].get("id")
                            if user_id:
                                return str(user_id)
                elif admin_response.status_code == 404:
                    return None
                else:  # pragma: no cover - defensive logging
                    self.logger.error(
                        "Admin API email lookup failed for %s: %s %s",
                        email,
                        admin_response.status_code,
                        admin_response.text,
                    )
            except httpx.HTTPError as exc:  # pragma: no cover - defensive logging
                self.logger.error("Admin API lookup failed for %s: %s", email, exc)

        return None

    def assign_user_role(self, *, user_id: str, role: str, assigned_by: Optional[str]) -> None:
        client = self._require_client()
        payload = {
            "user_id": user_id,
            "role": role,
        }
        if assigned_by:
            payload["assigned_by"] = assigned_by
        try:
            client.table("user_roles").upsert(payload, on_conflict="user_id").execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to upsert role %s for %s: %s", role, user_id, exc)
            raise RuntimeError("Failed to assign user role") from exc

    def add_user_to_groups(self, *, user_id: str, group_ids: Sequence[str], added_by: Optional[str]) -> None:
        if not group_ids:
            return
        client = self._require_client()
        payload = []
        for group_id in group_ids:
            entry = {
                "group_id": group_id,
                "user_id": user_id,
            }
            if added_by:
                entry["added_by"] = added_by
            payload.append(entry)

        try:
            client.table("group_members").upsert(payload, on_conflict="group_id,user_id").execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to upsert group memberships for %s: %s", user_id, exc)
            raise RuntimeError("Failed to add user to groups") from exc

    def grant_group_folder_permissions(
        self,
        *,
        group_ids: Sequence[str],
        folder_paths: Sequence[str],
        granted_by: Optional[str],
    ) -> None:
        if not group_ids or not folder_paths:
            return

        client = self._require_client()
        payload = []
        for group_id in group_ids:
            for folder_path in folder_paths:
                entry = {
                    "group_id": group_id,
                    "folder_path": folder_path,
                }
                if granted_by:
                    entry["granted_by"] = granted_by
                payload.append(entry)

        try:
            client.table("group_folder_permissions").upsert(
                payload,
                on_conflict="group_id,folder_path",
            ).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to upsert folder permissions: %s", exc)
            raise RuntimeError("Failed to grant folder permissions") from exc

    def create_user_assignments(
        self,
        *,
        email: str,
        role: Optional[str],
        group_ids: Sequence[str],
        folder_paths: Sequence[str],
        acting_user_id: Optional[str],
    ) -> dict[str, Any]:
        user_id = self._find_user_id_by_email(email)
        if not user_id:
            raise RuntimeError("No Supabase profile found for the provided email")

        if role:
            self.assign_user_role(user_id=user_id, role=role, assigned_by=acting_user_id)
        self.add_user_to_groups(user_id=user_id, group_ids=group_ids, added_by=acting_user_id)
        self.grant_group_folder_permissions(
            group_ids=group_ids,
            folder_paths=folder_paths,
            granted_by=acting_user_id,
        )
        return {"user_id": user_id}

    def replace_group_access(self, *, group_id: str, accessible_group_ids: Sequence[str], acting_user_id: Optional[str]) -> None:
        client = self._require_client()
        try:
            client.table("group_access").delete().eq("group_id", group_id).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to clear group access for %s: %s", group_id, exc)
            raise RuntimeError("Failed to update group access") from exc

        if not accessible_group_ids:
            return

        payload = []
        for allowed_group in accessible_group_ids:
            entry = {
                "group_id": group_id,
                "can_access_group_id": allowed_group,
            }
            if acting_user_id:
                entry["granted_by"] = acting_user_id
            payload.append(entry)

        try:
            client.table("group_access").upsert(payload, on_conflict="group_id,can_access_group_id").execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to insert group access for %s: %s", group_id, exc)
            raise RuntimeError("Failed to persist group access") from exc

    def replace_group_folder_permissions(
        self,
        *,
        group_id: str,
        folder_paths: Sequence[str],
        acting_user_id: Optional[str],
    ) -> None:
        client = self._require_client()
        try:
            client.table("group_folder_permissions").delete().eq("group_id", group_id).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to clear folder permissions for %s: %s", group_id, exc)
            raise RuntimeError("Failed to update folder permissions") from exc

        if not folder_paths:
            return

        payload = []
        for folder_path in folder_paths:
            entry = {
                "group_id": group_id,
                "folder_path": folder_path,
            }
            if acting_user_id:
                entry["granted_by"] = acting_user_id
            payload.append(entry)

        try:
            client.table("group_folder_permissions").upsert(
                payload,
                on_conflict="group_id,folder_path",
            ).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to insert folder permissions for %s: %s", group_id, exc)
            raise RuntimeError("Failed to persist folder permissions") from exc

    def delete_user(self, *, target_user_id: str, acting_user_id: Optional[str]) -> None:
        if acting_user_id and target_user_id == acting_user_id:
            raise RuntimeError("You cannot remove your own account")

        client = self._require_client()
        try:
            response = (
                client.table("user_roles")
                .select("role")
                .eq("user_id", target_user_id)
                .limit(1)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to inspect role for %s: %s", target_user_id, exc)
            raise RuntimeError("Failed to delete user") from exc

        records = response.data or []
        existing_role = records[0].get("role") if records else None
        if existing_role and str(existing_role).lower().replace("-", "_") == "superadmin":
            raise RuntimeError("Cannot delete a superadmin account")

        try:
            client.table("user_roles").delete().eq("user_id", target_user_id).execute()
            client.table("group_members").delete().eq("user_id", target_user_id).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to remove user %s: %s", target_user_id, exc)
            raise RuntimeError("Failed to delete user") from exc
