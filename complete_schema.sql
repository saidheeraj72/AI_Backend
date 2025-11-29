-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

CREATE TABLE public.branch_members (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  branch_id uuid NOT NULL,
  user_id uuid NOT NULL,
  role_id uuid,
  is_primary_branch boolean DEFAULT false,
  joined_at timestamp with time zone DEFAULT now(),
  CONSTRAINT branch_members_pkey PRIMARY KEY (id),
  CONSTRAINT branch_members_branch_id_fkey FOREIGN KEY (branch_id) REFERENCES public.branches(id),
  CONSTRAINT branch_members_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.profiles(id),
  CONSTRAINT branch_members_role_id_fkey FOREIGN KEY (role_id) REFERENCES public.roles(id)
);
CREATE TABLE public.branches (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  organization_id uuid NOT NULL,
  name text NOT NULL,
  code text NOT NULL,
  location text NOT NULL,
  timezone text DEFAULT 'UTC'::text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT branches_pkey PRIMARY KEY (id),
  CONSTRAINT branches_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id)
);
CREATE TABLE public.chat_messages (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  session_id uuid NOT NULL,
  role text NOT NULL CHECK (role = ANY (ARRAY['user'::text, 'assistant'::text, 'system'::text])),
  content text NOT NULL,
  metadata jsonb DEFAULT '{}'::jsonb,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT chat_messages_pkey PRIMARY KEY (id),
  CONSTRAINT chat_messages_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.chat_sessions(id)
);
CREATE TABLE public.chat_sessions (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL,
  title text DEFAULT 'New Chat'::text,
  model text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  deleted_at timestamp with time zone,
  CONSTRAINT chat_sessions_pkey PRIMARY KEY (id),
  CONSTRAINT chat_sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.profiles(id)
);
CREATE TABLE public.document_permissions (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  document_id uuid NOT NULL,
  user_id uuid,
  group_id uuid,
  can_view boolean DEFAULT true,
  can_edit boolean DEFAULT false,
  can_delete boolean DEFAULT false,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT document_permissions_pkey PRIMARY KEY (id),
  CONSTRAINT document_permissions_document_id_fkey FOREIGN KEY (document_id) REFERENCES public.documents(id),
  CONSTRAINT document_permissions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.profiles(id),
  CONSTRAINT document_permissions_group_id_fkey FOREIGN KEY (group_id) REFERENCES public.groups(id)
);
CREATE TABLE public.documents (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  branch_id uuid NOT NULL,
  folder_id uuid,
  owner_id uuid,
  title text NOT NULL,
  storage_path text NOT NULL,
  file_size bigint,
  mime_type text,
  metadata jsonb DEFAULT '{}'::jsonb,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  deleted_at timestamp with time zone,
  description text,
  status text DEFAULT 'pending_review'::text CHECK (status = ANY (ARRAY['pending_review'::text, 'approved'::text, 'rejected'::text])),
  CONSTRAINT documents_pkey PRIMARY KEY (id),
  CONSTRAINT documents_branch_id_fkey FOREIGN KEY (branch_id) REFERENCES public.branches(id),
  CONSTRAINT documents_folder_id_fkey FOREIGN KEY (folder_id) REFERENCES public.folders(id),
  CONSTRAINT documents_owner_id_fkey FOREIGN KEY (owner_id) REFERENCES public.profiles(id)
);
CREATE TABLE public.folder_permissions (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  folder_id uuid NOT NULL,
  user_id uuid,
  group_id uuid,
  can_view boolean DEFAULT true,
  can_upload boolean DEFAULT false,
  can_edit boolean DEFAULT false,
  can_delete boolean DEFAULT false,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT folder_permissions_pkey PRIMARY KEY (id),
  CONSTRAINT folder_permissions_folder_id_fkey FOREIGN KEY (folder_id) REFERENCES public.folders(id),
  CONSTRAINT folder_permissions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.profiles(id),
  CONSTRAINT folder_permissions_group_id_fkey FOREIGN KEY (group_id) REFERENCES public.groups(id)
);
CREATE TABLE public.folders (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  branch_id uuid NOT NULL,
  parent_id uuid,
  name text NOT NULL,
  created_by uuid,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  deleted_at timestamp with time zone,
  description text,
  CONSTRAINT folders_pkey PRIMARY KEY (id),
  CONSTRAINT folders_branch_id_fkey FOREIGN KEY (branch_id) REFERENCES public.branches(id),
  CONSTRAINT folders_parent_id_fkey FOREIGN KEY (parent_id) REFERENCES public.folders(id),
  CONSTRAINT folders_created_by_fkey FOREIGN KEY (created_by) REFERENCES public.profiles(id)
);
CREATE TABLE public.group_members (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  group_id uuid NOT NULL,
  user_id uuid NOT NULL,
  added_at timestamp with time zone DEFAULT now(),
  CONSTRAINT group_members_pkey PRIMARY KEY (id),
  CONSTRAINT group_members_group_id_fkey FOREIGN KEY (group_id) REFERENCES public.groups(id),
  CONSTRAINT group_members_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.profiles(id)
);
CREATE TABLE public.groups (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  organization_id uuid NOT NULL,
  branch_id uuid,
  name text NOT NULL,
  description text,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT groups_pkey PRIMARY KEY (id),
  CONSTRAINT groups_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id),
  CONSTRAINT groups_branch_id_fkey FOREIGN KEY (branch_id) REFERENCES public.branches(id)
);
CREATE TABLE public.organization_members (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  organization_id uuid NOT NULL,
  user_id uuid NOT NULL,
  role_id uuid,
  joined_at timestamp with time zone DEFAULT now(),
  CONSTRAINT organization_members_pkey PRIMARY KEY (id),
  CONSTRAINT organization_members_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id),
  CONSTRAINT organization_members_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.profiles(id),
  CONSTRAINT organization_members_role_id_fkey FOREIGN KEY (role_id) REFERENCES public.roles(id)
);
CREATE TABLE public.organizations (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  name text NOT NULL,
  slug text NOT NULL UNIQUE,
  domain text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT organizations_pkey PRIMARY KEY (id)
);
CREATE TABLE public.profiles (
  id uuid NOT NULL,
  email text NOT NULL UNIQUE,
  full_name text,
  avatar_url text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT profiles_pkey PRIMARY KEY (id),
  CONSTRAINT profiles_id_fkey FOREIGN KEY (id) REFERENCES auth.users(id)
);
CREATE TABLE public.roles (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  organization_id uuid NOT NULL,
  name text NOT NULL,
  permissions jsonb DEFAULT '{}'::jsonb,
  is_system_role boolean DEFAULT false,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT roles_pkey PRIMARY KEY (id),
  CONSTRAINT roles_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id)
);
CREATE TABLE public.team_experience (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  profile_id uuid NOT NULL,
  company text NOT NULL,
  role text NOT NULL,
  years_duration numeric,
  description text,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT team_experience_pkey PRIMARY KEY (id),
  CONSTRAINT team_experience_profile_id_fkey FOREIGN KEY (profile_id) REFERENCES public.team_profiles(id)
);
CREATE TABLE public.team_profiles (
  id uuid NOT NULL,
  organization_id uuid NOT NULL,
  employee_code text,
  job_title text,
  department text,
  reporting_manager_id uuid,
  weekly_hours numeric DEFAULT 40.00,
  overtime_allowed boolean DEFAULT false,
  daily_timings_from time without time zone,
  daily_timings_to time without time zone,
  timezone text,
  employment_start_date date,
  employment_end_date date,
  billing_rate numeric,
  billing_currency text DEFAULT 'USD'::text,
  profile_summary text,
  is_active boolean DEFAULT true,
  status text DEFAULT 'Active'::text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT team_profiles_pkey PRIMARY KEY (id),
  CONSTRAINT team_profiles_id_fkey FOREIGN KEY (id) REFERENCES public.profiles(id),
  CONSTRAINT team_profiles_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id),
  CONSTRAINT team_profiles_reporting_manager_id_fkey FOREIGN KEY (reporting_manager_id) REFERENCES public.profiles(id)
);
CREATE TABLE public.team_qualifications (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  profile_id uuid NOT NULL,
  degree text NOT NULL,
  institution text,
  year_completed integer,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT team_qualifications_pkey PRIMARY KEY (id),
  CONSTRAINT team_qualifications_profile_id_fkey FOREIGN KEY (profile_id) REFERENCES public.team_profiles(id)
);
CREATE TABLE public.team_skills (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  profile_id uuid NOT NULL,
  skill_name text NOT NULL,
  proficiency text,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT team_skills_pkey PRIMARY KEY (id),
  CONSTRAINT team_skills_profile_id_fkey FOREIGN KEY (profile_id) REFERENCES public.team_profiles(id)
);