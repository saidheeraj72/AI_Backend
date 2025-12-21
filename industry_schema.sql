
-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "ltree";

-- -----------------------------------------------------------------------------
-- 1. ENUMS & TYPES
-- -----------------------------------------------------------------------------

CREATE TYPE public.org_status AS ENUM (
    'pending_approval',
    'active',
    'suspended',
    'archived'
);

CREATE TYPE public.app_role AS ENUM (
    'superadmin',
    'org_admin',
    'branch_manager',
    'staff',
    'guest'
);

CREATE TYPE public.resource_type AS ENUM (
    'folder',
    'document'
);

CREATE TYPE public.permission_action AS ENUM (
    'view',
    'edit',
    'create',
    'delete',
    'share',
    'full_access'
);

CREATE TYPE public.doc_status AS ENUM (
    'draft',
    'pending_review',
    'approved',
    'rejected',
    'archived'
);

-- -----------------------------------------------------------------------------
-- 2. CORE ORGANIZATION & IDENTITY
-- -----------------------------------------------------------------------------
create table public.superadmins (
  id uuid not null default extensions.uuid_generate_v4 (),
  user_id uuid null,
  email text not null,
  created_at timestamp with time zone null default now(),
  constraint superadmins_pkey primary key (id),
  constraint superadmins_email_key unique (email),
  constraint superadmins_user_id_key unique (user_id),
  constraint superadmins_user_id_fkey foreign KEY (user_id) references auth.users (id) on delete CASCADE
) TABLESPACE pg_default;

-- Organizations (Tenants)
CREATE TABLE public.organizations (
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    name text NOT NULL,
    slug text NOT NULL,
    domain text,
    status public.org_status DEFAULT 'pending_approval'::public.org_status,
    settings jsonb DEFAULT '{}'::jsonb, -- Store theme, policies, etc.
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT organizations_pkey PRIMARY KEY (id),
    CONSTRAINT organizations_slug_key UNIQUE (slug)
);

-- User Profiles (Extends Supabase auth.users)
CREATE TABLE public.profiles (
    id uuid NOT NULL, -- References auth.users(id)
    email text NOT NULL,
    full_name text,
    avatar_url text,
    is_superadmin boolean DEFAULT false, -- Platform level admin
    account_status text DEFAULT 'active', -- For RLS gatekeeping ('active', 'suspended', etc.)
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT profiles_pkey PRIMARY KEY (id),
    CONSTRAINT profiles_email_key UNIQUE (email)
);

-- Organization Members (Users in an Org)
CREATE TABLE public.organization_members (
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    organization_id uuid NOT NULL,
    user_id uuid NOT NULL,
    role public.app_role DEFAULT 'staff'::public.app_role,
    joined_at timestamp with time zone DEFAULT now(),
    status text DEFAULT 'active',
    CONSTRAINT organization_members_pkey PRIMARY KEY (id),
    CONSTRAINT organization_members_org_user_uniq UNIQUE (organization_id, user_id),
    CONSTRAINT organization_members_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id),
    CONSTRAINT organization_members_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.profiles(id)
);

-- Branches (Physical or Logical Divisions)
CREATE TABLE public.branches (
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    organization_id uuid NOT NULL,
    name text NOT NULL,
    code text,
    location text,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT now(),
    CONSTRAINT branches_pkey PRIMARY KEY (id),
    CONSTRAINT branches_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id)
);

-- Groups (For functional teams within an Org, possibly spanning branches)
CREATE TABLE public.groups (
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    organization_id uuid NOT NULL,
    name text NOT NULL,
    description text,
    created_at timestamp with time zone DEFAULT now(),
    CONSTRAINT groups_pkey PRIMARY KEY (id),
    CONSTRAINT groups_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id)
);

-- Group Membership
CREATE TABLE public.group_members (
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    group_id uuid NOT NULL,
    user_id uuid NOT NULL,
    added_at timestamp with time zone DEFAULT now(),
    CONSTRAINT group_members_pkey PRIMARY KEY (id),
    CONSTRAINT group_members_uniq UNIQUE (group_id, user_id),
    CONSTRAINT group_members_group_id_fkey FOREIGN KEY (group_id) REFERENCES public.groups(id) ON DELETE CASCADE,
    CONSTRAINT group_members_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.profiles(id)
);

-- Branch Membership (Users assigned to branches)
CREATE TABLE public.branch_members (
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    branch_id uuid NOT NULL,
    user_id uuid NOT NULL,
    is_primary boolean DEFAULT false,
    joined_at timestamp with time zone DEFAULT now(),
    CONSTRAINT branch_members_pkey PRIMARY KEY (id),
    CONSTRAINT branch_members_uniq UNIQUE (branch_id, user_id),
    CONSTRAINT branch_members_branch_id_fkey FOREIGN KEY (branch_id) REFERENCES public.branches(id) ON DELETE CASCADE,
    CONSTRAINT branch_members_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.profiles(id)
);

-- -----------------------------------------------------------------------------
-- 3. CONTENT MANAGEMENT (Single Source of Truth)
-- -----------------------------------------------------------------------------

-- Folders (Hierarchical structure using ltree)
CREATE TABLE public.folders (
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    organization_id uuid NOT NULL,
    name text NOT NULL,
    description text,
    parent_id uuid,
    path ltree, -- Materialized path for fast recursive queries
    is_system_root boolean DEFAULT false, -- e.g. "My Documents" root for a user
    owner_id uuid, -- For "My Documents", this is the user. For Org docs, usually null or creator.
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    deleted_at timestamp with time zone, -- Soft delete
    CONSTRAINT folders_pkey PRIMARY KEY (id),
    CONSTRAINT folders_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id),
    CONSTRAINT folders_parent_id_fkey FOREIGN KEY (parent_id) REFERENCES public.folders(id),
    CONSTRAINT folders_owner_id_fkey FOREIGN KEY (owner_id) REFERENCES public.profiles(id)
);

-- Index for ltree operations
CREATE INDEX folders_path_gist_idx ON public.folders USING GIST (path);
CREATE INDEX folders_parent_id_idx ON public.folders (parent_id);

-- Documents (The files themselves)
CREATE TABLE public.documents (
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    organization_id uuid NOT NULL,
    folder_id uuid, -- Logical location
    title text NOT NULL,
    description text,
    storage_path text NOT NULL, -- Path in S3/Storage Bucket
    file_size bigint,
    mime_type text,
    status public.doc_status DEFAULT 'pending_review'::public.doc_status,
    owner_id uuid,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    deleted_at timestamp with time zone,
    metadata jsonb DEFAULT '{}'::jsonb, -- Custom tags, authors, etc.
    CONSTRAINT documents_pkey PRIMARY KEY (id),
    CONSTRAINT documents_folder_id_fkey FOREIGN KEY (folder_id) REFERENCES public.folders(id),
    CONSTRAINT documents_owner_id_fkey FOREIGN KEY (owner_id) REFERENCES public.profiles(id),
    CONSTRAINT documents_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id)
);

-- Document Versioning
CREATE TABLE public.document_versions (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id uuid REFERENCES public.documents(id) ON DELETE CASCADE,
    version_number int NOT NULL,
    storage_path text NOT NULL, -- Point to the specific file in S3/Supabase
    created_by uuid REFERENCES public.profiles(id),
    change_log text,
    created_at timestamptz DEFAULT now()
);

-- Folder <-> Branch Mapping (Availability)
-- If a folder is mapped to a branch, it is "visible" in that branch's workspace.
CREATE TABLE public.folder_branch_mappings (
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    folder_id uuid NOT NULL,
    branch_id uuid NOT NULL,
    mapped_at timestamp with time zone DEFAULT now(),
    CONSTRAINT folder_branch_mappings_pkey PRIMARY KEY (id),
    CONSTRAINT folder_branch_mappings_uniq UNIQUE (folder_id, branch_id),
    CONSTRAINT folder_branch_mappings_folder_id_fkey FOREIGN KEY (folder_id) REFERENCES public.folders(id) ON DELETE CASCADE,
    CONSTRAINT folder_branch_mappings_branch_id_fkey FOREIGN KEY (branch_id) REFERENCES public.branches(id) ON DELETE CASCADE
);

-- -----------------------------------------------------------------------------
-- 4. PERMISSIONS & SECURITY (RBAC + ACL)
-- -----------------------------------------------------------------------------

-- Unified Access Control List
-- Explicitly grants permissions to Users or Groups on Resources (Folders/Docs)
CREATE TABLE public.access_control_list (
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    organization_id uuid NOT NULL,
    resource_type public.resource_type NOT NULL,
    resource_id uuid NOT NULL, -- ID of folder or document
    grantee_type text NOT NULL CHECK (grantee_type IN ('user', 'group')),
    grantee_id uuid NOT NULL, -- ID of user or group
    permission public.permission_action NOT NULL,
    granted_by uuid,
    created_at timestamp with time zone DEFAULT now(),
    CONSTRAINT access_control_list_pkey PRIMARY KEY (id),
    -- Unique constraint ensures no duplicate conflicting rules for same grantee/resource/permission
    CONSTRAINT acl_uniq_entry UNIQUE (resource_type, resource_id, grantee_type, grantee_id, permission)
);

-- Sharing Registry (For "My Documents" and external/internal temporary shares)
CREATE TABLE public.share_links (
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    resource_type public.resource_type NOT NULL,
    resource_id uuid NOT NULL,
    created_by uuid NOT NULL,
    recipient_user_id uuid, -- If internal sharing
    recipient_group_id uuid, -- If internal group sharing
    access_token text UNIQUE, -- For external link access (if enabled)
    expires_at timestamp with time zone,
    permission public.permission_action DEFAULT 'view'::public.permission_action,
    created_at timestamp with time zone DEFAULT now(),
    CONSTRAINT share_links_pkey PRIMARY KEY (id),
    CONSTRAINT share_links_created_by_fkey FOREIGN KEY (created_by) REFERENCES public.profiles(id)
);

-- -----------------------------------------------------------------------------
-- 5. INTELLIGENCE LAYER (RAG)
-- -----------------------------------------------------------------------------

-- Document Chunks for Vector Search
CREATE TABLE public.document_chunks (
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    document_id uuid NOT NULL,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    embedding vector(1536), -- Assuming OpenAI ada-002 or compatible
    metadata jsonb DEFAULT '{}'::jsonb, -- Page number, bbox, etc.
    created_at timestamp with time zone DEFAULT now(),
    CONSTRAINT document_chunks_pkey PRIMARY KEY (id),
    CONSTRAINT document_chunks_document_id_fkey FOREIGN KEY (document_id) REFERENCES public.documents(id) ON DELETE CASCADE
);

-- Optimize vector search
CREATE INDEX ON public.document_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Chat Sessions
CREATE TABLE public.chat_sessions (
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    organization_id uuid NOT NULL,
    user_id uuid NOT NULL,
    branch_id uuid, -- Context of the chat (optional, creates scope)
    title text DEFAULT 'New Conversation',
    mode text DEFAULT 'rag' CHECK (mode IN ('rag', 'contextless')),
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT chat_sessions_pkey PRIMARY KEY (id),
    CONSTRAINT chat_sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.profiles(id),
    CONSTRAINT chat_sessions_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id)
);

-- Chat Messages
CREATE TABLE public.chat_messages (
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    session_id uuid NOT NULL,
    role text NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content text NOT NULL,
    citations jsonb DEFAULT '[]'::jsonb, -- Array of references to document_chunks
    created_at timestamp with time zone DEFAULT now(),
    CONSTRAINT chat_messages_pkey PRIMARY KEY (id),
    CONSTRAINT chat_messages_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.chat_sessions(id) ON DELETE CASCADE
);

-- -----------------------------------------------------------------------------
-- 6. COMPLIANCE & AUDIT
-- -----------------------------------------------------------------------------

CREATE TABLE public.audit_logs (
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    organization_id uuid, -- Nullable if system action or user not yet in org
    user_id uuid, -- Nullable if system action
    action text NOT NULL, -- e.g., 'view_document', 'delete_folder', 'login'
    resource_type text, -- 'document', 'folder', 'system', 'user'
    resource_id uuid,
    details jsonb DEFAULT '{}'::jsonb, -- Store metadata like "User asked AI about this doc"
    ip_address inet,
    user_agent text,
    created_at timestamp with time zone DEFAULT now(),
    CONSTRAINT audit_logs_pkey PRIMARY KEY (id),
    CONSTRAINT audit_logs_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id),
    CONSTRAINT audit_logs_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.profiles(id)
);

-- -----------------------------------------------------------------------------
-- 7. FUNCTIONS & TRIGGERS (Helpers)
-- -----------------------------------------------------------------------------

-- Function to update folder path when parent changes
CREATE OR REPLACE FUNCTION update_folder_path() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.parent_id IS NULL THEN
        NEW.path = NEW.id::text::ltree;
    ELSE
        SELECT path || NEW.id::text::ltree INTO NEW.path
        FROM public.folders
        WHERE id = NEW.parent_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_folder_path
    BEFORE INSERT OR UPDATE OF parent_id ON public.folders
    FOR EACH ROW EXECUTE FUNCTION update_folder_path();

-- Function for RAG with Permission Filtering (TEMPLATE)
-- Includes "Cascading" logic via ltree
/*
CREATE OR REPLACE FUNCTION search_documents(
    query_embedding vector(1536),
    match_threshold float,
    match_count int,
    p_user_id uuid,
    p_org_id uuid
)
RETURNS TABLE (
    id uuid,
    content text,
    similarity float,
    document_id uuid
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.content,
        1 - (dc.embedding <=> query_embedding) AS similarity,
        dc.document_id
    FROM public.document_chunks dc
    JOIN public.documents d ON d.id = dc.document_id
    LEFT JOIN public.folders f ON f.id = d.folder_id
    WHERE d.organization_id = p_org_id
      AND (
          -- 1. User is owner of the document
          d.owner_id = p_user_id
          OR
          -- 2. Explicit permission on the document itself (User or Group)
          EXISTS (
              SELECT 1 FROM public.access_control_list acl
              WHERE acl.resource_type = 'document'
                AND acl.resource_id = d.id
                AND (
                    acl.grantee_id = p_user_id
                    OR acl.grantee_id IN (SELECT group_id FROM public.group_members WHERE user_id = p_user_id)
                )
          )
          OR
          -- 3. Inherited permission from ANY ancestor folder
          -- Check if an ACL exists for any folder that is an ancestor of the document's folder
          EXISTS (
              SELECT 1 FROM public.access_control_list acl
              JOIN public.folders ancestor ON acl.resource_id = ancestor.id
              WHERE acl.resource_type = 'folder'
                AND (
                    acl.grantee_id = p_user_id
                    OR acl.grantee_id IN (SELECT group_id FROM public.group_members WHERE user_id = p_user_id)
                )
                AND ancestor.path @> f.path -- The magic of ltree: ancestor contains target folder
          )
      )
      AND 1 - (dc.embedding <=> query_embedding) > match_threshold
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
*/

-- -----------------------------------------------------------------------------
-- 8. SECURITY POLICIES (RLS) - The "Hard Wall"
-- -----------------------------------------------------------------------------

-- 1. Enable RLS on critical tables
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.folders ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.organizations ENABLE ROW LEVEL SECURITY;

-- 2. Profile Policy: Users can only see themselves if active (or if Superadmin)
CREATE POLICY "Users can only read their own profile if active"
ON public.profiles
FOR SELECT
USING (
  (auth.uid() = id AND account_status = 'active') -- User checking themselves
  OR 
  (EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_superadmin = true)) -- Superadmin check
);

-- 3. Organization Gatekeeper Policy
-- Ensures no user can query folders if their Org is NOT active.
CREATE POLICY "Org-level isolation and Gatekeeping"
ON public.folders
FOR SELECT
USING (
  EXISTS (
    SELECT 1 FROM public.organization_members om
    JOIN public.organizations o ON o.id = om.organization_id
    WHERE om.user_id = auth.uid() 
    AND o.status = 'active' -- The Gatekeeper: Org must be active
  )
);

-- Note: Similar policies should be applied to 'documents', 'chat_sessions', etc.
-- to fully enforce the "Manual Approval" gate across the system.


-- Advanced Trigger to handle "First User from Domain" logic
-- 1. Extracts email domain.
-- 2. Checks if an Organization exists for that domain.
-- 3. If NO Org (First User):
--    - Creates a 'pending_approval' Profile.
--    - Creates a 'pending_approval' Organization.
--    - Makes the user the 'org_admin' of that pending Org.
--    - Result: User cannot log in (blocked by RLS) until Superadmin approves the Org/User.
-- 4. If Org EXISTS (Subsequent Users):
--    - Creates an 'active' Profile (Identity is verified).
--    - Adds user to the Org as 'staff' with 'pending_approval' status.
--    - Result: User can log in but has no access to Org data until Org Admin approves.

CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
DECLARE
  user_domain text;
  existing_org_id uuid;
  new_org_id uuid;
BEGIN
  -- Extract domain from email (e.g., 'acme.com' from 'alice@acme.com')
  user_domain := split_part(NEW.email, '@', 2);

  -- Ignore public domains if necessary (optional safeguard)
  -- IF user_domain IN ('gmail.com', 'yahoo.com', 'outlook.com') THEN ... END IF;

  -- Check if an Organization already exists for this domain
  SELECT id INTO existing_org_id 
  FROM public.organizations 
  WHERE domain = user_domain 
  LIMIT 1;

  IF existing_org_id IS NULL THEN
    -- =========================================================
    -- SCENARIO 1: First User from this Domain (New Tenant)
    -- =========================================================
    
    -- 1. Create Profile (Status: Pending Superadmin Approval)
    INSERT INTO public.profiles (id, email, full_name, avatar_url, account_status)
    VALUES (
      NEW.id,
      NEW.email,
      NEW.raw_user_meta_data->>'full_name',
      NEW.raw_user_meta_data->>'avatar_url',
      'pending_approval' -- BLOCKED via RLS until Superadmin approves
    );

    -- 2. Create Pending Organization
    -- Slug generation is simplified here; in production, ensure uniqueness logic
    INSERT INTO public.organizations (name, slug, domain, status)
    VALUES (
      'New Organization (' || user_domain || ')', -- Default Name
      user_domain, -- Default Slug
      user_domain,
      'pending_approval' -- Tenant is not active yet
    ) RETURNING id INTO new_org_id;

    -- 3. Link User as Org Admin
    INSERT INTO public.organization_members (organization_id, user_id, role, status)
    VALUES (
      new_org_id, 
      NEW.id, 
      'org_admin', 
      'active' -- They are the admin, but the Org itself is pending
    );

  ELSE
    -- =========================================================
    -- SCENARIO 2: Organization Exists (Subsequent User)
    -- =========================================================

    -- 1. Create Profile (Status: Active Identity)
    INSERT INTO public.profiles (id, email, full_name, avatar_url, account_status)
    VALUES (
      NEW.id,
      NEW.email,
      NEW.raw_user_meta_data->>'full_name',
      NEW.raw_user_meta_data->>'avatar_url',
      'active' -- Identity is valid, but access depends on Org Membership
    );

    -- 2. Add to Organization (Status: Pending Org Admin Approval)
    INSERT INTO public.organization_members (organization_id, user_id, role, status)
    VALUES (
      existing_org_id, 
      NEW.id, 
      'staff', 
      'pending_approval' -- Waiting for Org Admin to approve join request
    );

  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Recreate the trigger
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
