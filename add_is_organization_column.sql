-- 1. Add is_organization column to organizations table
ALTER TABLE public.organizations 
ADD COLUMN is_organization boolean DEFAULT false;

-- 2. Update the handle_new_user function to set is_organization based on metadata
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
DECLARE
  user_domain text;
  existing_org_id uuid;
  new_org_id uuid;
  user_usage_type text;
  is_org boolean;
BEGIN
  -- Extract domain from email
  user_domain := split_part(NEW.email, '@', 2);
  
  -- Extract usage_type from metadata (default to 'personal' if missing)
  user_usage_type := COALESCE(NEW.raw_user_meta_data->>'usage_type', 'personal');
  
  -- Determine is_organization flag
  is_org := (user_usage_type = 'organization');

  -- Check if an Organization already exists for this domain
  -- (Note: For personal accounts, we might want to ignore domain matching or handle differently, 
  -- but keeping existing logic for now, just adding the flag)
  SELECT id INTO existing_org_id 
  FROM public.organizations 
  WHERE domain = user_domain 
  LIMIT 1;

  IF existing_org_id IS NULL THEN
    -- =========================================================
    -- SCENARIO 1: First User from this Domain (New Tenant)
    -- =========================================================
    
    -- 1. Create Profile
    INSERT INTO public.profiles (id, email, full_name, avatar_url, account_status)
    VALUES (
      NEW.id,
      NEW.email,
      NEW.raw_user_meta_data->>'full_name',
      NEW.raw_user_meta_data->>'avatar_url',
      'pending_approval' -- BLOCKED via RLS until Superadmin approves
    );

    -- 2. Create Pending Organization with is_organization flag
    INSERT INTO public.organizations (name, slug, domain, status, is_organization)
    VALUES (
      'New Organization (' || user_domain || ')', -- Default Name
      user_domain, -- Default Slug
      user_domain,
      'pending_approval',
      is_org -- Set based on signup metadata
    ) RETURNING id INTO new_org_id;

    -- 3. Link User as Org Admin
    INSERT INTO public.organization_members (organization_id, user_id, role, status)
    VALUES (
      new_org_id, 
      NEW.id, 
      'org_admin', 
      'active'
    );

  ELSE
    -- =========================================================
    -- SCENARIO 2: Organization Exists (Subsequent User)
    -- =========================================================

    -- 1. Create Profile
    INSERT INTO public.profiles (id, email, full_name, avatar_url, account_status)
    VALUES (
      NEW.id,
      NEW.email,
      NEW.raw_user_meta_data->>'full_name',
      NEW.raw_user_meta_data->>'avatar_url',
      'active'
    );

    -- 2. Add to Organization
    INSERT INTO public.organization_members (organization_id, user_id, role, status)
    VALUES (
      existing_org_id, 
      NEW.id, 
      'staff', 
      'pending_approval'
    );

  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
