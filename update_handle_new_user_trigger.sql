-- Updated handle_new_user function to incorporate organization_name from signup metadata.
-- This replaces the previous version of the function.

CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
DECLARE
  user_domain text;
  user_full_name text;
  user_usage_type text;
  organization_name_from_signup text; -- New variable for organization name
  existing_org_id uuid;
  new_org_id uuid;
  is_public_domain boolean;
  generated_slug text;
BEGIN
  -- 1. Extract Details
  user_domain := split_part(NEW.email, '@', 2);
  user_full_name := COALESCE(NEW.raw_user_meta_data->>'full_name', 'User');
  user_usage_type := COALESCE(NEW.raw_user_meta_data->>'usage_type', 'personal');
  organization_name_from_signup := NEW.raw_user_meta_data->>'organization_name'; -- Extract new field
  
  -- Define Public Domains
  is_public_domain := (user_domain IN ('gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'icloud.com'));

  -- =========================================================
  -- PATH A: PERSONAL USE (or Public Domain) -> 1 Month Free Trial
  -- =========================================================
  IF user_usage_type = 'personal' OR is_public_domain THEN
    
    -- 1. Create Active Profile
    INSERT INTO public.profiles (id, email, full_name, avatar_url, account_status)
    VALUES (
      NEW.id,
      NEW.email,
      user_full_name,
      NEW.raw_user_meta_data->>'avatar_url',
      'active'
    );

    generated_slug := lower(regexp_replace(user_full_name, '[^a-zA-Z0-9]', '-', 'g')) || '-' || substring(md5(random()::text) from 1 for 6);

    -- 2. Create Personal Organization with 1 MONTH TRIAL
    INSERT INTO public.organizations (
      name, 
      slug, 
      domain, 
      status, 
      subscription_tier, 
      subscription_expires_at
    )
    VALUES (
      user_full_name || '''s Workspace',
      generated_slug, 
      null,
      'active',
      'free_trial',
      (now() + interval '1 month')
    ) RETURNING id INTO new_org_id;

    -- 3. Link User as Org Admin
    INSERT INTO public.organization_members (organization_id, user_id, role, status)
    VALUES (new_org_id, NEW.id, 'org_admin', 'active');

  -- =========================================================
  -- PATH B: ORGANIZATION / CORPORATE USE
  -- =========================================================
  ELSE
    
    SELECT id INTO existing_org_id FROM public.organizations WHERE domain = user_domain LIMIT 1;

    IF existing_org_id IS NULL THEN
      -- B1. New Corporate Tenant -> 1 Month Free Trial (Starts on creation)
      
      INSERT INTO public.profiles (id, email, full_name, avatar_url, account_status)
      VALUES (
        NEW.id,
        NEW.email,
        user_full_name,
        NEW.raw_user_meta_data->>'avatar_url',
        'pending_approval'
      );

      INSERT INTO public.organizations (
        name, 
        slug, 
        domain, 
        status, 
        subscription_tier, 
        subscription_expires_at
      )
      VALUES (
        COALESCE(organization_name_from_signup, user_domain || ' Organization'), -- Use provided name or default
        lower(regexp_replace(COALESCE(organization_name_from_signup, user_domain), '[^a-zA-Z0-9]', '-', 'g')), -- Slug from provided name
        user_domain,
        'pending_approval',
        'free_trial',
        (now() + interval '1 month')
      ) RETURNING id INTO new_org_id;

      INSERT INTO public.organization_members (organization_id, user_id, role, status)
      VALUES (new_org_id, NEW.id, 'org_admin', 'active');

    ELSE
      -- B2. Subsequent User (Existing Tenant) -> Inherits Organization's Subscription
      
      INSERT INTO public.profiles (id, email, full_name, avatar_url, account_status)
      VALUES (
        NEW.id,
        NEW.email,
        user_full_name,
        NEW.raw_user_meta_data->>'avatar_url',
        'active'
      );

      INSERT INTO public.organization_members (organization_id, user_id, role, status)
      VALUES (existing_org_id, NEW.id, 'staff', 'pending_approval');

    END IF;

  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
