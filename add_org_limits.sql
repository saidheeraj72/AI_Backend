-- Add user_limit and storage_limit_gb columns to organizations table
ALTER TABLE public.organizations 
ADD COLUMN user_limit integer DEFAULT 5, -- Default 5 users
ADD COLUMN storage_limit_gb integer DEFAULT 10; -- Default 10 GB

-- Optional: Update existing records to have these defaults if they are null (though DEFAULT handles new inserts)
UPDATE public.organizations 
SET user_limit = 5 
WHERE user_limit IS NULL;

UPDATE public.organizations 
SET storage_limit_gb = 10 
WHERE storage_limit_gb IS NULL;
