-- Allow Superadmins to view ALL organizations (active, pending, etc.)
-- This is required for the "Requests" tab to work.

CREATE POLICY "Superadmins can view all organizations"
ON public.organizations
FOR SELECT
USING (
  public.is_superadmin(auth.uid())
);
