-- Add Billing/Subscription columns to Organizations
ALTER TABLE public.organizations 
ADD COLUMN subscription_plan text DEFAULT 'free', -- 'free', 'pro', 'enterprise'
ADD COLUMN subscription_status text DEFAULT 'active', -- 'active', 'past_due', 'canceled'
ADD COLUMN billing_cycle text DEFAULT 'monthly', -- 'monthly', 'yearly'
ADD COLUMN next_billing_date timestamp with time zone,
ADD COLUMN modules jsonb DEFAULT '{}'::jsonb; -- e.g. {"rag_chat": "active", "pmo_planner": "requested"}
