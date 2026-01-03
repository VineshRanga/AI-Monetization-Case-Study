-- Migration 009: Fix model_regime_weekly and model_dataset_weekly schema to use regime_label
-- This migration standardizes the regime column naming.
-- All statements are idempotent and semicolon-safe (no DO blocks).

-- Fix model_regime_weekly
-- Add prob_risk_off if missing
ALTER TABLE model_regime_weekly
    ADD COLUMN IF NOT EXISTS prob_risk_off NUMERIC;

-- Add regime_label column if it doesn't exist
ALTER TABLE model_regime_weekly
    ADD COLUMN IF NOT EXISTS regime_label TEXT;

-- Add method column if it doesn't exist (required, NOT NULL)
ALTER TABLE model_regime_weekly
    ADD COLUMN IF NOT EXISTS method TEXT;

-- Drop old constraints if they exist (idempotent)
ALTER TABLE model_regime_weekly
    DROP CONSTRAINT IF EXISTS model_regime_weekly_regime_check;

ALTER TABLE model_regime_weekly
    DROP CONSTRAINT IF EXISTS model_regime_weekly_regime_label_check;

-- Add constraint for regime_label (idempotent: drop first, then add)
ALTER TABLE model_regime_weekly
    ADD CONSTRAINT model_regime_weekly_regime_label_check
    CHECK (regime_label IN ('RISK_ON', 'RISK_OFF'));

-- Fix model_dataset_weekly
-- Add regime_label column if it doesn't exist
ALTER TABLE model_dataset_weekly
    ADD COLUMN IF NOT EXISTS regime_label TEXT;

-- Note: We keep the old 'regime' columns for now to avoid breaking existing data
-- They can be dropped in a future migration if needed
-- Data migration from regime to regime_label should be done in Python with schema introspection
