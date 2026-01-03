-- Migration 011: Add fundamentals and macro feature columns to model_dataset_weekly
-- This migration ensures all feature columns exist for the EDGAR-based modeling dataset.
-- All statements are idempotent and semicolon-safe.

-- Add fundamentals feature columns if they don't exist
ALTER TABLE model_dataset_weekly
    ADD COLUMN IF NOT EXISTS net_debt NUMERIC,
    ADD COLUMN IF NOT EXISTS net_debt_ebitda NUMERIC,
    ADD COLUMN IF NOT EXISTS int_coverage NUMERIC,
    ADD COLUMN IF NOT EXISTS fcf_margin NUMERIC,
    ADD COLUMN IF NOT EXISTS capex_intensity NUMERIC;

-- Add macro feature columns if they don't exist
ALTER TABLE model_dataset_weekly
    ADD COLUMN IF NOT EXISTS spx_ret NUMERIC,
    ADD COLUMN IF NOT EXISTS qqq_ret NUMERIC,
    ADD COLUMN IF NOT EXISTS vix_chg NUMERIC,
    ADD COLUMN IF NOT EXISTS dgs2_chg NUMERIC,
    ADD COLUMN IF NOT EXISTS dgs10_chg NUMERIC,
    ADD COLUMN IF NOT EXISTS ig_oas_chg NUMERIC;

-- Note: Other columns (eq_ret, eq_vol_21d, bucket, regime_label) should already exist
-- from previous migrations. If they don't, they will need to be added in a future migration.

