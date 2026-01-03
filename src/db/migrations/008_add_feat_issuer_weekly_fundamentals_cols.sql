-- Migration 008: Add fundamentals columns to feat_issuer_weekly
-- This migration is idempotent and safe to run multiple times.

-- Add fundamentals lag and derived metrics columns if they don't exist
ALTER TABLE feat_issuer_weekly
    ADD COLUMN IF NOT EXISTS fundamentals_lag_quarters INT,
    ADD COLUMN IF NOT EXISTS net_debt NUMERIC,
    ADD COLUMN IF NOT EXISTS net_debt_ebitda NUMERIC,
    ADD COLUMN IF NOT EXISTS int_coverage NUMERIC,
    ADD COLUMN IF NOT EXISTS fcf_margin NUMERIC,
    ADD COLUMN IF NOT EXISTS capex_intensity NUMERIC;

