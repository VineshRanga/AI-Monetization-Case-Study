-- Migration 010: Add dcredit_proxy column to model_dataset_weekly
-- This migration ensures the EDGAR target column exists.
-- All statements are idempotent and semicolon-safe.

-- Add dcredit_proxy column if it doesn't exist
ALTER TABLE model_dataset_weekly
    ADD COLUMN IF NOT EXISTS dcredit_proxy NUMERIC;

-- Note: If old TRACE columns exist (e.g., y_dspread_bps, dspread_bps), they are kept
-- for backward compatibility. No columns are dropped.

-- Ensure primary key constraint exists (should already exist, but safe to check)
-- Primary key is (issuer_id, week_start) - this is already defined in CREATE TABLE
-- No additional indexes needed at this time

