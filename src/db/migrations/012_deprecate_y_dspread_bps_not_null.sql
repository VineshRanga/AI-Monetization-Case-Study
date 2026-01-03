-- Migration 012: Deprecate y_dspread_bps NOT NULL constraint
-- This migration allows the legacy TRACE target column to be NULL
-- so that EDGAR workflow (using dcredit_proxy) can insert rows successfully.
-- All statements are idempotent and semicolon-safe.

-- Drop NOT NULL constraint on legacy y_dspread_bps column
-- This allows NULL values so EDGAR inserts (using dcredit_proxy) can succeed
ALTER TABLE model_dataset_weekly
    ALTER COLUMN y_dspread_bps DROP NOT NULL;

-- Note: The column is kept for backward compatibility.
-- EDGAR workflow uses dcredit_proxy as the target, not y_dspread_bps.


