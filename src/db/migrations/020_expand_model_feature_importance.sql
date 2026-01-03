-- Migration 020: Expand model_feature_importance with raw and normalized columns
-- This migration adds raw and normalized importance columns for better analysis.
-- All statements are idempotent and semicolon-safe.

-- Add raw and normalized importance columns
ALTER TABLE model_feature_importance
    ADD COLUMN IF NOT EXISTS importance_raw NUMERIC,
    ADD COLUMN IF NOT EXISTS importance_norm NUMERIC;

-- Note: importance column is kept for backwards compatibility.
-- New code should populate importance_raw (gain) and importance_norm (normalized gain).

