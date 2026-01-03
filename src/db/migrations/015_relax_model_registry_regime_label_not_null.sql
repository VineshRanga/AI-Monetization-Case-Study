-- Migration 015: Relax NOT NULL constraint on model_registry.regime_label
-- This migration allows future inserts to work even if regime_label is not provided,
-- while keeping backwards compatibility with old schema.
-- All statements are idempotent and semicolon-safe.

-- Drop NOT NULL constraint on legacy regime_label column
ALTER TABLE model_registry
    ALTER COLUMN regime_label DROP NOT NULL;

-- Note: The regime_label column is kept for backwards compatibility.
-- New code should use regime, but regime_label is kept to avoid breaking old queries.

