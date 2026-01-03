-- Migration 014: Relax NOT NULL constraint on model_registry.name
-- This migration allows future inserts to work even if name is not provided,
-- while keeping backwards compatibility with old schema.
-- All statements are idempotent and semicolon-safe.

-- Drop NOT NULL constraint on legacy name column
ALTER TABLE model_registry
    ALTER COLUMN name DROP NOT NULL;

-- Note: The name column is kept for backwards compatibility.
-- New code should use model_name, but name is kept to avoid breaking old queries.

