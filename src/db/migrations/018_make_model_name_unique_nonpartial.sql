-- Migration 018: Make model_name unique constraint non-partial
-- This migration ensures ON CONFLICT (model_name) works by creating a full unique index.
-- Partial indexes (with WHERE clauses) don't work with ON CONFLICT unless the WHERE is specified.
-- All statements are idempotent and semicolon-safe.

-- A) Backfill model_name for legacy rows
UPDATE model_registry
SET model_name = name
WHERE model_name IS NULL AND name IS NOT NULL;

-- For any remaining NULL model_name rows, assign a unique legacy value
UPDATE model_registry
SET model_name = CONCAT('legacy_model_', model_id)
WHERE model_name IS NULL;

-- B) Ensure model_name is NOT NULL (so a full unique constraint makes sense)
ALTER TABLE model_registry
    ALTER COLUMN model_name SET NOT NULL;

-- C) Drop the partial unique index if it exists (handle both common names)
DROP INDEX IF EXISTS ux_model_registry_model_name;

-- D) Create a FULL unique index/constraint on model_name (no WHERE clause)
CREATE UNIQUE INDEX IF NOT EXISTS ux_model_registry_model_name_full
    ON model_registry(model_name);

-- Note: This full unique index enables ON CONFLICT (model_name) in UPSERT statements.
-- All rows must have a non-NULL model_name for this to work.

