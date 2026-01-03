-- Migration 017: Add unique index on model_registry.model_name
-- This migration ensures ON CONFLICT (model_name) works in UPSERT statements.
-- All statements are idempotent and semicolon-safe.

-- First, backfill model_name if any old rows only have legacy "name"
UPDATE model_registry
SET model_name = name
WHERE model_name IS NULL AND name IS NOT NULL;

-- Handle duplicates: keep the latest (highest model_id) per model_name, null out others
-- This is a safe fallback that only runs if duplicates exist
DO $$
BEGIN
  IF EXISTS (
    SELECT model_name
    FROM model_registry
    WHERE model_name IS NOT NULL
    GROUP BY model_name
    HAVING COUNT(*) > 1
  ) THEN
    -- Keep the highest model_id per model_name, null out the rest
    WITH ranked AS (
      SELECT model_id, model_name,
             ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY model_id DESC) AS rn
      FROM model_registry
      WHERE model_name IS NOT NULL
    )
    UPDATE model_registry m
    SET model_name = NULL
    FROM ranked r
    WHERE m.model_id = r.model_id AND r.rn > 1;
  END IF;
END $$;

-- Create unique index on model_name (allows safe re-runs of training)
CREATE UNIQUE INDEX IF NOT EXISTS ux_model_registry_model_name
    ON model_registry(model_name)
    WHERE model_name IS NOT NULL;

-- Note: This index enables ON CONFLICT (model_name) in UPSERT statements.
-- The WHERE clause allows NULL values (multiple rows can have NULL model_name).

