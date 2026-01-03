-- Migration 022: Add raw_json column to fragility_pillar_scores if missing
-- This handles cases where the table was created before migration 021 or with a different schema.

-- Add raw_json column if it doesn't exist
ALTER TABLE fragility_pillar_scores
    ADD COLUMN IF NOT EXISTS raw_json JSONB;

-- If the column was just added and is NULL, we can't set NOT NULL yet (would fail on existing rows)
-- Instead, we'll allow NULL for now and let the application handle it
-- The INSERT statements in build_fragility_score.py will populate it going forward

-- Note: If you need to enforce NOT NULL in the future, first backfill:
-- UPDATE fragility_pillar_scores SET raw_json = '{}'::jsonb WHERE raw_json IS NULL;
-- ALTER TABLE fragility_pillar_scores ALTER COLUMN raw_json SET NOT NULL;

