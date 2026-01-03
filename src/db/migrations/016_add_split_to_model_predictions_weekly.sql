-- Migration 016: Add split column to model_predictions_weekly
-- This migration adds the split column and matching unique index to support train/test separation.
-- All statements are idempotent and semicolon-safe.

-- Add split column if it doesn't exist
ALTER TABLE model_predictions_weekly
    ADD COLUMN IF NOT EXISTS split TEXT;

-- Backfill existing rows (if any) so we can enforce NOT NULL
UPDATE model_predictions_weekly
SET split = 'train'
WHERE split IS NULL;

-- Set default value
ALTER TABLE model_predictions_weekly
    ALTER COLUMN split SET DEFAULT 'train';

-- Enforce NOT NULL
ALTER TABLE model_predictions_weekly
    ALTER COLUMN split SET NOT NULL;

-- Add CHECK constraint (idempotent using DO block)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'model_predictions_weekly_split_check'
    ) THEN
        ALTER TABLE model_predictions_weekly
            ADD CONSTRAINT model_predictions_weekly_split_check
            CHECK (split IN ('train', 'test'));
    END IF;
END $$;

-- Ensure ON CONFLICT has a matching unique constraint/index
CREATE UNIQUE INDEX IF NOT EXISTS ux_model_predictions_weekly_key
    ON model_predictions_weekly (issuer_id, week_start, model_id, split);

-- Note: This index matches the ON CONFLICT clause in save_predictions.py:
-- ON CONFLICT (issuer_id, week_start, model_id, split)

