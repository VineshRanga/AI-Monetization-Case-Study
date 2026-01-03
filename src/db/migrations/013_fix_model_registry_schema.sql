-- Migration 013: Fix model_registry schema to align with Stage 3 outputs
-- This migration adds missing columns idempotently to support EDGAR-first model training.
-- All statements are idempotent and semicolon-safe.

-- Add missing columns if they don't exist
ALTER TABLE model_registry
    ADD COLUMN IF NOT EXISTS model_name TEXT,
    ADD COLUMN IF NOT EXISTS model_type TEXT,
    ADD COLUMN IF NOT EXISTS regime TEXT,
    ADD COLUMN IF NOT EXISTS target TEXT,
    ADD COLUMN IF NOT EXISTS train_start DATE,
    ADD COLUMN IF NOT EXISTS train_end DATE,
    ADD COLUMN IF NOT EXISTS test_start DATE,
    ADD COLUMN IF NOT EXISTS test_end DATE,
    ADD COLUMN IF NOT EXISTS hyperparameters JSONB,
    ADD COLUMN IF NOT EXISTS metrics JSONB,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();

-- Add unique index on model_name (allows safe re-runs of training)
CREATE UNIQUE INDEX IF NOT EXISTS ux_model_registry_model_name
    ON model_registry(model_name)
    WHERE model_name IS NOT NULL;

-- Note: model_id primary key should already exist from previous migrations.
-- This migration does not modify the primary key.

