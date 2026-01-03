-- Migration 019: Add model_feature_importance table
-- This migration creates a table to store feature importances from trained models.
-- All statements are idempotent and semicolon-safe.

-- Create model_feature_importance table
CREATE TABLE IF NOT EXISTS model_feature_importance (
    model_id BIGINT REFERENCES model_registry(model_id),
    feature TEXT NOT NULL,
    importance NUMERIC NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (model_id, feature)
);

-- Create index for efficient queries
CREATE INDEX IF NOT EXISTS idx_model_feature_importance_model
    ON model_feature_importance(model_id);

-- Note: This table stores feature importances (gain, weight, etc.) from XGBoost models.
-- Used for generating feature importance charts without requiring SHAP data.

