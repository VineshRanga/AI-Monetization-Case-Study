-- Migration 005: Add model training and prediction tables

-- Modeling dataset (materialized joined table)
CREATE TABLE IF NOT EXISTS model_dataset_weekly (
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    week_start DATE NOT NULL,
    regime_label TEXT NOT NULL,
    y_dspread_bps NUMERIC NOT NULL,
    eq_ret NUMERIC,
    eq_vol_21d NUMERIC,
    qqq_ret NUMERIC,
    spx_ret NUMERIC,
    vix_chg NUMERIC,
    vix_level NUMERIC,
    dgs2_chg NUMERIC,
    dgs10_chg NUMERIC,
    curve_2s10s_chg NUMERIC,
    ig_oas_chg NUMERIC,
    smh_ret NUMERIC,
    srvr_ret NUMERIC,
    issuer_mean_dspread NUMERIC,
    bucket TEXT,
    PRIMARY KEY (issuer_id, week_start)
);
CREATE INDEX IF NOT EXISTS idx_model_dataset_week ON model_dataset_weekly(week_start);
CREATE INDEX IF NOT EXISTS idx_model_dataset_regime ON model_dataset_weekly(regime_label);

-- Model registry
CREATE TABLE IF NOT EXISTS model_registry (
    model_id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    regime_label TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    train_start DATE,
    train_end DATE,
    features JSONB,
    hyperparams JSONB,
    metrics JSONB
);

-- Predictions
CREATE TABLE IF NOT EXISTS model_predictions_weekly (
    model_id BIGINT REFERENCES model_registry(model_id),
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    week_start DATE NOT NULL,
    y_true NUMERIC,
    y_pred NUMERIC,
    split_tag TEXT,
    PRIMARY KEY (model_id, issuer_id, week_start)
);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON model_predictions_weekly(model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_week ON model_predictions_weekly(week_start);

-- SHAP top factors (store compactly)
CREATE TABLE IF NOT EXISTS model_shap_weekly (
    model_id BIGINT REFERENCES model_registry(model_id),
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    week_start DATE NOT NULL,
    top_factors JSONB,
    base_value NUMERIC,
    PRIMARY KEY (model_id, issuer_id, week_start)
);
CREATE INDEX IF NOT EXISTS idx_shap_model ON model_shap_weekly(model_id);

