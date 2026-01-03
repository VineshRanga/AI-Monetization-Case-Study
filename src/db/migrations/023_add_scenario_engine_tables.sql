-- Migration 023: Add scenario engine tables for stress testing
-- All statements are idempotent and semicolon-safe.

-- Scenario definitions
CREATE TABLE IF NOT EXISTS scenario_definition (
    scenario_id BIGSERIAL PRIMARY KEY,
    scenario_name TEXT UNIQUE NOT NULL,
    scenario_desc TEXT NOT NULL,
    asof_date DATE NOT NULL,
    parameters_json JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scenario_definition_name ON scenario_definition(scenario_name);
CREATE INDEX IF NOT EXISTS idx_scenario_definition_date ON scenario_definition(asof_date);

-- Scenario shock paths (weekly time series of shock multipliers)
CREATE TABLE IF NOT EXISTS scenario_shock_path_weekly (
    scenario_id BIGINT NOT NULL REFERENCES scenario_definition(scenario_id) ON DELETE CASCADE,
    week_start DATE NOT NULL,
    shock_regime_mult NUMERIC NOT NULL,
    shock_base NUMERIC NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (scenario_id, week_start)
);

CREATE INDEX IF NOT EXISTS idx_scenario_shock_path_week ON scenario_shock_path_weekly(week_start);

-- Scenario results (issuer-level weekly predictions)
CREATE TABLE IF NOT EXISTS scenario_results_issuer_weekly (
    scenario_id BIGINT NOT NULL REFERENCES scenario_definition(scenario_id) ON DELETE CASCADE,
    issuer_id BIGINT NOT NULL REFERENCES dim_issuer(issuer_id),
    week_start DATE NOT NULL,
    baseline_pred_dcredit_proxy NUMERIC NOT NULL,
    scenario_pred_dcredit_proxy NUMERIC NOT NULL,
    uplift NUMERIC NOT NULL,
    drivers_json JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (scenario_id, issuer_id, week_start)
);

CREATE INDEX IF NOT EXISTS idx_scenario_results_scenario ON scenario_results_issuer_weekly(scenario_id);
CREATE INDEX IF NOT EXISTS idx_scenario_results_issuer ON scenario_results_issuer_weekly(issuer_id);
CREATE INDEX IF NOT EXISTS idx_scenario_results_week ON scenario_results_issuer_weekly(week_start);

-- Spillover groups (exposures to buckets)
CREATE TABLE IF NOT EXISTS scenario_spillover_groups (
    scenario_id BIGINT NOT NULL REFERENCES scenario_definition(scenario_id) ON DELETE CASCADE,
    group_name TEXT NOT NULL,
    spillover_index NUMERIC NOT NULL,
    drivers_json JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (scenario_id, group_name)
);

CREATE INDEX IF NOT EXISTS idx_scenario_spillover_scenario ON scenario_spillover_groups(scenario_id);

-- Note: These tables support the "2008 for AI" scenario engine:
-- - scenario_definition: metadata and parameters for each scenario
-- - scenario_shock_path_weekly: time-varying shock multipliers
-- - scenario_results_issuer_weekly: issuer-level credit deterioration trajectories
-- - scenario_spillover_groups: second-order propagation to banks/asset managers/tech supply chain

