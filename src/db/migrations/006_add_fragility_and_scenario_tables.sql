-- Migration 006: Add fragility scoring and scenario stress testing tables

-- Fundamentals (quarterly)
CREATE TABLE IF NOT EXISTS fact_fundamentals_quarterly (
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    period_end DATE NOT NULL,
    fiscal_year INT,
    fiscal_quarter INT,
    revenue NUMERIC,
    ebitda NUMERIC,
    op_income NUMERIC,
    net_income NUMERIC,
    total_debt NUMERIC,
    cash NUMERIC,
    net_debt NUMERIC,
    interest_expense NUMERIC,
    capex NUMERIC,
    cfo NUMERIC,
    fcf NUMERIC,
    source TEXT NOT NULL DEFAULT 'SEC',
    PRIMARY KEY (issuer_id, period_end)
);
CREATE INDEX IF NOT EXISTS idx_fundamentals_period ON fact_fundamentals_quarterly(period_end);

-- Capital structure / maturity wall
CREATE TABLE IF NOT EXISTS fact_cap_structure (
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    asof_date DATE NOT NULL,
    maturity_0_12m NUMERIC,
    maturity_12_24m NUMERIC,
    maturity_24_60m NUMERIC,
    maturity_60m_plus NUMERIC,
    floating_debt_pct NUMERIC,
    lease_obligations NUMERIC,
    notes TEXT,
    source TEXT NOT NULL DEFAULT 'SEC_PARSE',
    PRIMARY KEY (issuer_id, asof_date)
);

-- Fragility inputs + scores
CREATE TABLE IF NOT EXISTS fragility_inputs (
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    asof_date DATE NOT NULL,

    maturity_24m_pct NUMERIC,
    floating_pct NUMERIC,

    fcf_after_capex NUMERIC,
    capex_intensity NUMERIC,
    capex_commitment_proxy NUMERIC,

    net_debt_ebitda NUMERIC,
    int_coverage NUMERIC,

    cyclicality_proxy NUMERIC,
    ai_concentration NUMERIC,

    structure_opacity_flag INT,
    PRIMARY KEY (issuer_id, asof_date)
);

CREATE TABLE IF NOT EXISTS fragility_score (
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    asof_date DATE NOT NULL,
    total_score NUMERIC,
    p_refi NUMERIC,
    p_cash_capex NUMERIC,
    p_leverage NUMERIC,
    p_cyc_ai NUMERIC,
    p_structure NUMERIC,
    notes TEXT,
    PRIMARY KEY (issuer_id, asof_date)
);

-- Scenario definitions + stress outputs
CREATE TABLE IF NOT EXISTS scenario_def (
    scenario_id BIGSERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    params JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS scenario_stress_results (
    scenario_id BIGINT REFERENCES scenario_def(scenario_id),
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    horizon_weeks INT NOT NULL,
    pred_cum_widen_bps NUMERIC,
    pred_peak_weekly_widen_bps NUMERIC,
    key_drivers JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (scenario_id, issuer_id, horizon_weeks)
);

CREATE TABLE IF NOT EXISTS scenario_weekly_paths (
    scenario_id BIGINT REFERENCES scenario_def(scenario_id),
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    week_start DATE NOT NULL,
    pred_dspread_bps NUMERIC,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (scenario_id, issuer_id, week_start)
);

CREATE TABLE IF NOT EXISTS scenario_portfolio_impact (
    scenario_id BIGINT REFERENCES scenario_def(scenario_id),
    portfolio_name TEXT,
    est_loss_pct NUMERIC,
    notes TEXT,
    PRIMARY KEY (scenario_id, portfolio_name)
);

-- Add CIK to dim_issuer for SEC lookups
ALTER TABLE dim_issuer ADD COLUMN IF NOT EXISTS cik TEXT;

