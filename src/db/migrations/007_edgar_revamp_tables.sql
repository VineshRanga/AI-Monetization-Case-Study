-- Migration 007: EDGAR-first revamp - new schema for SEC-based credit risk analysis

-- Update dim_issuer to include CIK
ALTER TABLE dim_issuer ADD COLUMN IF NOT EXISTS cik TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_issuer_cik ON dim_issuer(cik) WHERE cik IS NOT NULL;

-- SEC submissions (filing index)
CREATE TABLE IF NOT EXISTS fact_sec_submissions (
    cik TEXT NOT NULL,
    accession_no TEXT NOT NULL,
    form TEXT,
    filing_date DATE,
    report_date DATE,
    primary_doc TEXT,
    filing_url TEXT,
    PRIMARY KEY (cik, accession_no)
);
CREATE INDEX IF NOT EXISTS idx_submissions_cik ON fact_sec_submissions(cik);
CREATE INDEX IF NOT EXISTS idx_submissions_filing_date ON fact_sec_submissions(filing_date);

-- XBRL facts (normalized raw facts)
CREATE TABLE IF NOT EXISTS fact_xbrl_facts (
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    period_end DATE NOT NULL,
    taxonomy TEXT NOT NULL,
    tag TEXT NOT NULL,
    unit TEXT NOT NULL,
    value NUMERIC,
    form TEXT,
    filed_date DATE,
    frame TEXT,
    PRIMARY KEY (issuer_id, period_end, taxonomy, tag, unit)
);
CREATE INDEX IF NOT EXISTS idx_xbrl_issuer_period ON fact_xbrl_facts(issuer_id, period_end);
CREATE INDEX IF NOT EXISTS idx_xbrl_tag ON fact_xbrl_facts(tag);

-- Fundamentals quarterly (curated wide table) - update existing if present
CREATE TABLE IF NOT EXISTS fact_fundamentals_quarterly (
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    period_end DATE NOT NULL,
    fiscal_year INT,
    fiscal_quarter INT,
    revenue NUMERIC,
    op_income NUMERIC,
    net_income NUMERIC,
    total_debt NUMERIC,
    debt_current NUMERIC,
    long_term_debt NUMERIC,
    cash NUMERIC,
    interest_expense NUMERIC,
    capex NUMERIC,
    cfo NUMERIC,
    fcf NUMERIC,
    shares_outstanding NUMERIC,
    source TEXT NOT NULL DEFAULT 'SEC',
    PRIMARY KEY (issuer_id, period_end)
);
CREATE INDEX IF NOT EXISTS idx_fundamentals_period ON fact_fundamentals_quarterly(period_end);

-- Add missing columns if table exists from migration 006
ALTER TABLE fact_fundamentals_quarterly ADD COLUMN IF NOT EXISTS debt_current NUMERIC;
ALTER TABLE fact_fundamentals_quarterly ADD COLUMN IF NOT EXISTS long_term_debt NUMERIC;
ALTER TABLE fact_fundamentals_quarterly ADD COLUMN IF NOT EXISTS shares_outstanding NUMERIC;

-- Equity prices daily (update existing if present)
CREATE TABLE IF NOT EXISTS fact_equity_price_daily (
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    close NUMERIC,
    adj_close NUMERIC,
    volume NUMERIC,
    source TEXT NOT NULL,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_equity_symbol_date ON fact_equity_price_daily(symbol, date);

-- Weekly market features (update existing if present)
CREATE TABLE IF NOT EXISTS fact_weekly_market (
    week_start DATE PRIMARY KEY,
    spx_ret NUMERIC,
    qqq_ret NUMERIC,
    smh_ret NUMERIC,
    srvr_ret NUMERIC,
    vix_level NUMERIC,
    vix_chg NUMERIC,
    dgs2_chg NUMERIC,
    dgs10_chg NUMERIC,
    curve_2s10s_chg NUMERIC,
    ig_oas_chg NUMERIC,
    hy_oas_chg NUMERIC
);

-- NEW TARGET: Credit proxy weekly
CREATE TABLE IF NOT EXISTS fact_credit_proxy_weekly (
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    week_start DATE NOT NULL,
    dd NUMERIC,
    pd_proxy NUMERIC,
    credit_proxy_level NUMERIC,
    dcredit_proxy NUMERIC,
    method TEXT NOT NULL,
    PRIMARY KEY (issuer_id, week_start)
);
CREATE INDEX IF NOT EXISTS idx_credit_proxy_issuer_week ON fact_credit_proxy_weekly(issuer_id, week_start);

-- Issuer weekly features (update existing if present)
CREATE TABLE IF NOT EXISTS feat_issuer_weekly (
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    week_start DATE NOT NULL,
    bucket TEXT,
    eq_ret NUMERIC,
    eq_vol_21d NUMERIC,
    issuer_mean_target NUMERIC,
    fundamentals_lag_quarters INT,
    net_debt NUMERIC,
    net_debt_ebitda NUMERIC,
    int_coverage NUMERIC,
    fcf_margin NUMERIC,
    capex_intensity NUMERIC,
    PRIMARY KEY (issuer_id, week_start)
);
CREATE INDEX IF NOT EXISTS idx_feat_issuer_week ON feat_issuer_weekly(issuer_id, week_start);

-- Regime labels (keep existing)
CREATE TABLE IF NOT EXISTS model_regime_weekly (
    week_start DATE PRIMARY KEY,
    regime TEXT NOT NULL CHECK (regime IN ('RISK_ON', 'RISK_OFF')),
    prob_risk_off NUMERIC
);

-- Model dataset (update target column)
CREATE TABLE IF NOT EXISTS model_dataset_weekly (
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    week_start DATE NOT NULL,
    dcredit_proxy NUMERIC,
    regime TEXT,
    eq_ret NUMERIC,
    eq_vol_21d NUMERIC,
    bucket TEXT,
    net_debt NUMERIC,
    net_debt_ebitda NUMERIC,
    int_coverage NUMERIC,
    fcf_margin NUMERIC,
    capex_intensity NUMERIC,
    spx_ret NUMERIC,
    qqq_ret NUMERIC,
    vix_chg NUMERIC,
    dgs2_chg NUMERIC,
    dgs10_chg NUMERIC,
    ig_oas_chg NUMERIC,
    PRIMARY KEY (issuer_id, week_start)
);

-- Model registry (keep existing)
CREATE TABLE IF NOT EXISTS model_registry (
    model_id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    regime TEXT,
    train_start DATE,
    train_end DATE,
    test_start DATE,
    test_end DATE,
    hyperparameters JSONB,
    metrics JSONB,
    feature_importance JSONB,
    model_path TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Model predictions (update target column)
CREATE TABLE IF NOT EXISTS model_predictions_weekly (
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    week_start DATE NOT NULL,
    model_id BIGINT REFERENCES model_registry(model_id),
    split TEXT NOT NULL CHECK (split IN ('train', 'test')),
    y_true NUMERIC,
    y_pred NUMERIC,
    PRIMARY KEY (issuer_id, week_start, model_id, split)
);

-- SHAP explanations (keep existing)
CREATE TABLE IF NOT EXISTS model_shap_weekly (
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    week_start DATE NOT NULL,
    model_id BIGINT REFERENCES model_registry(model_id),
    base_value NUMERIC,
    top_features JSONB,
    PRIMARY KEY (issuer_id, week_start, model_id)
);

-- Fragility inputs and scores (keep existing, ensure EDGAR-based)
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
    fcf_margin NUMERIC,
    fcf_volatility NUMERIC,
    equity_vol_21d NUMERIC,
    equity_drawdown_12m NUMERIC,
    structure_opacity_flag BOOLEAN,
    PRIMARY KEY (issuer_id, asof_date)
);

CREATE TABLE IF NOT EXISTS fragility_score (
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    asof_date DATE NOT NULL,
    pillar_refi NUMERIC,
    pillar_cash_capex NUMERIC,
    pillar_leverage NUMERIC,
    pillar_cyclicality NUMERIC,
    pillar_structure NUMERIC,
    fragility_score NUMERIC,
    fragility_rank INT,
    PRIMARY KEY (issuer_id, asof_date)
);

-- Scenario definitions and results (keep existing)
CREATE TABLE IF NOT EXISTS scenario_def (
    scenario_id TEXT PRIMARY KEY,
    scenario_name TEXT NOT NULL,
    description TEXT,
    horizon_weeks INT NOT NULL,
    market_shocks JSONB,
    issuer_shocks JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS scenario_weekly_paths (
    scenario_id TEXT REFERENCES scenario_def(scenario_id),
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    week_offset INT NOT NULL,
    week_start DATE NOT NULL,
    regime TEXT,
    dcredit_proxy_pred NUMERIC,
    credit_proxy_level NUMERIC,
    PRIMARY KEY (scenario_id, issuer_id, week_offset)
);

CREATE TABLE IF NOT EXISTS scenario_stress_results (
    scenario_id TEXT REFERENCES scenario_def(scenario_id),
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    initial_fragility NUMERIC,
    final_fragility NUMERIC,
    max_deterioration NUMERIC,
    weeks_to_max_deterioration INT,
    avg_dcredit_proxy NUMERIC,
    PRIMARY KEY (scenario_id, issuer_id)
);

CREATE TABLE IF NOT EXISTS scenario_portfolio_impact (
    scenario_id TEXT REFERENCES scenario_def(scenario_id),
    portfolio_name TEXT NOT NULL,
    initial_value NUMERIC,
    final_value NUMERIC,
    loss_pct NUMERIC,
    PRIMARY KEY (scenario_id, portfolio_name)
);

