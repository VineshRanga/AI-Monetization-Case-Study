-- Migration 004: Add feature store tables for weekly features and regime labels

-- Daily equity price table
CREATE TABLE IF NOT EXISTS fact_equity_price_daily (
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    close NUMERIC,
    adj_close NUMERIC,
    volume NUMERIC,
    source TEXT NOT NULL,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_equity_date ON fact_equity_price_daily(date);
CREATE INDEX IF NOT EXISTS idx_equity_symbol ON fact_equity_price_daily(symbol);

-- Weekly market features
CREATE TABLE IF NOT EXISTS fact_weekly_market (
    week_start DATE PRIMARY KEY,
    spx_ret NUMERIC,
    qqq_ret NUMERIC,
    vix_level NUMERIC,
    vix_chg NUMERIC,
    dgs2_chg NUMERIC,
    dgs10_chg NUMERIC,
    curve_2s10s_chg NUMERIC,
    ig_oas_chg NUMERIC,
    hy_oas_chg NUMERIC,
    smh_ret NUMERIC,
    srvr_ret NUMERIC
);

-- Issuer feature store
CREATE TABLE IF NOT EXISTS feat_issuer_weekly (
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    week_start DATE NOT NULL,
    eq_ret NUMERIC,
    eq_vol_21d NUMERIC,
    spx_beta_52w NUMERIC,
    qqq_beta_52w NUMERIC,
    net_debt_ebitda NUMERIC,
    int_coverage NUMERIC,
    fcf_margin NUMERIC,
    capex_intensity NUMERIC,
    ai_concentration NUMERIC,
    bucket TEXT,
    PRIMARY KEY (issuer_id, week_start)
);
CREATE INDEX IF NOT EXISTS idx_feat_issuer_week ON feat_issuer_weekly(week_start);

-- Regime labels
CREATE TABLE IF NOT EXISTS model_regime_weekly (
    week_start DATE PRIMARY KEY,
    prob_risk_off NUMERIC NOT NULL,
    regime_label TEXT NOT NULL CHECK (regime_label IN ('RISK_ON', 'RISK_OFF')),
    method TEXT NOT NULL
);

