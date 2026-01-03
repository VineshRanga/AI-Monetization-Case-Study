-- Migration 003: Add fact tables for TRACE time series, UST yields, spreads, and weekly issuer series

-- Bond time series (normalized daily)
CREATE TABLE IF NOT EXISTS fact_bond_daily (
    cusip TEXT REFERENCES dim_bond(cusip),
    trade_date DATE NOT NULL,
    price NUMERIC NULL,
    ytm NUMERIC NULL,          -- yield-to-maturity or yield field available from TRACE dataset
    volume NUMERIC NULL,
    trades INT NULL,
    source TEXT DEFAULT 'FINRA_TRACE',
    PRIMARY KEY (cusip, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_fact_bond_daily_date ON fact_bond_daily(trade_date);

-- UST yield curve (daily, selected tenors)
CREATE TABLE IF NOT EXISTS fact_ust_yield_daily (
    date DATE NOT NULL,
    tenor TEXT NOT NULL,       -- '2Y','5Y','7Y','10Y','20Y','30Y'
    yield NUMERIC NOT NULL,
    source TEXT NOT NULL,      -- 'FRED' or 'CSV'
    PRIMARY KEY (date, tenor)
);
CREATE INDEX IF NOT EXISTS idx_ust_date ON fact_ust_yield_daily(date);

-- Bond spread proxy (daily)
CREATE TABLE IF NOT EXISTS fact_bond_spread_daily (
    cusip TEXT REFERENCES dim_bond(cusip),
    trade_date DATE NOT NULL,
    ust_tenor TEXT NOT NULL,
    ust_yield NUMERIC NOT NULL,
    bond_ytm NUMERIC NOT NULL,
    spread_bps NUMERIC NOT NULL,     -- (bond_ytm - ust_yield) * 10000
    PRIMARY KEY (cusip, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_spread_date ON fact_bond_spread_daily(trade_date);

-- Issuer weekly series (model target + features scaffold)
CREATE TABLE IF NOT EXISTS fact_issuer_spread_weekly (
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    week_start DATE NOT NULL,
    spread_bps NUMERIC NULL,         -- issuer-level spread proxy (selected bond or weighted)
    dspread_bps NUMERIC NULL,        -- weekly change
    bond_count INT NULL,
    method TEXT NOT NULL,            -- e.g. 'TOP1_LIQUID', 'TOP2_VOL_WEIGHTED'
    PRIMARY KEY (issuer_id, week_start)
);

-- Add bond selection columns to dim_bond
ALTER TABLE dim_bond ADD COLUMN IF NOT EXISTS is_selected BOOLEAN DEFAULT FALSE;
ALTER TABLE dim_bond ADD COLUMN IF NOT EXISTS selected_rank INT NULL;
ALTER TABLE dim_bond ADD COLUMN IF NOT EXISTS selected_reason TEXT NULL;

