-- AICreditRiskAnalysis Database Schema
-- Created for Prompt 1

-- Dimension: Issuers
CREATE TABLE IF NOT EXISTS dim_issuer (
    issuer_id BIGSERIAL PRIMARY KEY,
    ticker TEXT UNIQUE NOT NULL,
    issuer_name TEXT NOT NULL,
    bucket TEXT NOT NULL CHECK (bucket IN ('HYPERSCALER', 'SEMIS', 'DATACENTER')),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Dimension: Bonds
CREATE TABLE IF NOT EXISTS dim_bond (
    cusip TEXT PRIMARY KEY,
    issuer_id BIGINT REFERENCES dim_issuer(issuer_id),
    issuer_name TEXT,
    coupon NUMERIC NULL,
    maturity_date DATE NULL,
    issue_date DATE NULL,
    security_type TEXT NULL,
    first_seen_trade_date DATE NULL,
    last_seen_trade_date DATE NULL,
    is_active BOOLEAN NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- ETL Run Tracking
CREATE TABLE IF NOT EXISTS etl_run (
    run_id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ NULL,
    pipeline_stage TEXT,
    status TEXT CHECK (status IN ('RUNNING', 'OK', 'FAILED')),
    meta JSONB
);

-- Raw HTTP Payload Storage
CREATE TABLE IF NOT EXISTS raw_http_payload (
    payload_id BIGSERIAL PRIMARY KEY,
    run_id BIGINT REFERENCES etl_run(run_id),
    source TEXT,
    endpoint TEXT,
    request_params JSONB,
    retrieved_at TIMESTAMPTZ DEFAULT now(),
    http_status INT,
    payload JSONB
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_dim_bond_issuer_id ON dim_bond(issuer_id);
CREATE INDEX IF NOT EXISTS idx_dim_bond_last_seen_trade_date ON dim_bond(last_seen_trade_date);
CREATE INDEX IF NOT EXISTS idx_raw_http_payload_source_retrieved ON raw_http_payload(source, retrieved_at);
CREATE INDEX IF NOT EXISTS idx_raw_http_payload_payload_gin ON raw_http_payload USING GIN(payload);

