-- Migration 021: Add fragility scoring tables (5-pillar system)
-- All statements are idempotent and semicolon-safe.

-- Pillar scores table
CREATE TABLE IF NOT EXISTS fragility_pillar_scores (
    issuer_id BIGINT NOT NULL REFERENCES dim_issuer(issuer_id),
    asof_date DATE NOT NULL,
    pillar_name TEXT NOT NULL,
    raw_json JSONB NOT NULL,
    score_0_100 NUMERIC NOT NULL,
    method TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (issuer_id, asof_date, pillar_name)
);

CREATE INDEX IF NOT EXISTS idx_fragility_pillar_scores_issuer_date 
    ON fragility_pillar_scores(issuer_id, asof_date);

-- Total fragility scores table
CREATE TABLE IF NOT EXISTS fragility_scores (
    issuer_id BIGINT NOT NULL REFERENCES dim_issuer(issuer_id),
    asof_date DATE NOT NULL,
    total_score_0_100 NUMERIC NOT NULL,
    weights_json JSONB NOT NULL,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (issuer_id, asof_date)
);

CREATE INDEX IF NOT EXISTS idx_fragility_scores_date 
    ON fragility_scores(asof_date);

-- Note: These tables store the 5-pillar fragility scoring system:
-- P1: Refinancing wall / funding risk
-- P2: Cash generation vs committed capex
-- P3: Leverage / coverage
-- P4: Cyclicality & AI concentration
-- P5: Structure / opacity
