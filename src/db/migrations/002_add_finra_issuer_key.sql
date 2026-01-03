-- Migration 002: Add finra_issuer_key to dim_issuer
-- This stores the exact issuer filter value used in TRACE queries

ALTER TABLE dim_issuer ADD COLUMN IF NOT EXISTS finra_issuer_key TEXT;

