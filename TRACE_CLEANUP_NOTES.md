# TRACE Cleanup Notes

This project has been **pivoted from FINRA TRACE to SEC EDGAR-first** approach.

## Obsolete Files (Not Used in New Pipeline)

The following files are **not imported or called** by the new EDGAR pipeline:

### FINRA/TRACE Modules
- `src/finra/*` - All FINRA authentication and discovery modules
- `src/ingest/build_cusip_universe.py` - CUSIP universe builder (TRACE-dependent)
- `src/ingest/ingest_trace_timeseries.py` - TRACE time series ingestion
- `src/transform/build_spreads.py` - Bond spread computation
- `src/transform/select_bonds.py` - Bond selection logic

### Obsolete Scripts
- `scripts/finra_discover.py` - FINRA dataset discovery
- `scripts/build_cusips.py` - CUSIP universe building
- `scripts/ingest_trace.py` - TRACE ingestion
- `scripts/build_spreads.py` - Spread computation
- `scripts/build_weekly_target.py` - Old weekly target builder
- `scripts/pipeline_stage1_targets.py` - Old TRACE pipeline

### Old Migrations (Still Applied, But Tables Not Used)
- `src/db/migrations/002_add_finra_issuer_key.sql` - Adds `finra_issuer_key` (unused)
- `src/db/migrations/003_add_trace_and_spread_tables.sql` - Bond/spread tables (unused)

## New Pipeline

Use the new EDGAR pipeline scripts:
- `scripts/pipeline_edgar_stage1_ingest.py`
- `scripts/pipeline_edgar_stage2_features_targets.py`
- `scripts/pipeline_edgar_stage3_model_train.py`
- `scripts/pipeline_edgar_stage4_stress.py` (coming soon)

## Environment Variables

**Removed:**
- `FINRA_CLIENT_ID`
- `FINRA_CLIENT_SECRET`
- `TRACE_DATASET_NAME`

**Added:**
- `SEC_USER_AGENT` (required)

