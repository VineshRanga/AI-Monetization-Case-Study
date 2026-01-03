# Green Run Checklist

This document tracks the execution plan and green run status for the EDGAR-first revamp.

## ✅ Completed: Surgical Cleanup

1. **Deprecated TRACE/FINRA code moved to `deprecated/` folder:**
   - `src/finra/` → `deprecated/src/finra/`
   - `src/ingest/build_cusip_universe.py` → `deprecated/src/ingest/`
   - `src/ingest/ingest_trace_timeseries.py` → `deprecated/src/ingest/`
   - `src/transform/select_bonds.py` → `deprecated/src/transform/`
   - `src/transform/build_spreads.py` → `deprecated/src/transform/`
   - All TRACE scripts moved to `deprecated/scripts/`

2. **Environment cleanup:**
   - ✅ `.env.example` updated (removed FINRA keys, added SEC_USER_AGENT)
   - ✅ `.gitignore` already includes `.env`, `.env.*`, `/data/`

3. **New EDGAR modules created:**
   - ✅ `src/sec/__init__.py`
   - ✅ `src/sec/sec_client.py` (with rate limiting)
   - ✅ `src/sec/mapping.py` (ticker→CIK helpers)
   - ✅ `scripts/seed_issuers_edgar.py`
   - ✅ `scripts/ingest_sec_submissions.py`
   - ✅ `scripts/ingest_companyfacts.py`

4. **Transform modules:**
   - ✅ `src/transform/build_fundamentals_features.py` (45-day lag)
   - ✅ `src/transform/build_credit_proxy.py` (MERTON_SIMPLIFIED_V1)
   - ✅ `src/transform/build_model_dataset.py`
   - ✅ `src/transform/aggregate_weekly_market.py` (exists)
   - ✅ `src/transform/build_issuer_equity_features.py` (exists)

5. **Model modules:**
   - ✅ `src/models/train_spread_xgb.py` (updated for credit proxy)
   - ✅ `src/models/save_predictions.py` (updated for dcredit_proxy)
   - ✅ `src/models/shap_explain.py` (updated features, added compute_shap_for_models)

6. **Visualization:**
   - ✅ `src/viz/make_model_charts.py` (backtest, residuals, SHAP charts)

7. **Pipeline scripts:**
   - ✅ `scripts/pipeline_edgar_stage1_ingest.py`
   - ✅ `scripts/pipeline_edgar_stage2_features_targets.py`
   - ✅ `scripts/pipeline_edgar_stage3_model_train.py`
   - ✅ `scripts/summarize_findings.py`

8. **Database schema:**
   - ✅ Migration `007_edgar_revamp_tables.sql` created

## 🟡 Green Run Checklist (To Execute)

### Prerequisites
- [ ] PostgreSQL running and database `ai_credit_crisis` created
- [ ] `.env` file configured with:
  - `SEC_USER_AGENT="Your Name your.email@domain.com"`
  - `POSTGRES_*` variables
  - `FRED_API_KEY` (optional)
- [ ] Virtual environment activated
- [ ] Dependencies installed: `pip install -r requirements.txt`

### Stage 1: Ingestion
```bash
python3 scripts/pipeline_edgar_stage1_ingest.py
```

**Expected outputs:**
- [ ] `dim_issuer` has 15 issuers with CIKs
- [ ] `fact_fundamentals_quarterly` populated 2020–2025 (or best available)
- [ ] `fact_equity_price_daily` populated for issuers + SPY/QQQ/SMH
- [ ] `fact_equity_price_daily` has VIX, DGS2, DGS10, IG_OAS (if FRED_API_KEY set)

### Stage 2: Features and Targets
```bash
python3 scripts/pipeline_edgar_stage2_features_targets.py
```

**Expected outputs:**
- [ ] `fact_weekly_market` populated
- [ ] `feat_issuer_weekly` populated with eq_ret, eq_vol_21d
- [ ] `feat_issuer_weekly` has fundamentals features (net_debt, etc.)
- [ ] `fact_credit_proxy_weekly` populated with dcredit_proxy non-null for most weeks
- [ ] `model_dataset_weekly` populated

### Stage 3: Model Training
```bash
python3 scripts/pipeline_edgar_stage3_model_train.py
```

**Expected outputs:**
- [ ] `model_registry` has RISK_ON and RISK_OFF models
- [ ] `model_predictions_weekly` has 2025 test predictions
- [ ] SHAP stored (if model objects available)
- [ ] Charts saved to `reports/figures/`:
  - [ ] `backtest_actual_vs_pred.png`
  - [ ] `residuals.png`
  - [ ] `shap_summary.png`

### Stage 4: Summary
```bash
python3 scripts/summarize_findings.py
```

**Expected outputs:**
- [ ] 5 bullet plain-English recap printed
- [ ] Top 5 issuers with worst predicted deterioration in 2025 risk-off weeks
- [ ] Top 5 features driving risk-off outcomes (from SHAP)

## Notes

- **FRED_API_KEY**: Optional. If not set, pipeline will attempt CSV fallback or skip.
- **SHAP computation**: May fail if models aren't saved to disk (expected for now).
- **Charts**: Will be generated if predictions exist, even if SHAP is missing.

## Known Issues / Limitations

1. SHAP computation requires model objects to be saved/loaded (not yet implemented)
2. Some edge cases in credit proxy calculation may need refinement
3. FRED CSV fallback path needs to be created manually if API key not available

