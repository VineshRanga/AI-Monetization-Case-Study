# AI Credit Crisis Case Study

Quantitative credit risk analysis for 15 tech and data center issuers (2020-2025). Uses SEC EDGAR fundamentals, equity prices, and optional FRED macro data to build a regime-gated credit proxy model with scenario stress testing.

## What It Does

- Regime-gated XGBoost models predicting weekly credit proxy changes
- Fragility scoring (5 pillars: refinancing, cash, leverage, cyclicality, structure)
- Scenario engine for AI monetization shock and funding freeze stress tests
- All figures saved to `reports/figures/`

Full write-up and chart explanations: [White paper PDF link here]

## Quick Start

```bash
# Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Setup database
psql -U postgres
CREATE DATABASE ai_credit_crisis;
\q

# Copy env template and fill in values
cp .env.example .env
# Edit .env: set SEC_USER_AGENT (required), POSTGRES_* (if needed), FRED_API_KEY (optional)

# Initialize schema
python3 scripts/init_db.py

# Run pipeline stages
python3 scripts/pipeline_edgar_stage1_ingest.py
python3 scripts/pipeline_edgar_stage2_features_targets.py
python3 scripts/pipeline_edgar_stage3_model_train.py

# Generate fragility scores and scenarios
python3 scripts/run_fragility.py
python3 scripts/run_scenarios.py

# Generate all figures
python3 -m scripts.generate_report_figures
```

## Outputs

- Model predictions and metrics in Postgres
- Fragility scores (issuer-level, 5 pillars)
- Scenario results (baseline vs shock trajectories)
- 11 publication-quality figures in `reports/figures/`

## Limitations

- Credit proxy is not actual spreads (Merton-style distance-to-default approximation)
- Macro series (FRED) optional; regime uses equity-only fallback if missing
- Exploratory research project, not production risk system

## Data Sources

- SEC EDGAR: Free public fundamentals (requires User-Agent header)
- Stooq: Free historical equity prices
- FRED API: Optional macro series (VIX, rates, OAS)

See `NOTICES.md` for third-party data terms and required disclaimers.

## License

MIT
