"""Build fundamentals lag features for weekly issuer features."""
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import Dict

from src.db.db import get_engine
from src.transform.aggregate_weekly_features import get_week_start


def build_fundamentals_features() -> Dict:
    """
    Build fundamentals lag features for feat_issuer_weekly.
    
    For each issuer-week, attach latest available quarterly fundamentals
    with a lag to avoid look-ahead (period_end <= week_start - 45 days).
    """
    engine = get_engine()
    
    # Schema validation: Check that required columns exist
    print("  Validating schema...")
    required_columns = [
        "fundamentals_lag_quarters",
        "net_debt",
        "net_debt_ebitda",
        "int_coverage",
        "fcf_margin",
        "capex_intensity",
    ]
    
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'feat_issuer_weekly'
            """)
        )
        existing_columns = {row[0] for row in result}
    
    missing_columns = [col for col in required_columns if col not in existing_columns]
    
    if missing_columns:
        print(f"  ✗ Error: Missing required columns in feat_issuer_weekly:")
        for col in missing_columns:
            print(f"     - {col}")
        print("\n  Run migration 008 to add missing columns:")
        print("     python3 scripts/run_migrations.py")
        print("  Or run pipeline_edgar_stage1_ingest.py (which runs all migrations)")
        import sys
        sys.exit(1)
    
    print("  ✓ Schema validation passed")
    
    # Get all issuer-weeks
    with engine.connect() as conn:
        issuer_weeks_df = pd.read_sql(
            text("""
                SELECT issuer_id, week_start
                FROM feat_issuer_weekly
                ORDER BY issuer_id, week_start
            """),
            conn
        )
    
    if issuer_weeks_df.empty:
        print("No issuer-weeks found. Run build_equity_features first.")
        return {"rows_updated": 0}
    
    # Get all fundamentals
    with engine.connect() as conn:
        fundamentals_df = pd.read_sql(
            text("""
                SELECT issuer_id, period_end, revenue, op_income, net_income,
                    total_debt, debt_current, long_term_debt, cash, interest_expense,
                    capex, cfo, fcf, shares_outstanding
                FROM fact_fundamentals_quarterly
                ORDER BY issuer_id, period_end
            """),
            conn
        )
    
    if fundamentals_df.empty:
        print("No fundamentals found. Run ingest_companyfacts first.")
        return {"rows_updated": 0}
    
    fundamentals_df["period_end"] = pd.to_datetime(fundamentals_df["period_end"]).dt.date
    
    # For each issuer-week, find latest fundamentals with lag
    rows_updated = 0
    
    for _, row in issuer_weeks_df.iterrows():
        issuer_id = row["issuer_id"]
        week_start = pd.to_datetime(row["week_start"]).date()
        
        # Lag cutoff: week_start - 45 days
        lag_cutoff = week_start - timedelta(days=45)
        
        # Get latest fundamentals for this issuer before lag cutoff
        issuer_fundamentals = fundamentals_df[
            (fundamentals_df["issuer_id"] == issuer_id) &
            (fundamentals_df["period_end"] <= lag_cutoff)
        ].sort_values("period_end", ascending=False)
        
        if issuer_fundamentals.empty:
            continue
        
        latest = issuer_fundamentals.iloc[0]
        
        # Compute derived metrics
        total_debt = latest["total_debt"]
        if total_debt is None:
            debt_current = latest["debt_current"]
            long_term_debt = latest["long_term_debt"]
            if debt_current is not None and long_term_debt is not None:
                total_debt = debt_current + long_term_debt
            elif long_term_debt is not None:
                total_debt = long_term_debt
        
        cash = latest["cash"]
        net_debt = None
        if total_debt is not None and cash is not None:
            net_debt = total_debt - cash
        
        # Net debt / EBITDA (or revenue proxy)
        net_debt_ebitda = None
        if net_debt is not None:
            ebitda = latest["op_income"]  # Approximate EBITDA with op_income
            if ebitda is not None and ebitda > 0:
                net_debt_ebitda = net_debt / ebitda
            elif latest["revenue"] is not None and latest["revenue"] > 0:
                # Fallback to revenue
                net_debt_ebitda = net_debt / latest["revenue"]
        
        # Interest coverage
        int_coverage = None
        op_income = latest["op_income"]
        interest_expense = latest["interest_expense"]
        if op_income is not None and interest_expense is not None and interest_expense > 0:
            int_coverage = op_income / interest_expense
        
        # FCF margin
        fcf_margin = None
        fcf = latest["fcf"]
        revenue = latest["revenue"]
        if fcf is not None and revenue is not None and revenue > 0:
            fcf_margin = fcf / revenue
        
        # Capex intensity
        capex_intensity = None
        capex = latest["capex"]
        if capex is not None and revenue is not None and revenue > 0:
            capex_intensity = capex / revenue
        
        # Compute lag in quarters (approximate)
        fundamentals_lag_quarters = None
        if total_debt is not None:  # If we have data, compute lag
            days_lag = (week_start - latest["period_end"]).days
            fundamentals_lag_quarters = max(0, days_lag // 90)  # Approximate quarters
        
        # Update feat_issuer_weekly
        with engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE feat_issuer_weekly
                    SET fundamentals_lag_quarters = :fundamentals_lag_quarters,
                        net_debt = :net_debt,
                        net_debt_ebitda = :net_debt_ebitda,
                        int_coverage = :int_coverage,
                        fcf_margin = :fcf_margin,
                        capex_intensity = :capex_intensity
                    WHERE issuer_id = :issuer_id AND week_start = :week_start
                """),
                {
                    "issuer_id": issuer_id,
                    "week_start": week_start,
                    "fundamentals_lag_quarters": fundamentals_lag_quarters,
                    "net_debt": float(net_debt) if net_debt is not None else None,
                    "net_debt_ebitda": float(net_debt_ebitda) if net_debt_ebitda is not None else None,
                    "int_coverage": float(int_coverage) if int_coverage is not None else None,
                    "fcf_margin": float(fcf_margin) if fcf_margin is not None else None,
                    "capex_intensity": float(capex_intensity) if capex_intensity is not None else None,
                }
            )
            rows_updated += 1
    
    return {"rows_updated": rows_updated}


if __name__ == "__main__":
    result = build_fundamentals_features()
    print(f"Updated {result['rows_updated']} issuer-weeks with fundamentals features")

