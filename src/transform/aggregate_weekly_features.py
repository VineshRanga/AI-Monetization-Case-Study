"""Aggregate daily market data to weekly features."""
import sys
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import Dict

from src.db.db import get_engine


def get_week_start(d: date) -> date:
    """Get Monday of the week containing date d."""
    days_since_monday = d.weekday()
    return d - timedelta(days=days_since_monday)


def aggregate_weekly_market() -> Dict:
    """Aggregate daily market data to weekly features."""
    engine = get_engine()
    
    # Validation: Check if required data exists
    print("  Validating data availability...")
    with engine.connect() as conn:
        equity_count = conn.execute(text("SELECT COUNT(*) FROM fact_equity_price_daily")).fetchone()[0]
        macro_count = conn.execute(
            text("SELECT COUNT(*) FROM fact_equity_price_daily WHERE symbol IN ('VIX', 'DGS2', 'DGS10', 'IG_OAS')")
        ).fetchone()[0]
    
    if equity_count == 0:
        print("  ✗ Error: No equity price data available.")
        print("     Run pipeline_edgar_stage1_ingest.py and ensure equity ingestion succeeded.")
        sys.exit(1)
    
    if macro_count == 0:
        print("  ⚠ Warning: No macro data (VIX, rates, OAS) available.")
        print("     FRED ingestion may have failed. Continuing with equity-only features...")
    
    print(f"  ✓ Found {equity_count} equity price rows, {macro_count} macro rows")
    
    # Get SPY data (proxy for SPX)
    with engine.connect() as conn:
        spy_df = pd.read_sql(
            text("""
                SELECT date, close, adj_close
                FROM fact_equity_price_daily
                WHERE symbol = 'SPY'
                ORDER BY date
            """),
            conn
        )
    
    if spy_df.empty or "date" not in spy_df.columns:
        print("  ⚠ Warning: No SPY data available. Skipping SPX return features.")
        spy_df = pd.DataFrame(columns=["date", "close", "adj_close", "week_start"])
    
    # Get QQQ data
    with engine.connect() as conn:
        qqq_df = pd.read_sql(
            text("""
                SELECT date, close, adj_close
                FROM fact_equity_price_daily
                WHERE symbol = 'QQQ'
                ORDER BY date
            """),
            conn
        )
    
    if qqq_df.empty or "date" not in qqq_df.columns:
        print("  ⚠ Warning: No QQQ data available. Skipping QQQ return features.")
        qqq_df = pd.DataFrame(columns=["date", "close", "adj_close", "week_start"])
    
    # Get SMH data
    with engine.connect() as conn:
        smh_df = pd.read_sql(
            text("""
                SELECT date, close, adj_close
                FROM fact_equity_price_daily
                WHERE symbol = 'SMH'
                ORDER BY date
            """),
            conn
        )
    
    if smh_df.empty or "date" not in smh_df.columns:
        print("  ⚠ Warning: No SMH data available. Skipping SMH return features.")
        smh_df = pd.DataFrame(columns=["date", "close", "adj_close", "week_start"])
    
    # Get SRVR data (optional)
    with engine.connect() as conn:
        srvr_df = pd.read_sql(
            text("""
                SELECT date, close, adj_close
                FROM fact_equity_price_daily
                WHERE symbol = 'SRVR'
                ORDER BY date
            """),
            conn
        )
    
    if srvr_df.empty or "date" not in srvr_df.columns:
        print("  ⚠ Warning: No SRVR data available. Skipping SRVR return features.")
        srvr_df = pd.DataFrame(columns=["date", "close", "adj_close", "week_start"])
    
    # Get VIX data
    with engine.connect() as conn:
        vix_df = pd.read_sql(
            text("""
                SELECT date, close
                FROM fact_equity_price_daily
                WHERE symbol = 'VIX'
                ORDER BY date
            """),
            conn
        )
    
    if vix_df.empty or "date" not in vix_df.columns:
        print("  ⚠ Warning: No VIX data available (FRED ingestion empty). Skipping VIX weekly features.")
        vix_df = pd.DataFrame(columns=["date", "close", "week_start"])
    
    # Get DGS2 data
    with engine.connect() as conn:
        dgs2_df = pd.read_sql(
            text("""
                SELECT date, close
                FROM fact_equity_price_daily
                WHERE symbol = 'DGS2'
                ORDER BY date
            """),
            conn
        )
    
    if dgs2_df.empty or "date" not in dgs2_df.columns:
        print("  ⚠ Warning: No DGS2 data available. Skipping DGS2 weekly features.")
        dgs2_df = pd.DataFrame(columns=["date", "close", "week_start"])
    
    # Get DGS10 data
    with engine.connect() as conn:
        dgs10_df = pd.read_sql(
            text("""
                SELECT date, close
                FROM fact_equity_price_daily
                WHERE symbol = 'DGS10'
                ORDER BY date
            """),
            conn
        )
    
    if dgs10_df.empty or "date" not in dgs10_df.columns:
        print("  ⚠ Warning: No DGS10 data available. Skipping DGS10 weekly features.")
        dgs10_df = pd.DataFrame(columns=["date", "close", "week_start"])
    
    # Get IG_OAS data
    with engine.connect() as conn:
        ig_oas_df = pd.read_sql(
            text("""
                SELECT date, close
                FROM fact_equity_price_daily
                WHERE symbol = 'IG_OAS'
                ORDER BY date
            """),
            conn
        )
    
    if ig_oas_df.empty or "date" not in ig_oas_df.columns:
        print("  ⚠ Warning: No IG OAS data available. Skipping IG OAS weekly features.")
        ig_oas_df = pd.DataFrame(columns=["date", "close", "week_start"])
    
    # Convert dates and create week_start column
    # Ensure week_start is created BEFORE any filtering operations
    for df_name, df in [
        ("SPY", spy_df), ("QQQ", qqq_df), ("SMH", smh_df), ("SRVR", srvr_df),
        ("VIX", vix_df), ("DGS2", dgs2_df), ("DGS10", dgs10_df), ("IG_OAS", ig_oas_df)
    ]:
        if df.empty:
            # Create empty week_start column for consistency
            df["week_start"] = pd.Series(dtype='object')
            continue
        
        if "date" not in df.columns:
            print(f"  ⚠ Warning: {df_name} dataframe missing 'date' column. Skipping.")
            df["week_start"] = pd.Series(dtype='object')
            continue
        
        # Convert date and create week_start
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["week_start"] = df["date"].apply(get_week_start)
    
    # Aggregate by week
    weekly_data = []
    
    # Get all unique week_start values
    all_weeks = set()
    for df in [spy_df, qqq_df, smh_df, srvr_df, vix_df, dgs2_df, dgs10_df, ig_oas_df]:
        if not df.empty:
            all_weeks.update(df["week_start"].unique())
    
    for week_start in sorted(all_weeks):
        week_data = {"week_start": week_start}
        
        # SPY return (with guardrails)
        if not spy_df.empty and "week_start" in spy_df.columns:
            spy_week = spy_df[spy_df["week_start"] == week_start]
            if not spy_week.empty and "adj_close" in spy_week.columns:
                first_close = spy_week.iloc[0]["adj_close"]
                last_close = spy_week.iloc[-1]["adj_close"]
                if first_close > 0:
                    week_data["spx_ret"] = (last_close / first_close) - 1
        
        # QQQ return (with guardrails)
        if not qqq_df.empty and "week_start" in qqq_df.columns:
            qqq_week = qqq_df[qqq_df["week_start"] == week_start]
            if not qqq_week.empty and "adj_close" in qqq_week.columns:
                first_close = qqq_week.iloc[0]["adj_close"]
                last_close = qqq_week.iloc[-1]["adj_close"]
                if first_close > 0:
                    week_data["qqq_ret"] = (last_close / first_close) - 1
        
        # SMH return (with guardrails)
        if not smh_df.empty and "week_start" in smh_df.columns:
            smh_week = smh_df[smh_df["week_start"] == week_start]
            if not smh_week.empty and "adj_close" in smh_week.columns:
                first_close = smh_week.iloc[0]["adj_close"]
                last_close = smh_week.iloc[-1]["adj_close"]
                if first_close > 0:
                    week_data["smh_ret"] = (last_close / first_close) - 1
        
        # SRVR return (with guardrails)
        if not srvr_df.empty and "week_start" in srvr_df.columns:
            srvr_week = srvr_df[srvr_df["week_start"] == week_start]
            if not srvr_week.empty and "adj_close" in srvr_week.columns:
                first_close = srvr_week.iloc[0]["adj_close"]
                last_close = srvr_week.iloc[-1]["adj_close"]
                if first_close > 0:
                    week_data["srvr_ret"] = (last_close / first_close) - 1
        
        # VIX level and change (with guardrails)
        if not vix_df.empty and "week_start" in vix_df.columns:
            vix_week = vix_df[vix_df["week_start"] == week_start]
            if not vix_week.empty and "close" in vix_week.columns:
                week_data["vix_level"] = vix_week.iloc[-1]["close"]
                week_data["vix_chg"] = vix_week.iloc[-1]["close"] - vix_week.iloc[0]["close"]
        
        # DGS2 change (with guardrails)
        if not dgs2_df.empty and "week_start" in dgs2_df.columns:
            dgs2_week = dgs2_df[dgs2_df["week_start"] == week_start]
            if not dgs2_week.empty and "close" in dgs2_week.columns:
                week_data["dgs2_chg"] = dgs2_week.iloc[-1]["close"] - dgs2_week.iloc[0]["close"]
        
        # DGS10 change (with guardrails)
        if not dgs10_df.empty and "week_start" in dgs10_df.columns:
            dgs10_week = dgs10_df[dgs10_df["week_start"] == week_start]
            if not dgs10_week.empty and "close" in dgs10_week.columns:
                week_data["dgs10_chg"] = dgs10_week.iloc[-1]["close"] - dgs10_week.iloc[0]["close"]
        
        # Curve 2s10s change (with guardrails)
        if (not dgs2_df.empty and "week_start" in dgs2_df.columns and
            not dgs10_df.empty and "week_start" in dgs10_df.columns):
            dgs2_week = dgs2_df[dgs2_df["week_start"] == week_start]
            dgs10_week = dgs10_df[dgs10_df["week_start"] == week_start]
            if (not dgs2_week.empty and "close" in dgs2_week.columns and
                not dgs10_week.empty and "close" in dgs10_week.columns):
                curve_first = dgs10_week.iloc[0]["close"] - dgs2_week.iloc[0]["close"]
                curve_last = dgs10_week.iloc[-1]["close"] - dgs2_week.iloc[-1]["close"]
                week_data["curve_2s10s_chg"] = curve_last - curve_first
        
        # IG OAS change (with guardrails)
        if not ig_oas_df.empty and "week_start" in ig_oas_df.columns:
            ig_oas_week = ig_oas_df[ig_oas_df["week_start"] == week_start]
            if not ig_oas_week.empty and "close" in ig_oas_week.columns:
                week_data["ig_oas_chg"] = ig_oas_week.iloc[-1]["close"] - ig_oas_week.iloc[0]["close"]
        
        weekly_data.append(week_data)
    
    # Upsert to database
    with engine.begin() as conn:
        for week_data in weekly_data:
            conn.execute(
                text("""
                    INSERT INTO fact_weekly_market (
                        week_start, spx_ret, qqq_ret, vix_level, vix_chg,
                        dgs2_chg, dgs10_chg, curve_2s10s_chg, ig_oas_chg,
                        hy_oas_chg, smh_ret, srvr_ret
                    )
                    VALUES (
                        :week_start, :spx_ret, :qqq_ret, :vix_level, :vix_chg,
                        :dgs2_chg, :dgs10_chg, :curve_2s10s_chg, :ig_oas_chg,
                        :hy_oas_chg, :smh_ret, :srvr_ret
                    )
                    ON CONFLICT (week_start) DO UPDATE SET
                        spx_ret = EXCLUDED.spx_ret,
                        qqq_ret = EXCLUDED.qqq_ret,
                        vix_level = EXCLUDED.vix_level,
                        vix_chg = EXCLUDED.vix_chg,
                        dgs2_chg = EXCLUDED.dgs2_chg,
                        dgs10_chg = EXCLUDED.dgs10_chg,
                        curve_2s10s_chg = EXCLUDED.curve_2s10s_chg,
                        ig_oas_chg = EXCLUDED.ig_oas_chg,
                        hy_oas_chg = EXCLUDED.hy_oas_chg,
                        smh_ret = EXCLUDED.smh_ret,
                        srvr_ret = EXCLUDED.srvr_ret
                """),
                {
                    "week_start": week_data["week_start"],
                    "spx_ret": week_data.get("spx_ret"),
                    "qqq_ret": week_data.get("qqq_ret"),
                    "vix_level": week_data.get("vix_level"),
                    "vix_chg": week_data.get("vix_chg"),
                    "dgs2_chg": week_data.get("dgs2_chg"),
                    "dgs10_chg": week_data.get("dgs10_chg"),
                    "curve_2s10s_chg": week_data.get("curve_2s10s_chg"),
                    "ig_oas_chg": week_data.get("ig_oas_chg"),
                    "hy_oas_chg": None,  # Not implemented yet
                    "smh_ret": week_data.get("smh_ret"),
                    "srvr_ret": week_data.get("srvr_ret"),
                }
            )
    
    return {"weeks_created": len(weekly_data)}

