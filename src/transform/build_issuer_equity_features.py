"""Build issuer equity features (returns, volatility)."""
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import Dict

from src.db.db import get_engine
from src.transform.aggregate_weekly_features import get_week_start


def build_issuer_equity_features() -> Dict:
    """Build weekly issuer equity features."""
    engine = get_engine()
    
    # Get all issuers
    with engine.connect() as conn:
        issuers_df = pd.read_sql(
            text("SELECT issuer_id, ticker, bucket FROM dim_issuer ORDER BY ticker"),
            conn
        )
    
    if issuers_df.empty:
        print("No issuers found")
        return {"rows_created": 0}
    
    # Get equity prices for all issuer tickers
    tickers = issuers_df["ticker"].tolist()
    ticker_str = "', '".join(tickers)
    
    with engine.connect() as conn:
        equity_df = pd.read_sql(
            text(f"""
                SELECT symbol, date, adj_close, close
                FROM fact_equity_price_daily
                WHERE symbol IN ('{ticker_str}')
                ORDER BY symbol, date
            """),
            conn
        )
    
    if equity_df.empty:
        print("No equity price data found")
        return {"rows_created": 0}
    
    equity_df["date"] = pd.to_datetime(equity_df["date"]).dt.date
    equity_df["week_start"] = equity_df["date"].apply(get_week_start)
    
    # Compute daily returns
    equity_df = equity_df.sort_values(["symbol", "date"])
    equity_df["daily_ret"] = equity_df.groupby("symbol")["adj_close"].pct_change()
    
    # Build features for each issuer-week
    feature_rows = []
    
    for _, issuer_row in issuers_df.iterrows():
        issuer_id = issuer_row["issuer_id"]
        ticker = issuer_row["ticker"]
        bucket = issuer_row["bucket"]
        
        issuer_equity = equity_df[equity_df["symbol"] == ticker].copy()
        
        if issuer_equity.empty:
            continue
        
        # Group by week
        for week_start, week_data in issuer_equity.groupby("week_start"):
            week_data = week_data.sort_values("date")
            
            # Weekly return
            if len(week_data) > 0:
                first_close = week_data.iloc[0]["adj_close"]
                last_close = week_data.iloc[-1]["adj_close"]
                if first_close > 0:
                    eq_ret = (last_close / first_close) - 1
                else:
                    eq_ret = None
            else:
                eq_ret = None
            
            # 21-trading-day rolling volatility
            # Get last date of week
            week_end_date = week_data.iloc[-1]["date"]
            
            # Get 21 trading days before week_end_date
            window_start = week_end_date - timedelta(days=42)  # Approximate
            
            window_data = issuer_equity[
                (issuer_equity["date"] >= window_start) &
                (issuer_equity["date"] <= week_end_date)
            ]
            
            if len(window_data) >= 10:  # Need at least 10 days
                daily_returns = window_data["daily_ret"].dropna()
                if len(daily_returns) > 0:
                    # Annualized volatility
                    eq_vol_21d = daily_returns.std() * np.sqrt(252)
                else:
                    eq_vol_21d = None
            else:
                eq_vol_21d = None
            
            feature_rows.append({
                "issuer_id": issuer_id,
                "week_start": week_start,
                "eq_ret": eq_ret,
                "eq_vol_21d": eq_vol_21d,
                "bucket": bucket,
            })
    
    # Upsert to database
    if feature_rows:
        with engine.begin() as conn:
            for row in feature_rows:
                conn.execute(
                    text("""
                        INSERT INTO feat_issuer_weekly (
                            issuer_id, week_start, eq_ret, eq_vol_21d, bucket
                        )
                        VALUES (
                            :issuer_id, :week_start, :eq_ret, :eq_vol_21d, :bucket
                        )
                        ON CONFLICT (issuer_id, week_start) DO UPDATE SET
                            eq_ret = EXCLUDED.eq_ret,
                            eq_vol_21d = EXCLUDED.eq_vol_21d,
                            bucket = EXCLUDED.bucket
                    """),
                    {
                        "issuer_id": int(row["issuer_id"]),
                        "week_start": row["week_start"],
                        "eq_ret": float(row["eq_ret"]) if row["eq_ret"] is not None else None,
                        "eq_vol_21d": float(row["eq_vol_21d"]) if row["eq_vol_21d"] is not None else None,
                        "bucket": str(row["bucket"]),
                    }
                )
    
    return {"rows_created": len(feature_rows)}

