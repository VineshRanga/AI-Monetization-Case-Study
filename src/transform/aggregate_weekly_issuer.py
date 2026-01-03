"""Aggregate bond spreads to weekly issuer-level series."""
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


def aggregate_weekly_issuer(method: str = "TOP2_VOL_WEIGHTED") -> Dict:
    """
    Aggregate bond spreads to weekly issuer-level series.
    
    Methods:
    - TOP1_LIQUID: use the single most liquid bond per issuer
    - TOP2_VOL_WEIGHTED: take up to 2 selected bonds and compute volume-weighted average
    """
    engine = get_engine()
    
    # Get bond spreads with volume and issuer info
    with engine.connect() as conn:
        query = text("""
            SELECT 
                s.cusip,
                s.trade_date,
                s.spread_bps,
                b.issuer_id,
                i.ticker,
                COALESCE(d.volume, 0) as volume
            FROM fact_bond_spread_daily s
            JOIN dim_bond b ON s.cusip = b.cusip
            JOIN dim_issuer i ON b.issuer_id = i.issuer_id
            LEFT JOIN fact_bond_daily d ON s.cusip = d.cusip AND s.trade_date = d.trade_date
            WHERE b.is_selected = TRUE
            ORDER BY s.trade_date, i.ticker, s.cusip
        """)
        
        df = pd.read_sql(query, conn)
    
    if df.empty:
        print("No spread data found")
        return {"weeks_created": 0}
    
    # Convert dates
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["week_start"] = df["trade_date"].apply(get_week_start)
    
    # Aggregate to weekly by bond
    weekly_bond = df.groupby(["cusip", "week_start", "issuer_id", "ticker"]).agg({
        "spread_bps": "median",  # Robust to outliers
        "volume": "sum",
    }).reset_index()
    
    # Aggregate to issuer level
    weekly_issuer_data = []
    
    for (issuer_id, week_start), group in weekly_bond.groupby(["issuer_id", "week_start"]):
        ticker = group["ticker"].iloc[0]
        
        if method == "TOP1_LIQUID":
            # Use bond with highest volume in this week
            top_bond = group.nlargest(1, "volume")
            if not top_bond.empty:
                spread_bps = top_bond["spread_bps"].iloc[0]
                bond_count = 1
                method_used = "TOP1_LIQUID"
            else:
                continue
        
        elif method == "TOP2_VOL_WEIGHTED":
            # Use top 2 bonds by volume, volume-weighted average
            top_bonds = group.nlargest(2, "volume")
            if not top_bonds.empty:
                total_volume = top_bonds["volume"].sum()
                if total_volume > 0:
                    spread_bps = (top_bonds["spread_bps"] * top_bonds["volume"]).sum() / total_volume
                else:
                    spread_bps = top_bonds["spread_bps"].median()
                bond_count = len(top_bonds)
                method_used = "TOP2_VOL_WEIGHTED"
            else:
                continue
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        weekly_issuer_data.append({
            "issuer_id": issuer_id,
            "week_start": week_start,
            "spread_bps": float(spread_bps),
            "bond_count": bond_count,
            "method": method_used,
        })
    
    # Compute weekly change (dspread_bps)
    weekly_df = pd.DataFrame(weekly_issuer_data)
    weekly_df = weekly_df.sort_values(["issuer_id", "week_start"])
    
    weekly_df["dspread_bps"] = weekly_df.groupby("issuer_id")["spread_bps"].diff()
    
    # Upsert to database
    with engine.begin() as conn:
        for _, row in weekly_df.iterrows():
            conn.execute(
                text("""
                    INSERT INTO fact_issuer_spread_weekly (
                        issuer_id, week_start, spread_bps, dspread_bps, bond_count, method
                    )
                    VALUES (
                        :issuer_id, :week_start, :spread_bps, :dspread_bps, :bond_count, :method
                    )
                    ON CONFLICT (issuer_id, week_start) DO UPDATE SET
                        spread_bps = EXCLUDED.spread_bps,
                        dspread_bps = EXCLUDED.dspread_bps,
                        bond_count = EXCLUDED.bond_count,
                        method = EXCLUDED.method
                """),
                {
                    "issuer_id": int(row["issuer_id"]),
                    "week_start": row["week_start"],
                    "spread_bps": float(row["spread_bps"]) if pd.notna(row["spread_bps"]) else None,
                    "dspread_bps": float(row["dspread_bps"]) if pd.notna(row["dspread_bps"]) else None,
                    "bond_count": int(row["bond_count"]),
                    "method": str(row["method"]),
                }
            )
    
    return {
        "weeks_created": len(weekly_df),
        "issuers": weekly_df["issuer_id"].nunique(),
    }

