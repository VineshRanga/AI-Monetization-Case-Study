"""Compute bond spreads vs U.S. Treasury yields."""
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import Dict

from src.db.db import get_engine


def choose_ust_tenor(maturity_date: date, trade_date: date) -> str:
    """
    Choose nearest UST tenor based on time-to-maturity.
    
    Available tenors: 2Y, 5Y, 7Y, 10Y, 20Y, 30Y
    """
    if not maturity_date:
        return "10Y"  # Default
    
    years_to_maturity = (maturity_date - trade_date).days / 365.25
    
    # Choose nearest tenor
    if years_to_maturity <= 3.5:
        return "2Y"
    elif years_to_maturity <= 6:
        return "5Y"
    elif years_to_maturity <= 8.5:
        return "7Y"
    elif years_to_maturity <= 15:
        return "10Y"
    elif years_to_maturity <= 25:
        return "20Y"
    else:
        return "30Y"


def build_spreads() -> Dict:
    """Compute bond spreads vs UST yields."""
    engine = get_engine()
    
    # Get all bond daily data with maturity info
    with engine.connect() as conn:
        query = text("""
            SELECT 
                f.cusip,
                f.trade_date,
                f.ytm,
                b.maturity_date
            FROM fact_bond_daily f
            JOIN dim_bond b ON f.cusip = b.cusip
            WHERE f.ytm IS NOT NULL
            ORDER BY f.cusip, f.trade_date
        """)
        
        bond_df = pd.read_sql(query, conn)
    
    if bond_df.empty:
        print("No bond data found")
        return {"rows_matched": 0, "rows_total": 0}
    
    # Get UST yields
    with engine.connect() as conn:
        query = text("""
            SELECT date, tenor, yield
            FROM fact_ust_yield_daily
            ORDER BY date, tenor
        """)
        
        ust_df = pd.read_sql(query, conn)
    
    if ust_df.empty:
        print("No UST yield data found")
        return {"rows_matched": 0, "rows_total": len(bond_df)}
    
    # Convert dates
    bond_df["trade_date"] = pd.to_datetime(bond_df["trade_date"]).dt.date
    bond_df["maturity_date"] = pd.to_datetime(bond_df["maturity_date"]).dt.date
    ust_df["date"] = pd.to_datetime(ust_df["date"]).dt.date
    
    # Compute spreads
    spreads = []
    matched = 0
    
    for _, bond_row in bond_df.iterrows():
        cusip = bond_row["cusip"]
        trade_date = bond_row["trade_date"]
        ytm = bond_row["ytm"]
        maturity_date = bond_row["maturity_date"]
        
        # Choose tenor
        tenor = choose_ust_tenor(maturity_date, trade_date)
        
        # Get UST yield
        ust_row = ust_df[
            (ust_df["date"] == trade_date) &
            (ust_df["tenor"] == tenor)
        ]
        
        if not ust_row.empty:
            ust_yield = ust_row.iloc[0]["yield"]
            spread_bps = (ytm - ust_yield) * 10000
            
            spreads.append({
                "cusip": cusip,
                "trade_date": trade_date,
                "ust_tenor": tenor,
                "ust_yield": ust_yield,
                "bond_ytm": ytm,
                "spread_bps": spread_bps,
            })
            matched += 1
    
    # Upsert spreads
    if spreads:
        spread_df = pd.DataFrame(spreads)
        
        with engine.begin() as conn:
            for _, row in spread_df.iterrows():
                conn.execute(
                    text("""
                        INSERT INTO fact_bond_spread_daily (
                            cusip, trade_date, ust_tenor, ust_yield, bond_ytm, spread_bps
                        )
                        VALUES (
                            :cusip, :trade_date, :ust_tenor, :ust_yield, :bond_ytm, :spread_bps
                        )
                        ON CONFLICT (cusip, trade_date) DO UPDATE SET
                            ust_tenor = EXCLUDED.ust_tenor,
                            ust_yield = EXCLUDED.ust_yield,
                            bond_ytm = EXCLUDED.bond_ytm,
                            spread_bps = EXCLUDED.spread_bps
                    """),
                    {
                        "cusip": str(row["cusip"]),
                        "trade_date": row["trade_date"],
                        "ust_tenor": str(row["ust_tenor"]),
                        "ust_yield": float(row["ust_yield"]),
                        "bond_ytm": float(row["bond_ytm"]),
                        "spread_bps": float(row["spread_bps"]),
                    }
                )
    
    match_rate = (matched / len(bond_df)) * 100 if len(bond_df) > 0 else 0
    
    return {
        "rows_matched": matched,
        "rows_total": len(bond_df),
        "match_rate": match_rate,
    }

