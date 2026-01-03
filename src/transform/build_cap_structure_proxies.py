"""Build capital structure proxies from bond universe."""
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import Dict, Optional

from src.db.db import get_engine


def build_cap_structure_proxies(asof_date: Optional[date] = None) -> Dict:
    """Build capital structure proxies from bond universe."""
    engine = get_engine()
    
    if asof_date is None:
        # Use latest period_end from fundamentals, or default to 2025-12-31
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT MAX(period_end) FROM fact_fundamentals_quarterly")
            ).fetchone()
            
            if result[0]:
                asof_date = result[0]
            else:
                asof_date = date(2025, 12, 31)
    
    print(f"Computing capital structure proxies as of {asof_date}")
    
    # Get selected bonds with maturity dates
    with engine.connect() as conn:
        bonds_df = pd.read_sql(
            text("""
                SELECT 
                    b.issuer_id,
                    b.cusip,
                    b.maturity_date,
                    i.ticker
                FROM dim_bond b
                JOIN dim_issuer i ON b.issuer_id = i.issuer_id
                WHERE b.is_selected = TRUE
                    AND b.maturity_date IS NOT NULL
            """),
            conn
        )
    
    if bonds_df.empty:
        print("No selected bonds found")
        return {"rows_created": 0}
    
    bonds_df["maturity_date"] = pd.to_datetime(bonds_df["maturity_date"]).dt.date
    bonds_df["asof_date"] = asof_date
    
    # Compute maturity buckets
    bonds_df["maturity_0_12m"] = (
        (bonds_df["maturity_date"] >= asof_date) &
        (bonds_df["maturity_date"] <= asof_date + timedelta(days=365))
    ).astype(int)
    
    bonds_df["maturity_12_24m"] = (
        (bonds_df["maturity_date"] > asof_date + timedelta(days=365)) &
        (bonds_df["maturity_date"] <= asof_date + timedelta(days=730))
    ).astype(int)
    
    bonds_df["maturity_24_60m"] = (
        (bonds_df["maturity_date"] > asof_date + timedelta(days=730)) &
        (bonds_df["maturity_date"] <= asof_date + timedelta(days=1825))
    ).astype(int)
    
    bonds_df["maturity_60m_plus"] = (
        bonds_df["maturity_date"] > asof_date + timedelta(days=1825)
    ).astype(int)
    
    # Aggregate by issuer
    cap_structure = []
    
    for issuer_id in bonds_df["issuer_id"].unique():
        issuer_bonds = bonds_df[bonds_df["issuer_id"] == issuer_id]
        total_bonds = len(issuer_bonds)
        
        if total_bonds == 0:
            continue
        
        maturity_0_12m = issuer_bonds["maturity_0_12m"].sum()
        maturity_12_24m = issuer_bonds["maturity_12_24m"].sum()
        maturity_24_60m = issuer_bonds["maturity_24_60m"].sum()
        maturity_60m_plus = issuer_bonds["maturity_60m_plus"].sum()
        
        # Compute maturity_24m_pct (proxy)
        maturity_24m_pct = (maturity_0_12m + maturity_12_24m) / total_bonds if total_bonds > 0 else None
        
        cap_structure.append({
            "issuer_id": issuer_id,
            "asof_date": asof_date,
            "maturity_0_12m": maturity_0_12m,
            "maturity_12_24m": maturity_12_24m,
            "maturity_24_60m": maturity_24_60m,
            "maturity_60m_plus": maturity_60m_plus,
            "maturity_24m_pct": maturity_24m_pct,
        })
    
    # Upsert to database
    with engine.begin() as conn:
        for row in cap_structure:
            conn.execute(
                text("""
                    INSERT INTO fact_cap_structure (
                        issuer_id, asof_date,
                        maturity_0_12m, maturity_12_24m, maturity_24_60m, maturity_60m_plus,
                        source
                    )
                    VALUES (
                        :issuer_id, :asof_date,
                        :maturity_0_12m, :maturity_12_24m, :maturity_24_60m, :maturity_60m_plus,
                        'BOND_UNIVERSE'
                    )
                    ON CONFLICT (issuer_id, asof_date) DO UPDATE SET
                        maturity_0_12m = EXCLUDED.maturity_0_12m,
                        maturity_12_24m = EXCLUDED.maturity_12_24m,
                        maturity_24_60m = EXCLUDED.maturity_24_60m,
                        maturity_60m_plus = EXCLUDED.maturity_60m_plus
                """),
                {
                    "issuer_id": int(row["issuer_id"]),
                    "asof_date": row["asof_date"],
                    "maturity_0_12m": int(row["maturity_0_12m"]),
                    "maturity_12_24m": int(row["maturity_12_24m"]),
                    "maturity_24_60m": int(row["maturity_24_60m"]),
                    "maturity_60m_plus": int(row["maturity_60m_plus"]),
                }
            )
    
    return {"rows_created": len(cap_structure)}

