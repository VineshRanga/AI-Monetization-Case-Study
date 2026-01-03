#!/usr/bin/env python3
"""Ingest TRACE time series for selected bonds."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.db import get_engine
from src.ingest.ingest_trace_timeseries import ingest_trace_timeseries
from sqlalchemy import text

if __name__ == "__main__":
    load_dotenv()
    
    print("="*60)
    print("Ingesting TRACE Time Series")
    print("="*60)
    
    engine = get_engine()
    
    # Check selected bonds
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM dim_bond WHERE is_selected = TRUE")
        )
        count = result.fetchone()[0]
    
    if count == 0:
        print("No bonds selected. Run scripts/select_bonds.py first.")
        sys.exit(1)
    
    print(f"\nIngesting data for {count} selected bonds...")
    
    try:
        result = ingest_trace_timeseries()
        
        print("\n" + "="*60)
        print("Ingest Summary")
        print("="*60)
        
        summaries = result.get("summaries", [])
        if summaries:
            import pandas as pd
            df = pd.DataFrame(summaries)
            
            # Group by issuer
            with engine.connect() as conn:
                for row in conn.execute(
                    text("""
                        SELECT i.ticker, i.issuer_name, COUNT(DISTINCT b.cusip) as bond_count
                        FROM dim_issuer i
                        JOIN dim_bond b ON i.issuer_id = b.issuer_id
                        WHERE b.is_selected = TRUE
                        GROUP BY i.ticker, i.issuer_name
                        ORDER BY i.ticker
                    """)
                ):
                    ticker, name, bond_count = row
                    issuer_data = df[df['cusip'].isin([
                        c for c in summaries if any(
                            s['cusip'] == c for s in summaries
                        )
                    ])]
                    # Get actual data for this issuer's bonds
                    issuer_bonds = [
                        s for s in summaries
                        if any(
                            b['cusip'] == s['cusip']
                            for b in [
                                {"cusip": c} for c in [
                                    r[0] for r in conn.execute(
                                        text("""
                                            SELECT b.cusip
                                            FROM dim_bond b
                                            JOIN dim_issuer i ON b.issuer_id = i.issuer_id
                                            WHERE i.ticker = :ticker AND b.is_selected = TRUE
                                        """),
                                        {"ticker": ticker}
                                    )
                                ]
                            ]
                        )
                    ]
                    
                    if issuer_bonds:
                        total_rows = sum(s['rows_inserted'] for s in issuer_bonds)
                        date_min = min(s['date_min'] for s in issuer_bonds if s.get('date_min'))
                        date_max = max(s['date_max'] for s in issuer_bonds if s.get('date_max'))
                        print(f"{ticker}: {total_rows} rows, {date_min} to {date_max}")
        
        print(f"\nTotal CUSIPs processed: {len(summaries)}")
        print(f"Summary saved to: data/processed/trace_ingest_summary.csv")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

