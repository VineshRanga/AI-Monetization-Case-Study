#!/usr/bin/env python3
"""Build weekly issuer-level spread series."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.db import get_engine
from src.transform.aggregate_weekly_issuer import aggregate_weekly_issuer
from sqlalchemy import text

if __name__ == "__main__":
    load_dotenv()
    
    print("="*60)
    print("Building Weekly Issuer Spread Series")
    print("="*60)
    
    method = "TOP2_VOL_WEIGHTED"  # Can be changed to TOP1_LIQUID
    
    try:
        result = aggregate_weekly_issuer(method=method)
        
        print(f"\nAggregation Summary:")
        print(f"  Method: {method}")
        print(f"  Weeks created: {result['weeks_created']}")
        print(f"  Issuers: {result['issuers']}")
        
        # Show summary by issuer
        engine = get_engine()
        with engine.connect() as conn:
            summary = conn.execute(
                text("""
                    SELECT 
                        i.ticker,
                        COUNT(*) as weeks,
                        MIN(week_start) as min_week,
                        MAX(week_start) as max_week
                    FROM fact_issuer_spread_weekly w
                    JOIN dim_issuer i ON w.issuer_id = i.issuer_id
                    GROUP BY i.ticker
                    ORDER BY i.ticker
                """)
            ).fetchall()
            
            print("\nWeeks per issuer:")
            print("-" * 60)
            for row in summary:
                print(f"  {row[0]:6} {row[1]:4} weeks, {row[2]} to {row[3]}")
            
            # Show sample for one issuer
            sample_issuer = conn.execute(
                text("""
                    SELECT i.ticker
                    FROM dim_issuer i
                    JOIN fact_issuer_spread_weekly w ON i.issuer_id = w.issuer_id
                    LIMIT 1
                """)
            ).fetchone()
            
            if sample_issuer:
                ticker = sample_issuer[0]
                samples = conn.execute(
                    text("""
                        SELECT week_start, spread_bps, dspread_bps, bond_count
                        FROM fact_issuer_spread_weekly w
                        JOIN dim_issuer i ON w.issuer_id = i.issuer_id
                        WHERE i.ticker = :ticker
                        ORDER BY week_start DESC
                        LIMIT 10
                    """),
                    {"ticker": ticker}
                ).fetchall()
                
                print(f"\nSample weeks for {ticker} (10 most recent):")
                print("-" * 60)
                for row in samples:
                    print(f"  {row[0]} | spread: {row[1]:.1f} bps | Δspread: {row[2]:.1f if row[2] else 'N/A'} bps | bonds: {row[3]}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

