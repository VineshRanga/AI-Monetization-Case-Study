#!/usr/bin/env python3
"""Compute bond spreads vs UST yields."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transform.build_spreads import build_spreads

if __name__ == "__main__":
    load_dotenv()
    
    print("="*60)
    print("Computing Bond Spreads")
    print("="*60)
    
    try:
        result = build_spreads()
        
        print(f"\nSpread Computation Summary:")
        print(f"  Total bond rows: {result['rows_total']}")
        print(f"  Matched to UST: {result['rows_matched']}")
        print(f"  Match rate: {result['match_rate']:.1f}%")
        
        # Show sample spreads
        from src.db.db import get_engine
        from sqlalchemy import text
        
        engine = get_engine()
        with engine.connect() as conn:
            samples = conn.execute(
                text("""
                    SELECT 
                        s.cusip,
                        b.issuer_name,
                        s.trade_date,
                        s.ust_tenor,
                        s.spread_bps
                    FROM fact_bond_spread_daily s
                    JOIN dim_bond b ON s.cusip = b.cusip
                    ORDER BY s.trade_date DESC
                    LIMIT 10
                """)
            ).fetchall()
            
            if samples:
                print("\nSample spreads (10 most recent):")
                print("-" * 60)
                for row in samples:
                    print(f"  {row[0]} | {row[1][:30]:30} | {row[2]} | {row[3]} | {row[4]:.1f} bps")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

