#!/usr/bin/env python3
"""Build regime labels."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.db import get_engine
from src.models.build_regime_labels import build_regime_labels
from sqlalchemy import text

if __name__ == "__main__":
    load_dotenv()
    
    print("="*60)
    print("Building Regime Labels")
    print("="*60)
    
    # Check if weekly market data exists
    engine = get_engine()
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM fact_weekly_market")
        ).fetchone()[0]
    
    if count == 0:
        print("No weekly market data found. Run scripts/aggregate_weekly_features.py first.")
        sys.exit(1)
    
    try:
        result = build_regime_labels()
        
        print(f"\nRegime Labeling Summary:")
        print(f"  Weeks labeled: {result['weeks_labeled']}")
        print(f"  RISK_OFF weeks: {result['risk_off_count']} ({result['risk_off_pct']:.1f}%)")
        
        # Show top 10 risk-off weeks
        with engine.connect() as conn:
            top_risk_off = conn.execute(
                text("""
                    SELECT week_start, prob_risk_off, regime_label
                    FROM model_regime_weekly
                    ORDER BY prob_risk_off DESC
                    LIMIT 10
                """)
            ).fetchall()
            
            print("\nTop 10 risk-off weeks by probability:")
            print("-" * 60)
            for row in top_risk_off:
                print(f"  {row[0]} | prob: {row[1]:.3f} | {row[2]}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

