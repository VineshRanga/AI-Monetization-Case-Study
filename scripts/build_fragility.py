#!/usr/bin/env python3
"""Build fragility scores."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scoring.build_fragility import build_fragility_score

if __name__ == "__main__":
    load_dotenv()
    
    print("="*60)
    print("Building Fragility Scores")
    print("="*60)
    
    try:
        result = build_fragility_score()
        
        scores_df = result["scores_df"]
        
        print(f"\nCreated {result['rows_created']} fragility scores")
        
        # Print ranking
        print("\nFragility Score Ranking:")
        print("-" * 80)
        ranked = scores_df.sort_values("total_score", ascending=False)
        for _, row in ranked.iterrows():
            print(f"{row['ticker']:6} | Total: {row['total_score']:5.1f} | "
                  f"Refi: {row['p_refi']*100:5.1f} | Cash/Capex: {row['p_cash_capex']*100:5.1f} | "
                  f"Leverage: {row['p_leverage']*100:5.1f} | Cyc/AI: {row['p_cyc_ai']*100:5.1f} | "
                  f"Structure: {row['p_structure']*100:5.1f}")
        
        print(f"\nScores exported to: data/processed/fragility_scores.csv")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

