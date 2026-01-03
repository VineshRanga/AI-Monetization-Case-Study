#!/usr/bin/env python3
"""Aggregate daily market data to weekly features."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transform.aggregate_weekly_features import aggregate_weekly_market

if __name__ == "__main__":
    load_dotenv()
    
    print("="*60)
    print("Aggregating Weekly Market Features")
    print("="*60)
    
    try:
        result = aggregate_weekly_market()
        print(f"\nCreated {result['weeks_created']} weekly market records")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

