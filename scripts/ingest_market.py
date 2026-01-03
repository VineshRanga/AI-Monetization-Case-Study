#!/usr/bin/env python3
"""Ingest market data from FRED API."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.ingest_market import ingest_market_data

if __name__ == "__main__":
    load_dotenv()
    
    print("="*60)
    print("Ingesting Market Data")
    print("="*60)
    
    try:
        result = ingest_market_data()
        
        print(f"\nIngest Summary:")
        print(f"  Total rows: {result['total_rows']}")
        print("\nSeries breakdown:")
        for series, rows in result['series_results'].items():
            print(f"  {series}: {rows} rows")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

