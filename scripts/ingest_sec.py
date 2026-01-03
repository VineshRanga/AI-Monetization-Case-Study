#!/usr/bin/env python3
"""Ingest SEC fundamentals."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.ingest_sec_fundamentals import ingest_sec_fundamentals

if __name__ == "__main__":
    load_dotenv()
    
    print("="*60)
    print("Ingesting SEC Fundamentals")
    print("="*60)
    
    try:
        results = ingest_sec_fundamentals()
        
        print("\nCoverage Summary:")
        print("-" * 60)
        for ticker, result in results.items():
            print(f"  {ticker}: {result['rows']} periods ({result['status']})")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

