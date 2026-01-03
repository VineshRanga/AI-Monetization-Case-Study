#!/usr/bin/env python3
"""Build issuer equity features."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transform.build_issuer_equity_features import build_issuer_equity_features

if __name__ == "__main__":
    load_dotenv()
    
    print("="*60)
    print("Building Issuer Equity Features")
    print("="*60)
    
    try:
        result = build_issuer_equity_features()
        print(f"\nCreated {result['rows_created']} issuer-week feature rows")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

