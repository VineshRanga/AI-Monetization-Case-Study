#!/usr/bin/env python3
"""Build modeling dataset."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transform.build_model_dataset import build_model_dataset

if __name__ == "__main__":
    load_dotenv()
    
    print("="*60)
    print("Building Modeling Dataset")
    print("="*60)
    
    try:
        result = build_model_dataset()
        print(f"\nCreated {result['rows_created']} rows")
        print("\nRegime distribution:")
        for regime, count in result['regime_counts'].items():
            print(f"  {regime}: {count}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

