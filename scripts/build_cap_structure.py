#!/usr/bin/env python3
"""Build capital structure proxies."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transform.build_cap_structure_proxies import build_cap_structure_proxies

if __name__ == "__main__":
    load_dotenv()
    
    print("="*60)
    print("Building Capital Structure Proxies")
    print("="*60)
    
    try:
        result = build_cap_structure_proxies()
        print(f"\nCreated {result['rows_created']} capital structure records")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

