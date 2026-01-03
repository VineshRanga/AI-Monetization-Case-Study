#!/usr/bin/env python3
"""Test FINRA access token retrieval."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.finra.auth import FinraAuthError, get_access_token

if __name__ == "__main__":
    load_dotenv()
    
    try:
        token = get_access_token()
        print("✓ FINRA token OK, prefix:", token[:12])
        print("✓ Token length:", len(token), "characters")
        sys.exit(0)
    except FinraAuthError as e:
        print(f"✗ Authentication failed: {e}")
        print("\nPlease ensure FINRA_CLIENT_ID and FINRA_CLIENT_SECRET are set in .env")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)

