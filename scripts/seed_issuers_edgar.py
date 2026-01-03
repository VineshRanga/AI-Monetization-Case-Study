#!/usr/bin/env python3
"""Seed issuers from SEC company tickers exchange mapping."""
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.universe import ISSUER_UNIVERSE
from src.db.db import get_engine
from src.sec.sec_client import get_company_tickers_exchange

if __name__ == "__main__":
    load_dotenv()
    
    print("="*80)
    print("Seeding Issuers from SEC EDGAR")
    print("="*80)
    
    # Get SEC ticker mapping
    print("\n[1/2] Fetching SEC company tickers exchange mapping...")
    try:
        tickers_data = get_company_tickers_exchange()
        print("  ✓ Fetched ticker mapping")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        sys.exit(1)
    
    # Build ticker -> CIK mapping using the mapping utility
    from src.sec.mapping import build_ticker_to_cik_map
    
    ticker_map = build_ticker_to_cik_map(tickers_data)
    
    # Guardrail: Check if mapping is empty
    if len(ticker_map) == 0:
        print(f"\n  ✗ ERROR: No tickers found in SEC mapping!")
        print(f"  Response type: {type(tickers_data)}")
        if isinstance(tickers_data, dict):
            print(f"  Response keys: {list(tickers_data.keys())}")
            if "data" in tickers_data:
                print(f"  Data type: {type(tickers_data['data'])}")
                if isinstance(tickers_data["data"], list) and len(tickers_data["data"]) > 0:
                    print(f"  First data entry type: {type(tickers_data['data'][0])}")
                    print(f"  First data entry (first 200 chars): {str(tickers_data['data'][0])[:200]}")
        else:
            print(f"  Response (first 200 chars): {str(tickers_data)[:200]}")
        sys.exit(1)
    
    print(f"  Found {len(ticker_map)} tickers in SEC mapping")
    
    # Debug: Print first 5 tickers
    sample_tickers = list(ticker_map.keys())[:5]
    print(f"  Sample tickers: {', '.join(sample_tickers)}")
    
    # Seed issuers
    print("\n[2/2] Upserting issuers into dim_issuer...")
    engine = get_engine()
    
    missing_tickers = []
    seeded_count = 0
    
    with engine.begin() as conn:
        for issuer in ISSUER_UNIVERSE:
            ticker = issuer["ticker"]
            issuer_name = issuer["issuer_name"]
            bucket = issuer["bucket"]
            
            # Get CIK from SEC mapping
            ticker_info = ticker_map.get(ticker)
            
            if not ticker_info:
                missing_tickers.append(ticker)
                print(f"  ⚠ Warning: {ticker} not found in SEC mapping")
                # Still insert without CIK (can be updated later)
                cik = None
                sec_name = issuer_name
            else:
                cik = ticker_info["cik"]
                # Use SEC name if available, otherwise use config name
                sec_name = ticker_info["name"] if ticker_info["name"] else issuer_name
                
            conn.execute(
                text("""
                    INSERT INTO dim_issuer (ticker, cik, issuer_name, bucket)
                    VALUES (:ticker, :cik, :issuer_name, :bucket)
                    ON CONFLICT (ticker) DO UPDATE SET
                        cik = EXCLUDED.cik,
                        issuer_name = EXCLUDED.issuer_name,
                        bucket = EXCLUDED.bucket
                """),
                {
                    "ticker": ticker,
                    "cik": cik,
                    "issuer_name": sec_name,
                    "bucket": bucket,
                }
            )
            seeded_count += 1
            if cik:
                print(f"  ✓ {ticker}: CIK={cik}, Name={sec_name}")
            else:
                print(f"  ⚠ {ticker}: No CIK found, Name={sec_name}")
    
    # Verification query
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) as total, COUNT(cik) as with_cik FROM dim_issuer")
        )
        row = result.fetchone()
        total = row[0]
        with_cik = row[1]
        print(f"  Total issuers: {total}")
        print(f"  Issuers with CIK: {with_cik}")
        if with_cik == len(ISSUER_UNIVERSE):
            print(f"  ✅ All {len(ISSUER_UNIVERSE)} issuers have CIKs!")
        else:
            print(f"  ⚠ Expected {len(ISSUER_UNIVERSE)} issuers with CIK, found {with_cik}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Seeded: {seeded_count}/{len(ISSUER_UNIVERSE)} issuers")
    
    if missing_tickers:
        print(f"\n  ⚠ Missing from SEC mapping: {', '.join(missing_tickers)}")
        print("     These issuers were inserted without CIK and may need manual lookup.")
    
    # Print mapping table
    print("\n" + "="*80)
    print("ISSUER MAPPING TABLE")
    print("="*80)
    print(f"{'Ticker':<8} {'CIK':<12} {'Name':<50} {'Bucket':<15}")
    print("-" * 85)
    
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT ticker, cik, issuer_name, bucket FROM dim_issuer ORDER BY ticker")
        )
        for row in result:
            ticker, cik, name, bucket = row
            cik_str = cik if cik else "N/A"
            print(f"{ticker:<8} {cik_str:<12} {name[:48]:<50} {bucket:<15}")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)

