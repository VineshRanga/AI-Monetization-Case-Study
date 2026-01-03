#!/usr/bin/env python3
"""Ingest equity prices from Stooq."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.ingest_equity import ingest_equity_prices
from src.db.db import get_engine
from sqlalchemy import text

if __name__ == "__main__":
    load_dotenv()
    
    print("="*80)
    print("Ingesting Equity Prices from Stooq")
    print("="*80)
    
    result = ingest_equity_prices()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Total rows inserted: {result['total_rows']}")
    
    # Validation query
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    engine = get_engine()
    with engine.connect() as conn:
        result_query = conn.execute(
            text("""
                SELECT symbol, COUNT(*) as rows
                FROM fact_equity_price_daily
                GROUP BY symbol
                ORDER BY symbol
            """)
        )
        print(f"{'Symbol':<10} {'Rows':<10}")
        print("-" * 20)
        for row in result_query:
            symbol, count = row
            print(f"{symbol:<10} {count:<10}")
        
        total = conn.execute(text("SELECT COUNT(*) FROM fact_equity_price_daily")).fetchone()[0]
        print(f"\n  Total rows in fact_equity_price_daily: {total}")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
