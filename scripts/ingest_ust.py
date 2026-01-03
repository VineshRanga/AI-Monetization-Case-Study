#!/usr/bin/env python3
"""Ingest U.S. Treasury yield data."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.db import get_engine
from src.ingest.ingest_ust import ingest_ust_yields
from sqlalchemy import text

if __name__ == "__main__":
    load_dotenv()
    
    print("="*60)
    print("Ingesting U.S. Treasury Yields")
    print("="*60)
    
    try:
        result = ingest_ust_yields()
        
        print(f"\nIngest method: {result['method']}")
        print(f"Rows inserted: {result['rows_inserted']}")
        
        # Show summary
        engine = get_engine()
        with engine.connect() as conn:
            result_query = conn.execute(
                text("""
                    SELECT 
                        tenor,
                        COUNT(*) as count,
                        MIN(date) as min_date,
                        MAX(date) as max_date,
                        AVG(yield) as avg_yield
                    FROM fact_ust_yield_daily
                    GROUP BY tenor
                    ORDER BY tenor
                """)
            )
            
            print("\nUST Yield Summary:")
            print("-" * 60)
            for row in result_query:
                print(f"  {row[0]:4} {row[1]:5} rows, {row[2]} to {row[3]}, avg yield: {row[4]:.2f}%")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

