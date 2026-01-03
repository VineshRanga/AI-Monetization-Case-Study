#!/usr/bin/env python3
"""Master pipeline: Stage 1 - EDGAR ingestion."""
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.db import get_engine

if __name__ == "__main__":
    load_dotenv()
    
    print("="*80)
    print("PIPELINE STAGE 1: EDGAR Ingestion")
    print("="*80)
    
    engine = get_engine()
    
    # Step 1: Run migrations
    print("\n[1/6] Running migrations...")
    try:
        migrations_dir = Path(__file__).parent.parent / "src" / "db" / "migrations"
        if migrations_dir.exists():
            migration_files = sorted(migrations_dir.glob("*.sql"))
            for migration_file in migration_files:
                print(f"  Running {migration_file.name}...")
                try:
                    # Each migration file runs in its own transaction
                    # Execute entire file as one script (handles DO blocks, dollar-quoted strings, etc.)
                    with open(migration_file, "r") as f:
                        sql_content = f.read()
                    
                    with engine.begin() as conn:
                        conn.exec_driver_sql(sql_content)
                    
                    print(f"    ✓ {migration_file.name} completed")
                except Exception as e:
                    print(f"    ✗ {migration_file.name} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"\n  Migration {migration_file.name} failed. Stopping.")
                    sys.exit(1)
            print(f"  ✓ Completed {len(migration_files)} migration(s)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Seed issuers from EDGAR
    print("\n[2/6] Seeding issuers from SEC EDGAR...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "seed_issuers_edgar.py")],
            check=True,
            capture_output=False
        )
        print("  ✓ Issuers seeded")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Ingest SEC submissions
    print("\n[3/6] Ingesting SEC submissions...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "ingest_sec_submissions.py")],
            check=True,
            capture_output=False
        )
        print("  ✓ Submissions ingested")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Ingest company facts
    print("\n[4/6] Ingesting SEC company facts...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "ingest_companyfacts.py")],
            check=True,
            capture_output=False
        )
        print("  ✓ Company facts ingested")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: Ingest equity prices
    print("\n[5/6] Ingesting equity prices...")
    try:
        from src.ingest.ingest_equity import ingest_equity_prices
        result = ingest_equity_prices()
        print(f"  ✓ Equity prices ingested: {result.get('total_rows', 0)} rows")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 6: Ingest FRED macro data
    print("\n[6/6] Ingesting FRED macro data...")
    try:
        from src.ingest.ingest_market import ingest_market_data
        result = ingest_market_data()
        print(f"  ✓ FRED data ingested: {result.get('total_rows', 0)} rows via {result.get('method', 'unknown')}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Validation
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    with engine.connect() as conn:
        # Issuers
        result = conn.execute(text("SELECT COUNT(*) FROM dim_issuer"))
        issuer_count = result.fetchone()[0]
        print(f"  Issuers: {issuer_count}")
        
        # Fundamentals
        result = conn.execute(text("SELECT COUNT(*) FROM fact_fundamentals_quarterly"))
        fund_count = result.fetchone()[0]
        print(f"  Fundamentals quarters: {fund_count}")
        
        # Equity prices
        result = conn.execute(text("SELECT COUNT(DISTINCT symbol) FROM fact_equity_price_daily"))
        equity_symbols = result.fetchone()[0]
        print(f"  Equity symbols: {equity_symbols}")
    
    print("\n" + "="*80)
    print("PIPELINE STAGE 1 COMPLETE!")
    print("="*80)

