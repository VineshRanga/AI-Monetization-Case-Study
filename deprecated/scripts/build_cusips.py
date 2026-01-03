#!/usr/bin/env python3
"""End-to-end script to build CUSIP universe."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.db import get_engine
from src.ingest.build_cusip_universe import build_cusip_universe
from src.ingest.output_manifests import write_manifests
from sqlalchemy import text

if __name__ == "__main__":
    load_dotenv()
    
    print("="*60)
    print("Building CUSIP Universe")
    print("="*60)
    
    # Ensure DB schema initialized (user should run init_db.py first)
    engine = get_engine()
    
    # Run migrations
    print("\n1. Running migrations...")
    try:
        migrations_dir = Path(__file__).parent.parent / "src" / "db" / "migrations"
        if migrations_dir.exists():
            migration_files = sorted(migrations_dir.glob("*.sql"))
            for migration_file in migration_files:
                print(f"  Running {migration_file.name}...")
                with open(migration_file, "r") as f:
                    sql_content = f.read()
                with engine.begin() as conn:
                    for statement in sql_content.split(";"):
                        statement = statement.strip()
                        if statement:
                            conn.execute(text(statement))
            print(f"  Completed {len(migration_files)} migration(s)")
    except Exception as e:
        print(f"  Warning: Migration error (may be OK if already run): {e}")
    
    # Seed issuers
    print("\n2. Seeding issuers...")
    try:
        from src.config.universe import ISSUER_UNIVERSE as config_universe
        
        with engine.begin() as conn:
            for issuer in config_universe:
                ticker = issuer["ticker"]
                issuer_name = issuer["issuer_name"]
                bucket = issuer["bucket"]
                
                # Upsert
                conn.execute(
                    text("""
                        INSERT INTO dim_issuer (ticker, issuer_name, bucket)
                        VALUES (:ticker, :issuer_name, :bucket)
                        ON CONFLICT (ticker) DO UPDATE SET
                            issuer_name = EXCLUDED.issuer_name,
                            bucket = EXCLUDED.bucket
                    """),
                    {
                        "ticker": ticker,
                        "issuer_name": issuer_name,
                        "bucket": bucket
                    }
                )
        print(f"  Seeded {len(config_universe)} issuers")
    except Exception as e:
        print(f"  Error seeding issuers: {e}")
        sys.exit(1)
    
    # Build CUSIP universe
    print("\n3. Building CUSIP universe...")
    try:
        result = build_cusip_universe()
        print("\n" + "="*60)
        print("CUSIP Universe Build Summary")
        print("="*60)
        print(f"Successful issuers: {len(result['successful'])}")
        for ticker, name, count in result["successful"]:
            print(f"  {ticker}: {count} CUSIPs")
        if result["failed"]:
            print(f"\nFailed issuers: {len(result['failed'])}")
            for item in result["failed"]:
                print(f"  {item}")
    except Exception as e:
        print(f"  Error building universe: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Write manifests
    print("\n4. Writing CSV manifests...")
    try:
        data_dir = Path(__file__).parent.parent / "data" / "processed"
        write_manifests(engine, data_dir)
    except Exception as e:
        print(f"  Error writing manifests: {e}")
        import traceback
        traceback.print_exc()
    
    # Validation queries
    print("\n5. Validation queries...")
    with engine.connect() as conn:
        # CUSIP count per issuer
        result = conn.execute(
            text("""
                SELECT 
                    i.ticker,
                    i.issuer_name,
                    COUNT(b.cusip) as num_cusips
                FROM dim_issuer i
                LEFT JOIN dim_bond b ON i.issuer_id = b.issuer_id
                GROUP BY i.ticker, i.issuer_name
                ORDER BY num_cusips DESC, i.ticker
            """)
        )
        
        print("\nCUSIP count per issuer:")
        print("-" * 60)
        for row in result:
            print(f"  {row[0]:6} {row[1]:40} {row[2]:4} CUSIPs")
        
        # Sample bonds per issuer
        print("\nSample bonds (3 per issuer, sorted by last_seen_trade_date DESC):")
        print("-" * 60)
        for issuer_row in conn.execute(
            text("SELECT ticker FROM dim_issuer ORDER BY ticker")
        ):
            ticker = issuer_row[0]
            bonds = conn.execute(
                text("""
                    SELECT 
                        b.cusip,
                        b.issuer_name,
                        b.maturity_date,
                        b.last_seen_trade_date,
                        b.is_active
                    FROM dim_bond b
                    JOIN dim_issuer i ON b.issuer_id = i.issuer_id
                    WHERE i.ticker = :ticker
                    ORDER BY b.last_seen_trade_date DESC NULLS LAST
                    LIMIT 3
                """),
                {"ticker": ticker}
            ).fetchall()
            
            if bonds:
                print(f"\n  {ticker}:")
                for bond in bonds:
                    print(f"    CUSIP: {bond[0]}, Maturity: {bond[2]}, Last Trade: {bond[3]}, Active: {bond[4]}")
            else:
                print(f"\n  {ticker}: No bonds found")
    
    print("\n" + "="*60)
    print("Build complete!")
    print("="*60)

