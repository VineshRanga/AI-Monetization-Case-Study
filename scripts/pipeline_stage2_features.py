#!/usr/bin/env python3
"""End-to-end pipeline for building feature store."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.db import get_engine
from sqlalchemy import text

if __name__ == "__main__":
    load_dotenv()
    
    print("="*80)
    print("PIPELINE STAGE 2: Building Feature Store")
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
    
    # Step 2: Ingest market data
    print("\n[2/6] Ingesting market data (FRED)...")
    try:
        from src.ingest.ingest_market import ingest_market_data
        
        result = ingest_market_data()
        print(f"  ✓ Ingested {result['total_rows']} market data rows")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Ingest equity prices
    print("\n[3/6] Ingesting equity prices...")
    try:
        from src.ingest.ingest_equity import ingest_equity_prices
        
        result = ingest_equity_prices()
        print(f"  ✓ Ingested {result['total_rows']} equity price rows")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Aggregate weekly market features
    print("\n[4/6] Aggregating weekly market features...")
    try:
        from src.transform.aggregate_weekly_features import aggregate_weekly_market
        
        result = aggregate_weekly_market()
        print(f"  ✓ Created {result['weeks_created']} weekly market records")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: Build issuer equity features
    print("\n[5/6] Building issuer equity features...")
    try:
        from src.transform.build_issuer_equity_features import build_issuer_equity_features
        
        result = build_issuer_equity_features()
        print(f"  ✓ Created {result['rows_created']} issuer-week feature rows")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 6: Build regime labels
    print("\n[6/6] Building regime labels...")
    try:
        from src.models.build_regime_labels import build_regime_labels
        
        result = build_regime_labels()
        print(f"  ✓ Labeled {result['weeks_labeled']} weeks ({result['risk_off_pct']:.1f}% RISK_OFF)")
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
        # Count issuer-week rows
        count = conn.execute(
            text("SELECT COUNT(*) FROM feat_issuer_weekly")
        ).fetchone()[0]
        print(f"\nIssuer-week rows in feat_issuer_weekly: {count}")
        
        # Join coverage with targets
        coverage = conn.execute(
            text("""
                SELECT COUNT(*) as matched
                FROM feat_issuer_weekly f
                INNER JOIN fact_issuer_spread_weekly t
                    ON f.issuer_id = t.issuer_id
                    AND f.week_start = t.week_start
            """)
        ).fetchone()[0]
        
        target_count = conn.execute(
            text("SELECT COUNT(*) FROM fact_issuer_spread_weekly")
        ).fetchone()[0]
        
        coverage_pct = (coverage / target_count * 100) if target_count > 0 else 0
        print(f"Join coverage: {coverage}/{target_count} ({coverage_pct:.1f}%)")
        
        # Sample joined table for 2 issuers
        print("\nSample joined table (last 10 weeks for 2 issuers):")
        print("-" * 80)
        
        issuers = conn.execute(
            text("""
                SELECT DISTINCT i.ticker
                FROM dim_issuer i
                JOIN feat_issuer_weekly f ON i.issuer_id = f.issuer_id
                LIMIT 2
            """)
        ).fetchall()
        
        for (ticker,) in issuers:
            samples = conn.execute(
                text("""
                    SELECT 
                        f.week_start,
                        f.eq_ret,
                        f.eq_vol_21d,
                        t.spread_bps,
                        t.dspread_bps,
                        r.regime_label
                    FROM feat_issuer_weekly f
                    JOIN fact_issuer_spread_weekly t
                        ON f.issuer_id = t.issuer_id
                        AND f.week_start = t.week_start
                    LEFT JOIN model_regime_weekly r
                        ON f.week_start = r.week_start
                    JOIN dim_issuer i ON f.issuer_id = i.issuer_id
                    WHERE i.ticker = :ticker
                    ORDER BY f.week_start DESC
                    LIMIT 10
                """),
                {"ticker": ticker}
            ).fetchall()
            
            if samples:
                print(f"\n  {ticker}:")
                for row in samples:
                    eq_ret_str = f"{row[1]:.3f}" if row[1] is not None else "N/A"
                    vol_str = f"{row[2]:.2f}" if row[2] is not None else "N/A"
                    spread_str = f"{row[3]:.1f}" if row[3] is not None else "N/A"
                    dspread_str = f"{row[4]:.1f}" if row[4] is not None else "N/A"
                    regime = row[5] if row[5] else "N/A"
                    print(f"    {row[0]} | eq_ret: {eq_ret_str:7} | vol: {vol_str:6} | spread: {spread_str:7} | Δspread: {dspread_str:7} | {regime}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)

