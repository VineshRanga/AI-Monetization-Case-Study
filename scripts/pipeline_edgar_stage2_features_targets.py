#!/usr/bin/env python3
"""Master pipeline: Stage 2 - Features and targets."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))



if __name__ == "__main__":
    load_dotenv()
    
    print("="*80)
    print("PIPELINE STAGE 2: Features and Targets")
    print("="*80)
    
    # Step 0: Run migrations (ensure schema is up to date)
    print("\n[0/7] Running migrations...")
    try:
        from src.db.db import get_engine
        from pathlib import Path
        
        engine = get_engine()
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
                    error_str = str(e).lower()
                    # Gracefully handle migration 012: if y_dspread_bps column doesn't exist,
                    # that's OK (table might have been created from EDGAR-first migration 007)
                    if ("012" in migration_file.name and 
                        ("column" in error_str and "does not exist" in error_str and "y_dspread_bps" in error_str)):
                        print(f"    ⚠ {migration_file.name}: y_dspread_bps column doesn't exist (OK for EDGAR-first schema)")
                        continue
                    print(f"    ✗ {migration_file.name} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"\n  Migration {migration_file.name} failed. Stopping.")
                    sys.exit(1)
            
            print(f"  ✓ Completed {len(migration_files)} migration(s)")
    except Exception as e:
        print(f"  ✗ Error running migrations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 1: Aggregate weekly market
    print("\n[1/7] Aggregating weekly market features...")
    try:
        from src.transform.aggregate_weekly_features import aggregate_weekly_market
        result = aggregate_weekly_market()
        print(f"  ✓ Created {result.get('weeks_created', 0)} weekly market records")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Build issuer equity features
    print("\n[2/7] Building issuer equity features...")
    try:
        from src.transform.build_issuer_equity_features import build_issuer_equity_features
        result = build_issuer_equity_features()
        print(f"  ✓ Created {result.get('rows_created', 0)} issuer-week equity features")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Build fundamentals features
    print("\n[3/7] Building fundamentals lag features...")
    try:
        from src.transform.build_fundamentals_features import build_fundamentals_features
        result = build_fundamentals_features()
        print(f"  ✓ Updated {result.get('rows_updated', 0)} issuer-weeks with fundamentals")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Build credit proxy
    print("\n[4/7] Building credit proxy (Merton-style)...")
    try:
        from src.transform.build_credit_proxy import build_credit_proxy
        result = build_credit_proxy()
        print(f"  ✓ Created {result.get('rows_created', 0)} credit proxy records")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: Build regime labels
    print("\n[5/7] Building regime labels...")
    try:
        from src.models.build_regime_labels import build_regime_labels
        result = build_regime_labels()
        print(f"  ✓ Created {result.get('weeks_labeled', 0)} regime labels")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 6: Build model dataset
    print("\n[6/7] Building model dataset...")
    try:
        from src.transform.build_model_dataset import build_model_dataset
        result = build_model_dataset()
        print(f"  ✓ Created {result.get('rows_created', 0)} model dataset rows")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*80)
    print("PIPELINE STAGE 2 COMPLETE!")
    print("="*80)

