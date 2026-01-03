#!/usr/bin/env python3
"""Run database migrations in order."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.db import get_engine, run_sql_file
from sqlalchemy import text

if __name__ == "__main__":
    load_dotenv()
    
    migrations_dir = Path(__file__).parent.parent / "src" / "db" / "migrations"
    
    if not migrations_dir.exists():
        print(f"Migrations directory not found: {migrations_dir}")
        sys.exit(1)
    
    # Get all SQL files and sort by filename
    migration_files = sorted(migrations_dir.glob("*.sql"))
    
    if not migration_files:
        print("No migration files found")
        sys.exit(0)
    
    print(f"Found {len(migration_files)} migration(s)")
    
    engine = get_engine()
    
    for migration_file in migration_files:
        print(f"Running migration: {migration_file.name}")
        try:
            # Each migration file runs in its own transaction
            # Execute entire file as one script (handles DO blocks, dollar-quoted strings, etc.)
            with open(migration_file, "r") as f:
                sql_content = f.read()
            
            with engine.begin() as conn:
                conn.exec_driver_sql(sql_content)
            
            print(f"  ✓ {migration_file.name} completed")
        except Exception as e:
            print(f"  ✗ {migration_file.name} failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nMigration {migration_file.name} failed. Stopping.")
            sys.exit(1)
    
    print("All migrations completed successfully")

