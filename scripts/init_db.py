#!/usr/bin/env python3
"""Initialize the database schema."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.db import run_sql_file

if __name__ == "__main__":
    load_dotenv()
    
    schema_path = Path(__file__).parent.parent / "src" / "db" / "schema.sql"
    print(f"Running schema from: {schema_path}")
    
    try:
        run_sql_file(str(schema_path))
        print("DB initialized OK")
    except Exception as e:
        print(f"Error initializing DB: {e}")
        sys.exit(1)

