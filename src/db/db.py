"""Database utilities for AICreditRiskAnalysis."""
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Load environment variables
load_dotenv()


def get_engine() -> Engine:
    """
    Build a Postgres connection engine from environment variables.
    
    Returns:
        SQLAlchemy Engine instance
    """
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "ai_credit_crisis")
    user = os.getenv("POSTGRES_USER", "")
    password = os.getenv("POSTGRES_PASSWORD", "")
    
    if not user:
        raise ValueError("POSTGRES_USER must be set in environment")
    
    if password:
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    else:
        connection_string = f"postgresql://{user}@{host}:{port}/{db}"
    
    return create_engine(connection_string, echo=False)


def parse_sql_statements(sql_content: str) -> list:
    """
    Parse SQL content into statements, handling DO blocks properly.
    
    DO blocks contain semicolons, so we can't just split by ';'.
    This function detects DO $$ ... END $$; blocks and treats them as single statements.
    
    Args:
        sql_content: Raw SQL file content
    
    Returns:
        List of SQL statements (each as a string)
    """
    import re
    statements = []
    lines = sql_content.split('\n')
    current_statement = []
    in_do_block = False
    do_delimiter = None
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines and comments (but preserve them in DO blocks)
        if not stripped:
            if current_statement and in_do_block:
                current_statement.append(line)
            continue
        
        if stripped.startswith('--'):
            if current_statement and in_do_block:
                current_statement.append(line)
            continue
        
        # Check for DO block start
        do_match = re.search(r'DO\s+(\$\$|\$\w+\$)', stripped, re.IGNORECASE)
        if do_match:
            in_do_block = True
            do_delimiter = do_match.group(1)
            current_statement = [line]
            continue
        
        # Check for DO block end
        if in_do_block:
            current_statement.append(line)
            # Check for END delimiter;
            end_pattern = rf'END\s+{re.escape(do_delimiter)}\s*;'
            if re.search(end_pattern, stripped, re.IGNORECASE):
                statements.append('\n'.join(current_statement))
                current_statement = []
                in_do_block = False
                do_delimiter = None
            continue
        
        # Regular statement - check for semicolon
        if ';' in stripped:
            # Split by semicolon
            parts = stripped.split(';')
            for i, part in enumerate(parts):
                if part.strip():
                    current_statement.append(part.strip())
                # If this is not the last part, we have a complete statement
                if i < len(parts) - 1:
                    if current_statement:
                        statements.append('\n'.join(current_statement))
                        current_statement = []
        else:
            current_statement.append(line)
    
    # Add any remaining statement
    if current_statement:
        statements.append('\n'.join(current_statement))
    
    return statements


def run_sql_file(sql_file_path: str) -> None:
    """
    Execute a SQL file against the database.
    
    Executes the entire file as one script (not split by semicolons).
    Each file runs in its own transaction.
    
    Args:
        sql_file_path: Path to the SQL file to execute
    
    Raises:
        FileNotFoundError: If SQL file doesn't exist
        Exception: If SQL execution fails (transaction is rolled back automatically)
    """
    engine = get_engine()
    sql_path = Path(sql_file_path)
    
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_file_path}")
    
    with open(sql_path, "r") as f:
        sql_content = f.read()
    
    # Execute entire file as one script in a single transaction
    # This handles DO blocks, dollar-quoted strings, and multi-statement scripts correctly
    with engine.begin() as conn:
        conn.exec_driver_sql(sql_content)


def upsert_df(
    df: pd.DataFrame,
    table_name: str,
    pk_cols: list[str],
    engine: Optional[Engine] = None,
) -> None:
    """
    Insert or update DataFrame rows into a table.
    
    On conflict with primary key, updates existing rows.
    
    Args:
        df: DataFrame to upsert
        table_name: Target table name
        pk_cols: List of primary key column names
        engine: Optional SQLAlchemy engine (uses get_engine() if not provided)
    """
    if df.empty:
        return
    
    if engine is None:
        engine = get_engine()
    
    # Create temp table
    temp_table = f"{table_name}_temp_{datetime.now(timezone.utc).timestamp()}"
    df.to_sql(temp_table, engine, if_exists="replace", index=False)
    
    try:
        # Build upsert SQL
        all_cols = list(df.columns)
        update_cols = [col for col in all_cols if col not in pk_cols]
        
        set_clause = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_cols])
        pk_clause = " AND ".join([f"t.{col} = EXCLUDED.{col}" for col in pk_cols])
        
        upsert_sql = f"""
        INSERT INTO {table_name} ({', '.join(all_cols)})
        SELECT {', '.join(all_cols)} FROM {temp_table}
        ON CONFLICT ({', '.join(pk_cols)})
        DO UPDATE SET {set_clause}
        """
        
        with engine.begin() as conn:
            conn.execute(text(upsert_sql))
    finally:
        # Drop temp table
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))


def now_utc() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)

