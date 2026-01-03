#!/usr/bin/env python3
"""Seed the issuer universe into dim_issuer."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.universe import ISSUER_UNIVERSE
from src.db.db import get_engine, now_utc
from sqlalchemy import text

if __name__ == "__main__":
    load_dotenv()
    
    engine = get_engine()
    inserted = 0
    updated = 0
    
    with engine.begin() as conn:
        for issuer in ISSUER_UNIVERSE:
            ticker = issuer["ticker"]
            issuer_name = issuer["issuer_name"]
            bucket = issuer["bucket"]
            
            # Check if exists
            check = conn.execute(
                text("SELECT issuer_id FROM dim_issuer WHERE ticker = :ticker"),
                {"ticker": ticker}
            ).fetchone()
            
            if check:
                # Update
                conn.execute(
                    text("""
                        UPDATE dim_issuer
                        SET issuer_name = :issuer_name, bucket = :bucket
                        WHERE ticker = :ticker
                    """),
                    {
                        "ticker": ticker,
                        "issuer_name": issuer_name,
                        "bucket": bucket
                    }
                )
                updated += 1
            else:
                # Insert
                conn.execute(
                    text("""
                        INSERT INTO dim_issuer (ticker, issuer_name, bucket)
                        VALUES (:ticker, :issuer_name, :bucket)
                    """),
                    {
                        "ticker": ticker,
                        "issuer_name": issuer_name,
                        "bucket": bucket
                    }
                )
                inserted += 1
    
    print(f"Issuers seeded: {inserted} inserted, {updated} updated")
    print(f"Total issuers: {len(ISSUER_UNIVERSE)}")

