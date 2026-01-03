"""
Ingest market data from FRED API (VIX, rates, OAS).

FRED API Terms of Use Compliance:
- Required Notice: "This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis."
- Terms of Use: https://fred.stlouisfed.org/docs/api/terms_of_use.html
- Third-party data series may be copyrighted; users responsible for verification and permissions.
- No endorsement implied by Federal Reserve Bank of St. Louis.
- Rate limiting implemented to avoid unreasonable bandwidth usage.
"""
import os
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import requests
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db.db import get_engine

load_dotenv()

FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Rate limiting: max 1 request per second (conservative, well below FRED limits)
FRED_RATE_LIMIT_SECONDS = 1.0
_last_fred_request_time = 0.0

# FRED series mapping
FRED_SERIES = {
    "VIX": "VIXCLS",
    "DGS2": "DGS2",
    "DGS10": "DGS10",
    "IG_OAS": "BAMLC0A0CM",  # ICE BofA US Corporate OAS
    "HY_OAS": "BAMLH0A0HYM2",  # ICE BofA US High Yield OAS (optional)
}


def _rate_limit_fred():
    """Enforce rate limiting for FRED API requests."""
    global _last_fred_request_time
    now = time.time()
    elapsed = now - _last_fred_request_time
    if elapsed < FRED_RATE_LIMIT_SECONDS:
        time.sleep(FRED_RATE_LIMIT_SECONDS - elapsed)
    _last_fred_request_time = time.time()


def ingest_fred_series(
    engine: Engine,
    api_key: str,
    series_id: str,
    series_name: str,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
) -> int:
    """
    Ingest a single FRED series into fact_equity_price_daily (as daily macro data).
    
    Implements rate limiting to comply with FRED API Terms of Use.
    
    Returns number of rows inserted.
    """
    if not api_key:
        raise ValueError("FRED_API_KEY is required for API calls. Use CSV fallback if API key is not available.")
    
    # Rate limit: wait between requests
    _rate_limit_fred()
    
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }
    
    try:
        response = requests.get(FRED_API_BASE, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        observations = data.get("observations", [])
        
        rows_inserted = 0
        
        with engine.begin() as conn:
            for obs in observations:
                date_str = obs.get("date")
                value_str = obs.get("value")
                
                if date_str and value_str and value_str != ".":
                    try:
                        value = float(value_str)
                        date_obj = pd.to_datetime(date_str).date()
                        
                        # Store as equity price daily (using symbol for series name)
                        conn.execute(
                            text("""
                                INSERT INTO fact_equity_price_daily 
                                (symbol, date, close, adj_close, source)
                                VALUES (:symbol, :date, :close, :close, :source)
                                ON CONFLICT (symbol, date) DO UPDATE SET
                                    close = EXCLUDED.close,
                                    adj_close = EXCLUDED.adj_close
                            """),
                            {
                                "symbol": series_name,
                                "date": date_obj,
                                "close": value,
                                "source": "FRED",
                            }
                        )
                        rows_inserted += 1
                    except (ValueError, TypeError):
                        continue
        
        return rows_inserted
    
    except Exception as e:
        print(f"  Warning: Failed to fetch {series_name} ({series_id}): {e}")
        return 0


def ingest_market_data(start_date: str = "2020-01-01", end_date: str = "2025-12-31") -> Dict:
    """Ingest market data from FRED API or CSV fallback."""
    engine = get_engine()
    fred_api_key = os.getenv("FRED_API_KEY")
    
    # Try CSV fallback first if no API key
    if not fred_api_key:
        csv_path = Path(__file__).parent.parent.parent / "data" / "raw" / "fred" / "market_data.csv"
        if csv_path.exists():
            print("FRED_API_KEY not set, using CSV fallback...")
            rows = ingest_market_from_csv(engine, csv_path)
            return {
                "total_rows": rows,
                "method": "CSV",
                "series_results": {},
            }
        else:
            print("Warning: FRED_API_KEY not set and CSV fallback not found.")
            print(f"  Expected CSV at: {csv_path}")
            print("  Setting FRED_API_KEY in .env or placing CSV will enable ingestion.")
            return {
                "total_rows": 0,
                "method": "NONE",
                "series_results": {},
            }
    
    print("Ingesting market data from FRED API...")
    print("  Note: Rate limiting applied (1 request/second) per FRED API Terms of Use")
    
    total_rows = 0
    series_results = {}
    
    for series_name, series_id in FRED_SERIES.items():
        print(f"  Fetching {series_name} ({series_id})...")
        try:
            rows = ingest_fred_series(engine, fred_api_key, series_id, series_name, start_date, end_date)
            series_results[series_name] = rows
            total_rows += rows
            print(f"    Inserted {rows} rows")
        except ValueError as e:
            # API key missing
            print(f"    Error: {e}")
            break
        except Exception as e:
            print(f"    Warning: Failed to fetch {series_name}: {e}")
            series_results[series_name] = 0
    
    return {
        "total_rows": total_rows,
        "method": "FRED_API",
        "series_results": series_results,
    }


def ingest_market_from_csv(engine: Engine, csv_path: Path) -> int:
    """
    Fallback: Ingest market data from CSV.
    
    Expected columns: date, VIX, DGS2, DGS10, IG_OAS, HY_OAS
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if "date" not in df.columns:
        raise ValueError("CSV must have 'date' column")
    
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    rows_inserted = 0
    
    with engine.begin() as conn:
        for series_name in ["VIX", "DGS2", "DGS10", "IG_OAS", "HY_OAS"]:
            if series_name in df.columns:
                for _, row in df.iterrows():
                    value = row[series_name]
                    if pd.notna(value) and value != ".":
                        try:
                            value_float = float(value)
                            conn.execute(
                                text("""
                                    INSERT INTO fact_equity_price_daily 
                                    (symbol, date, close, adj_close, source)
                                    VALUES (:symbol, :date, :close, :close, :source)
                                    ON CONFLICT (symbol, date) DO UPDATE SET
                                        close = EXCLUDED.close,
                                        adj_close = EXCLUDED.adj_close
                                """),
                                {
                                    "symbol": series_name,
                                    "date": row["date"],
                                    "close": value_float,
                                    "source": "CSV",
                                }
                            )
                            rows_inserted += 1
                        except (ValueError, TypeError):
                            continue
    
    return rows_inserted

