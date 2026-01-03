"""Ingest U.S. Treasury yield data from FRED API or CSV."""
import os
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
FRED_SERIES = {
    "2Y": "DGS2",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}


def ingest_ust_from_fred(engine: Engine, api_key: str, start_date: str = "2020-01-01", end_date: str = "2025-12-31") -> int:
    """
    Ingest UST yields from FRED API.
    
    Returns number of rows inserted.
    """
    all_data = []
    
    for tenor, series_id in FRED_SERIES.items():
        print(f"  Fetching {tenor} ({series_id})...")
        
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
            
            for obs in observations:
                date_str = obs.get("date")
                value_str = obs.get("value")
                
                if date_str and value_str and value_str != ".":
                    try:
                        yield_val = float(value_str)
                        all_data.append({
                            "date": pd.to_datetime(date_str).date(),
                            "tenor": tenor,
                            "yield": yield_val,
                            "source": "FRED",
                        })
                    except (ValueError, TypeError):
                        continue
        
        except Exception as e:
            print(f"    Warning: Failed to fetch {tenor}: {e}")
            continue
    
    # Upsert to database
    if all_data:
        df = pd.DataFrame(all_data)
        with engine.begin() as conn:
            for _, row in df.iterrows():
                conn.execute(
                    text("""
                        INSERT INTO fact_ust_yield_daily (date, tenor, yield, source)
                        VALUES (:date, :tenor, :yield, :source)
                        ON CONFLICT (date, tenor) DO UPDATE SET
                            yield = EXCLUDED.yield,
                            source = EXCLUDED.source
                    """),
                    {
                        "date": row["date"],
                        "tenor": row["tenor"],
                        "yield": float(row["yield"]),
                        "source": "FRED",
                    }
                )
        
        return len(df)
    
    return 0


def ingest_ust_from_csv(engine: Engine, csv_path: Path) -> int:
    """
    Ingest UST yields from CSV file.
    
    Expected columns: date, DGS2, DGS5, DGS7, DGS10, DGS20, DGS30
    
    Returns number of rows inserted.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Normalize date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    else:
        raise ValueError("CSV must have 'date' column")
    
    all_data = []
    
    for tenor, series_col in FRED_SERIES.items():
        if series_col in df.columns:
            for _, row in df.iterrows():
                value = row[series_col]
                if pd.notna(value) and value != ".":
                    try:
                        yield_val = float(value)
                        all_data.append({
                            "date": row["date"],
                            "tenor": tenor,
                            "yield": yield_val,
                            "source": "CSV",
                        })
                    except (ValueError, TypeError):
                        continue
    
    # Upsert to database
    if all_data:
        df_out = pd.DataFrame(all_data)
        with engine.begin() as conn:
            for _, row in df_out.iterrows():
                conn.execute(
                    text("""
                        INSERT INTO fact_ust_yield_daily (date, tenor, yield, source)
                        VALUES (:date, :tenor, :yield, :source)
                        ON CONFLICT (date, tenor) DO UPDATE SET
                            yield = EXCLUDED.yield,
                            source = EXCLUDED.source
                    """),
                    {
                        "date": row["date"],
                        "tenor": row["tenor"],
                        "yield": float(row["yield"]),
                        "source": "CSV",
                    }
                )
        
        return len(df_out)
    
    return 0


def ingest_ust_yields(start_date: str = "2020-01-01", end_date: str = "2025-12-31") -> Dict:
    """Ingest UST yields using FRED API (primary) or CSV (fallback)."""
    engine = get_engine()
    
    # Try FRED API first
    fred_api_key = os.getenv("FRED_API_KEY")
    
    if fred_api_key:
        print("Using FRED API...")
        try:
            rows = ingest_ust_from_fred(engine, fred_api_key, start_date, end_date)
            return {"method": "FRED", "rows_inserted": rows}
        except Exception as e:
            print(f"FRED API failed: {e}")
            print("Falling back to CSV...")
    
    # Fallback to CSV
    csv_path = Path(__file__).parent.parent.parent / "data" / "raw" / "ust_yields.csv"
    
    if csv_path.exists():
        print("Using CSV file...")
        rows = ingest_ust_from_csv(engine, csv_path)
        return {"method": "CSV", "rows_inserted": rows}
    else:
        raise ValueError(
            f"Neither FRED_API_KEY set nor CSV file found at {csv_path}. "
            "Please set FRED_API_KEY in .env or provide ust_yields.csv"
        )

