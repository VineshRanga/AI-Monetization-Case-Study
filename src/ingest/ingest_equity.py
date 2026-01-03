"""Ingest equity prices for ETFs and issuer stocks."""
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import requests
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.config.universe import ISSUER_UNIVERSE
from src.db.db import get_engine

load_dotenv()

# Stooq CSV download base URL
STOOQ_BASE = "https://stooq.com/q/d/l/?s={symbol}&i=d"

# User-Agent header (required by Stooq)
STOOQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def download_stooq_csv(symbol: str, start_date: str = "2020-01-01", end_date: str = "2025-12-31") -> Optional[pd.DataFrame]:
    """
    Download daily OHLCV data from Stooq.
    
    Stooq requires .US suffix for US stocks (e.g., AAPL.US).
    Tries symbol as-is first, then symbol.US as fallback.
    
    Returns DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    # Try symbol as-is first, then with .US suffix
    symbols_to_try = [symbol]
    
    # Add .US suffix if not already present
    # ETFs might work without suffix, but try both
    if not symbol.endswith(('.US', '.UK', '.DE', '.JP')):
        symbols_to_try.append(f"{symbol}.US")
    
    for try_symbol in symbols_to_try:
        try:
            url = STOOQ_BASE.format(symbol=try_symbol)
            
            # Stooq requires date range in URL parameters
            # Format: &d1=YYYYMMDD&d2=YYYYMMDD
            d1 = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
            d2 = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
            url = f"{url}&d1={d1}&d2={d2}"
            
            response = requests.get(url, headers=STOOQ_HEADERS, timeout=30)
            response.raise_for_status()
            
            # Check if we got data
            text = response.text.strip()
            if not text or text == "No data":
                continue
            
            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(text))
            
            if df.empty:
                continue
            
            # Standardize column names (case-insensitive)
            df.columns = df.columns.str.strip()
            
            # Map common column name variations
            col_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ['date', 'time']:
                    col_mapping[col] = "Date"
                elif col_lower in ['open', 'o']:
                    col_mapping[col] = "Open"
                elif col_lower in ['high', 'h']:
                    col_mapping[col] = "High"
                elif col_lower in ['low', 'l']:
                    col_mapping[col] = "Low"
                elif col_lower in ['close', 'c']:
                    col_mapping[col] = "Close"
                elif col_lower in ['adj close', 'adj_close', 'adjusted close']:
                    col_mapping[col] = "Adj Close"
                elif col_lower in ['volume', 'vol', 'v']:
                    col_mapping[col] = "Volume"
            
            df = df.rename(columns=col_mapping)
            
            # Ensure Date column exists and is datetime
            if "Date" not in df.columns:
                continue
            
            df["Date"] = pd.to_datetime(df["Date"])
            
            # Return on first successful download
            return df
        
        except Exception as e:
            # Try next symbol variant
            continue
    
    # If all attempts failed
    return None


def ingest_equity_symbol(engine: Engine, symbol: str, start_date: str = "2020-01-01", end_date: str = "2025-12-31") -> int:
    """Ingest equity data for a single symbol."""
    df = download_stooq_csv(symbol, start_date, end_date)
    
    if df is None or df.empty:
        return 0
    
    rows_inserted = 0
    
    with engine.begin() as conn:
        for _, row in df.iterrows():
            try:
                # Parse date
                date_obj = row["Date"]
                if pd.isna(date_obj):
                    continue
                
                if isinstance(date_obj, pd.Timestamp):
                    date_obj = date_obj.date()
                elif isinstance(date_obj, str):
                    date_obj = pd.to_datetime(date_obj).date()
                else:
                    date_obj = date_obj.date() if hasattr(date_obj, 'date') else None
                
                if date_obj is None:
                    continue
                
                # Get close price (required)
                close = row.get("Close")
                if pd.isna(close) or close is None:
                    continue
                
                # Get adj_close (fallback to close)
                adj_close = row.get("Adj Close")
                if pd.isna(adj_close) or adj_close is None:
                    adj_close = close
                
                # Get volume (optional)
                volume = row.get("Volume")
                if pd.isna(volume) or volume is None:
                    volume = None
                
                # Insert/update
                conn.execute(
                    text("""
                        INSERT INTO fact_equity_price_daily 
                        (symbol, date, close, adj_close, volume, source)
                        VALUES (:symbol, :date, :close, :adj_close, :volume, :source)
                        ON CONFLICT (symbol, date) DO UPDATE SET
                            close = EXCLUDED.close,
                            adj_close = EXCLUDED.adj_close,
                            volume = EXCLUDED.volume
                    """),
                    {
                        "symbol": symbol,
                        "date": date_obj,
                        "close": float(close),
                        "adj_close": float(adj_close),
                        "volume": float(volume) if volume is not None else None,
                        "source": "STOOQ",
                    }
                )
                rows_inserted += 1
            except (ValueError, TypeError, KeyError) as e:
                # Skip invalid rows
                continue
    
    return rows_inserted


def ingest_equity_prices(start_date: str = "2020-01-01", end_date: str = "2025-12-31") -> Dict:
    """Ingest equity prices for ETFs and issuer stocks."""
    engine = get_engine()
    
    # ETF symbols
    etf_symbols = ["SPY", "QQQ", "SMH", "SRVR"]
    
    # Issuer tickers
    issuer_tickers = [issuer["ticker"] for issuer in ISSUER_UNIVERSE]
    
    all_symbols = etf_symbols + issuer_tickers
    
    print(f"Ingesting equity prices for {len(all_symbols)} symbols...")
    
    results = {}
    total_rows = 0
    
    for symbol in all_symbols:
        print(f"  Downloading {symbol}...", end=" ", flush=True)
        rows = ingest_equity_symbol(engine, symbol, start_date, end_date)
        results[symbol] = rows
        total_rows += rows
        if rows > 0:
            print(f"✓ Inserted {rows} rows")
        else:
            print(f"✗ No data found")
    
    return {
        "total_rows": total_rows,
        "symbol_results": results,
    }

