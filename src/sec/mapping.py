"""Ticker to CIK mapping utilities."""
from typing import Dict, Optional


def zero_pad_cik(cik: str) -> str:
    """Zero-pad CIK to 10 digits."""
    if not cik:
        return ""
    return cik.zfill(10)


def build_ticker_to_cik_map(tickers_data: dict) -> Dict[str, Dict[str, str]]:
    """
    Build ticker -> CIK mapping from SEC company tickers exchange data.
    
    SEC returns JSON in format:
    {
      "fields": ["cik","name","ticker","exchange"],
      "data": [[320193,"Apple Inc.","AAPL","Nasdaq"], ...]
    }
    
    Args:
        tickers_data: JSON response from get_company_tickers_exchange()
    
    Returns:
        Dict mapping ticker (uppercase) -> {"cik": cik_str, "name": name, "exchange": exchange}
    """
    ticker_map = {}
    
    if not isinstance(tickers_data, dict):
        return ticker_map
    
    # Handle SEC format: {"fields": [...], "data": [[...], ...]}
    if "fields" in tickers_data and "data" in tickers_data:
        fields = tickers_data["fields"]
        data = tickers_data["data"]
        
        # Build index map
        field_idx = {}
        for i, field_name in enumerate(fields):
            field_idx[field_name.lower()] = i
        
        # Get indices for fields we need
        cik_idx = field_idx.get("cik", -1)
        name_idx = field_idx.get("name", -1)
        ticker_idx = field_idx.get("ticker", -1)
        exchange_idx = field_idx.get("exchange", -1)
        
        if cik_idx < 0 or ticker_idx < 0:
            return ticker_map
        
        # Parse data rows
        for row in data:
            if not isinstance(row, list) or len(row) <= max(cik_idx, ticker_idx):
                continue
            
            try:
                cik = row[cik_idx]
                ticker = row[ticker_idx]
                name = row[name_idx] if name_idx >= 0 and name_idx < len(row) else ""
                exchange = row[exchange_idx] if exchange_idx >= 0 and exchange_idx < len(row) else ""
                
                # Normalize ticker
                ticker = str(ticker).upper().strip() if ticker else ""
                cik_str = str(cik).zfill(10) if cik else ""
                
                if ticker and cik_str:
                    ticker_map[ticker] = {
                        "cik": cik_str,
                        "name": str(name) if name else "",
                        "exchange": str(exchange) if exchange else "",
                    }
            except (IndexError, TypeError, ValueError):
                continue
    
    # Fallback: Handle legacy format (list of dicts)
    elif "data" in tickers_data:
        for entry in tickers_data["data"]:
            if isinstance(entry, dict):
                ticker = entry.get("ticker", "").upper().strip()
                cik_str = str(entry.get("cik_str", entry.get("cik", ""))).zfill(10)
                name = entry.get("name", entry.get("title", ""))
                exchange = entry.get("exchange", "")
                
                if ticker and cik_str:
                    ticker_map[ticker] = {
                        "cik": cik_str,
                        "name": str(name) if name else "",
                        "exchange": str(exchange) if exchange else "",
                    }
    
    # Fallback: Handle dict keyed by CIK or ticker
    elif isinstance(tickers_data, dict):
        for key, value in tickers_data.items():
            if isinstance(value, dict):
                ticker = value.get("ticker", key if key.isalpha() else "").upper().strip()
                cik_str = str(value.get("cik_str", value.get("cik", key if key.isdigit() else ""))).zfill(10)
                name = value.get("name", value.get("title", ""))
                exchange = value.get("exchange", "")
                
                if ticker and cik_str:
                    ticker_map[ticker] = {
                        "cik": cik_str,
                        "name": str(name) if name else "",
                        "exchange": str(exchange) if exchange else "",
                    }
    
    return ticker_map


def get_cik_for_ticker(ticker: str, tickers_data: dict) -> Optional[str]:
    """Get CIK for a given ticker."""
    mapping = build_ticker_to_cik_map(tickers_data)
    ticker_info = mapping.get(ticker.upper())
    return ticker_info.get("cik") if ticker_info else None

