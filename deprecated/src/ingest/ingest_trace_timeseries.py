"""Ingest TRACE time series data for selected bonds."""
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Engine
from tqdm import tqdm

from src.db.db import create_etl_run, get_engine, now_utc, update_etl_run
from src.finra.auth import get_access_token
from src.finra.discovery import get_metadata
from src.finra.field_map import infer_field_map
from src.finra.trace_client import GROUP, log_raw_payload, request_with_retries

load_dotenv()


def load_field_map() -> Dict:
    """Load field map from saved file or re-infer."""
    field_map_path = Path(__file__).parent.parent.parent / "data" / "interim" / "trace_field_map.json"
    
    if field_map_path.exists():
        with open(field_map_path, "r") as f:
            field_map = json.load(f)
            # Also load all available fields from metadata for field discovery
            token = get_access_token()
            dataset_name = os.getenv("TRACE_DATASET_NAME")
            metadata = get_metadata(token, GROUP, dataset_name)
            # Extract all field names
            fields = metadata
            if isinstance(metadata, dict):
                if "fields" in metadata:
                    fields = metadata["fields"]
                elif "columns" in metadata:
                    fields = metadata["columns"]
            if isinstance(fields, list):
                field_map["_all_fields"] = [
                    f.get("name", f) if isinstance(f, dict) else f
                    for f in fields
                ]
            return field_map
    
    # Re-infer if not found
    token = get_access_token()
    dataset_name = os.getenv("TRACE_DATASET_NAME")
    metadata = get_metadata(token, GROUP, dataset_name)
    field_map = infer_field_map(metadata)
    # Add all fields
    fields = metadata
    if isinstance(metadata, dict):
        if "fields" in metadata:
            fields = metadata["fields"]
        elif "columns" in metadata:
            fields = metadata["columns"]
    if isinstance(fields, list):
        field_map["_all_fields"] = [
            f.get("name", f) if isinstance(f, dict) else f
            for f in fields
        ]
    return field_map


def get_selected_cusips(engine: Engine) -> List[str]:
    """Get list of selected CUSIPs."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT cusip FROM dim_bond WHERE is_selected = TRUE ORDER BY cusip")
        )
        return [row[0] for row in result]


def query_trace_monthly_slice(
    token: str,
    dataset_name: str,
    field_map: Dict,
    cusip: str,
    start_date: str,
    end_date: str,
    engine: Engine,
    run_id: int,
) -> List[Dict]:
    """Query TRACE for a single CUSIP in a monthly date slice."""
    cusip_field = field_map["cusip_field"]
    trade_date_field = field_map["trade_date_field"]
    
    # Build fields list
    fields = [cusip_field, trade_date_field]
    
    # Map optional fields
    price_field = None
    ytm_field = None
    volume_field = None
    trades_field = None
    
    # Try to find price/yield/volume fields
    all_fields = field_map.get("_all_fields", [])
    for field in all_fields:
        field_lower = field.lower()
        if "price" in field_lower and not price_field:
            price_field = field
        if ("yield" in field_lower or "ytm" in field_lower) and not ytm_field:
            ytm_field = field
        if "volume" in field_lower and not volume_field:
            volume_field = field
        if "trade" in field_lower and "count" in field_lower and not trades_field:
            trades_field = field
    
    # Add available fields
    if price_field:
        fields.append(price_field)
    if ytm_field:
        fields.append(ytm_field)
    if volume_field:
        fields.append(volume_field)
    if trades_field:
        fields.append(trades_field)
    
    payload = {
        "fields": fields,
        "filters": [
            {
                "fieldName": cusip_field,
                "compareType": "equals",
                "value": cusip,
            }
        ],
        "dateRangeFilters": [
            {
                "fieldName": trade_date_field,
                "startDate": start_date,
                "endDate": end_date,
            }
        ],
        "limit": 10000,
    }
    
    status, response = request_with_retries(token, dataset_name, payload)
    log_raw_payload(
        engine, run_id, "FINRA_TRACE",
        f"/data/group/{GROUP}/name/{dataset_name}",
        payload, status, response
    )
    
    if status == 200:
        return response.get("data", [])
    else:
        return []


def normalize_trace_data(data: List[Dict], field_map: Dict) -> pd.DataFrame:
    """Normalize TRACE data to fact_bond_daily format."""
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    if df.empty:
        return pd.DataFrame()
    
    # Map fields
    cusip_field = field_map["cusip_field"]
    trade_date_field = field_map["trade_date_field"]
    
    result = pd.DataFrame()
    result["cusip"] = df[cusip_field]
    result["trade_date"] = pd.to_datetime(df[trade_date_field]).dt.date
    
    # Map optional fields
    price_field = None
    ytm_field = None
    volume_field = None
    trades_field = None
    
    for col in df.columns:
        col_lower = col.lower()
        if "price" in col_lower and not price_field:
            price_field = col
        if ("yield" in col_lower or "ytm" in col_lower) and not ytm_field:
            ytm_field = col
        if "volume" in col_lower and not volume_field:
            volume_field = col
        if "trade" in col_lower and "count" in col_lower and not trades_field:
            trades_field = col
    
    if price_field and price_field in df.columns:
        result["price"] = pd.to_numeric(df[price_field], errors="coerce")
    else:
        result["price"] = None
    
    if ytm_field and ytm_field in df.columns:
        result["ytm"] = pd.to_numeric(df[ytm_field], errors="coerce")
    else:
        result["ytm"] = None
    
    if volume_field and volume_field in df.columns:
        result["volume"] = pd.to_numeric(df[volume_field], errors="coerce")
    else:
        result["volume"] = None
    
    if trades_field and trades_field in df.columns:
        result["trades"] = pd.to_numeric(df[trades_field], errors="coerce").astype("Int64")
    else:
        result["trades"] = None
    
    result["source"] = "FINRA_TRACE"
    
    return result


def upsert_bond_daily(engine: Engine, df: pd.DataFrame):
    """Upsert bond daily data."""
    if df.empty:
        return
    
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text("""
                    INSERT INTO fact_bond_daily (
                        cusip, trade_date, price, ytm, volume, trades, source
                    )
                    VALUES (
                        :cusip, :trade_date, :price, :ytm, :volume, :trades, :source
                    )
                    ON CONFLICT (cusip, trade_date) DO UPDATE SET
                        price = EXCLUDED.price,
                        ytm = EXCLUDED.ytm,
                        volume = EXCLUDED.volume,
                        trades = EXCLUDED.trades,
                        source = EXCLUDED.source
                """),
                {
                    "cusip": str(row["cusip"]),
                    "trade_date": row["trade_date"],
                    "price": float(row["price"]) if pd.notna(row["price"]) else None,
                    "ytm": float(row["ytm"]) if pd.notna(row["ytm"]) else None,
                    "volume": float(row["volume"]) if pd.notna(row["volume"]) else None,
                    "trades": int(row["trades"]) if pd.notna(row["trades"]) else None,
                    "source": "FINRA_TRACE",
                }
            )


def ingest_trace_timeseries(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    dataset_name: Optional[str] = None,
) -> Dict:
    """Ingest TRACE time series for selected bonds."""
    if dataset_name is None:
        dataset_name = os.getenv("TRACE_DATASET_NAME")
        if not dataset_name:
            raise ValueError("TRACE_DATASET_NAME must be set in environment")
    
    engine = get_engine()
    run_id = create_etl_run(engine, "trace_timeseries_ingest")
    
    try:
        token = get_access_token()
        field_map = load_field_map()
        cusips = get_selected_cusips(engine)
        
        print(f"Ingesting TRACE data for {len(cusips)} selected CUSIPs...")
        
        summaries = []
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        for cusip in tqdm(cusips, desc="CUSIPs"):
            all_data = []
            current = start
            
            while current <= end:
                month_end = min(current + timedelta(days=31), end)
                slice_start = current.strftime("%Y-%m-%d")
                slice_end = month_end.strftime("%Y-%m-%d")
                
                data = query_trace_monthly_slice(
                    token, dataset_name, field_map, cusip,
                    slice_start, slice_end, engine, run_id
                )
                all_data.extend(data)
                
                current = month_end + timedelta(days=1)
                time.sleep(0.3)  # Rate limiting
            
            if all_data:
                df = normalize_trace_data(all_data, field_map)
                if not df.empty:
                    upsert_bond_daily(engine, df)
                    
                    summaries.append({
                        "cusip": cusip,
                        "rows_inserted": len(df),
                        "date_min": df["trade_date"].min(),
                        "date_max": df["trade_date"].max(),
                    })
        
        # Save summary
        summary_df = pd.DataFrame(summaries)
        output_dir = Path(__file__).parent.parent.parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_dir / "trace_ingest_summary.csv", index=False)
        
        update_etl_run(engine, run_id, "OK", {"total_cusips": len(cusips), "successful": len(summaries)})
        
        return {"summaries": summaries}
    
    except Exception as e:
        update_etl_run(engine, run_id, "FAILED", {"error": str(e)})
        raise

