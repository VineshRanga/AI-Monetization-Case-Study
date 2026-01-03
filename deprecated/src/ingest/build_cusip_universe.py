"""Build CUSIP universe from FINRA TRACE data."""
import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Engine
from tqdm import tqdm

from src.db.db import get_engine, now_utc
from src.finra.auth import get_access_token
from src.finra.discovery import get_metadata, has_cusip_field
from src.finra.field_map import infer_field_map
from src.finra.trace_client import (
    GROUP,
    log_raw_payload,
    request_with_retries,
)

load_dotenv()


def create_etl_run(engine: Engine, pipeline_stage: str) -> int:
    """Create an ETL run record and return run_id."""
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO etl_run (started_at, pipeline_stage, status)
                VALUES (:started_at, :pipeline_stage, 'RUNNING')
                RETURNING run_id
            """),
            {
                "started_at": now_utc(),
                "pipeline_stage": pipeline_stage,
            }
        )
        run_id = result.fetchone()[0]
    return run_id


def update_etl_run(engine: Engine, run_id: int, status: str, meta: Optional[Dict] = None):
    """Update ETL run status and metadata."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE etl_run
                SET finished_at = :finished_at, status = :status, meta = :meta
                WHERE run_id = :run_id
            """),
            {
                "run_id": run_id,
                "finished_at": now_utc(),
                "status": status,
                "meta": json.dumps(meta) if meta else None,
            }
        )


def get_issuers(engine: Engine) -> List[Dict]:
    """Get all issuers from dim_issuer."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT issuer_id, ticker, issuer_name, finra_issuer_key FROM dim_issuer")
        )
        return [
            {
                "issuer_id": row[0],
                "ticker": row[1],
                "issuer_name": row[2],
                "finra_issuer_key": row[3],
            }
            for row in result
        ]


def update_issuer_finra_key(engine: Engine, issuer_id: int, finra_key: str):
    """Update finra_issuer_key for an issuer."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE dim_issuer
                SET finra_issuer_key = :finra_key
                WHERE issuer_id = :issuer_id
            """),
            {"issuer_id": issuer_id, "finra_key": finra_key}
        )


def determine_issuer_filter(
    token: str,
    dataset_name: str,
    field_map: Dict,
    issuer_name: str,
    engine: Engine,
    run_id: int,
) -> str:
    """
    Determine the correct issuer filter value for FINRA queries.
    
    Returns the issuer filter string to use.
    """
    issuer_field = field_map["issuer_field"]
    trade_date_field = field_map["trade_date_field"]
    
    # First try: exact match
    payload = {
        "fields": [issuer_field, trade_date_field],
        "filters": [
            {
                "fieldName": issuer_field,
                "compareType": "equals",
                "value": issuer_name,
            }
        ],
        "dateRangeFilters": [
            {
                "fieldName": trade_date_field,
                "startDate": "2020-01-01",
                "endDate": "2020-01-31",  # Small date slice for testing
            }
        ],
        "limit": 10,
    }
    
    status, response = request_with_retries(token, dataset_name, payload)
    log_raw_payload(engine, run_id, "FINRA_TRACE", f"/data/group/{GROUP}/name/{dataset_name}", payload, status, response)
    
    if status == 200:
        # Check if we got results
        data = response.get("data", [])
        if data and len(data) > 0:
            # Found exact match
            return issuer_name
    
    # Fallback: query without issuer filter and inspect values
    payload_no_filter = {
        "fields": [issuer_field, trade_date_field],
        "dateRangeFilters": [
            {
                "fieldName": trade_date_field,
                "startDate": "2020-01-01",
                "endDate": "2020-03-31",
            }
        ],
        "limit": 1000,
    }
    
    status, response = request_with_retries(token, dataset_name, payload_no_filter)
    log_raw_payload(engine, run_id, "FINRA_TRACE", f"/data/group/{GROUP}/name/{dataset_name}", payload_no_filter, status, response)
    
    if status == 200:
        data = response.get("data", [])
        if data:
            # Find best match
            issuer_values = set()
            for row in data:
                if isinstance(row, dict) and issuer_field in row:
                    issuer_values.add(str(row[issuer_field]))
            
            # Try contains match
            for value in issuer_values:
                if issuer_name.lower() in value.lower() or value.lower() in issuer_name.lower():
                    return value
            
            # If no match, return the issuer_name as-is (will be used in query)
            print(f"  Warning: No exact match found for '{issuer_name}', using as-is")
            return issuer_name
    
    # Default: use issuer_name
    return issuer_name


def query_monthly_slice(
    token: str,
    dataset_name: str,
    field_map: Dict,
    issuer_filter: str,
    start_date: str,
    end_date: str,
    engine: Engine,
    run_id: int,
) -> List[Dict]:
    """Query a monthly date slice and return data."""
    issuer_field = field_map["issuer_field"]
    trade_date_field = field_map["trade_date_field"]
    cusip_field = field_map["cusip_field"]
    
    # Build fields list
    fields = [cusip_field, trade_date_field]
    if field_map.get("maturity_field"):
        fields.append(field_map["maturity_field"])
    if field_map.get("coupon_field"):
        fields.append(field_map["coupon_field"])
    if field_map.get("issue_date_field"):
        fields.append(field_map["issue_date_field"])
    if field_map.get("security_type_field"):
        fields.append(field_map["security_type_field"])
    
    payload = {
        "fields": fields,
        "filters": [
            {
                "fieldName": issuer_field,
                "compareType": "equals",
                "value": issuer_filter,
            }
        ],
        "dateRangeFilters": [
            {
                "fieldName": trade_date_field,
                "startDate": start_date,
                "endDate": end_date,
            }
        ],
        "limit": 10000,  # Adjust if needed
    }
    
    status, response = request_with_retries(token, dataset_name, payload)
    log_raw_payload(engine, run_id, "FINRA_TRACE", f"/data/group/{GROUP}/name/{dataset_name}", payload, status, response)
    
    if status == 200:
        return response.get("data", [])
    else:
        print(f"  Warning: Query failed with status {status}")
        return []


def aggregate_cusips(data: List[Dict], field_map: Dict) -> pd.DataFrame:
    """Aggregate data to get distinct CUSIPs with metadata."""
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    if df.empty:
        return df
    
    # Map field names
    cusip_field = field_map["cusip_field"]
    trade_date_field = field_map["trade_date_field"]
    
    # Group by CUSIP
    agg_dict = {
        trade_date_field: ["min", "max"],
    }
    
    # Add optional fields
    if field_map.get("maturity_field"):
        agg_dict[field_map["maturity_field"]] = lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else None
    if field_map.get("coupon_field"):
        agg_dict[field_map["coupon_field"]] = lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else None
    if field_map.get("issue_date_field"):
        agg_dict[field_map["issue_date_field"]] = lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else None
    if field_map.get("security_type_field"):
        agg_dict[field_map["security_type_field"]] = lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else None
    
    grouped = df.groupby(cusip_field).agg(agg_dict)
    
    # Flatten column names
    grouped.columns = ["_".join(col).strip() if col[1] else col[0] for col in grouped.columns]
    
    # Rename columns
    result = grouped.reset_index()
    result = result.rename(columns={
        cusip_field: "cusip",
        f"{trade_date_field}_min": "first_seen_trade_date",
        f"{trade_date_field}_max": "last_seen_trade_date",
    })
    
    # Rename optional fields
    if field_map.get("maturity_field"):
        old_col = field_map["maturity_field"]
        if f"{old_col}_<lambda>" in result.columns:
            result = result.rename(columns={f"{old_col}_<lambda>": "maturity_date"})
        elif old_col in result.columns:
            result = result.rename(columns={old_col: "maturity_date"})
    
    if field_map.get("coupon_field"):
        old_col = field_map["coupon_field"]
        if f"{old_col}_<lambda>" in result.columns:
            result = result.rename(columns={f"{old_col}_<lambda>": "coupon"})
        elif old_col in result.columns:
            result = result.rename(columns={old_col: "coupon"})
    
    if field_map.get("issue_date_field"):
        old_col = field_map["issue_date_field"]
        if f"{old_col}_<lambda>" in result.columns:
            result = result.rename(columns={f"{old_col}_<lambda>": "issue_date"})
        elif old_col in result.columns:
            result = result.rename(columns={old_col: "issue_date"})
    
    if field_map.get("security_type_field"):
        old_col = field_map["security_type_field"]
        if f"{old_col}_<lambda>" in result.columns:
            result = result.rename(columns={f"{old_col}_<lambda>": "security_type"})
        elif old_col in result.columns:
            result = result.rename(columns={old_col: "security_type"})
    
    return result


def upsert_bonds(engine: Engine, bonds_df: pd.DataFrame, issuer_id: int, issuer_name: str, end_date: str):
    """Upsert bonds into dim_bond."""
    if bonds_df.empty:
        return
    
    with engine.begin() as conn:
        for _, row in bonds_df.iterrows():
            cusip = str(row["cusip"])
            first_seen = row.get("first_seen_trade_date")
            last_seen = row.get("last_seen_trade_date")
            maturity = row.get("maturity_date")
            coupon = row.get("coupon")
            issue_date = row.get("issue_date")
            security_type = row.get("security_type")
            
            # Determine is_active
            is_active = None
            if maturity:
                try:
                    if isinstance(maturity, str):
                        maturity_date = datetime.strptime(maturity[:10], "%Y-%m-%d").date()
                    else:
                        maturity_date = pd.to_datetime(maturity).date()
                    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
                    is_active = maturity_date >= end_date_obj
                except Exception:
                    pass
            
            # Upsert
            conn.execute(
                text("""
                    INSERT INTO dim_bond (
                        cusip, issuer_id, issuer_name, coupon, maturity_date,
                        issue_date, security_type, first_seen_trade_date,
                        last_seen_trade_date, is_active, updated_at
                    )
                    VALUES (
                        :cusip, :issuer_id, :issuer_name, :coupon, :maturity_date,
                        :issue_date, :security_type, :first_seen_trade_date,
                        :last_seen_trade_date, :is_active, :updated_at
                    )
                    ON CONFLICT (cusip) DO UPDATE SET
                        issuer_id = EXCLUDED.issuer_id,
                        issuer_name = EXCLUDED.issuer_name,
                        coupon = EXCLUDED.coupon,
                        maturity_date = EXCLUDED.maturity_date,
                        issue_date = EXCLUDED.issue_date,
                        security_type = EXCLUDED.security_type,
                        first_seen_trade_date = EXCLUDED.first_seen_trade_date,
                        last_seen_trade_date = EXCLUDED.last_seen_trade_date,
                        is_active = EXCLUDED.is_active,
                        updated_at = EXCLUDED.updated_at
                """),
                {
                    "cusip": cusip,
                    "issuer_id": issuer_id,
                    "issuer_name": issuer_name,
                    "coupon": float(coupon) if pd.notna(coupon) else None,
                    "maturity_date": maturity if pd.notna(maturity) else None,
                    "issue_date": issue_date if pd.notna(issue_date) else None,
                    "security_type": str(security_type) if pd.notna(security_type) else None,
                    "first_seen_trade_date": first_seen if pd.notna(first_seen) else None,
                    "last_seen_trade_date": last_seen if pd.notna(last_seen) else None,
                    "is_active": is_active,
                    "updated_at": now_utc(),
                }
            )


def build_cusip_universe(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    dataset_name: Optional[str] = None,
) -> Dict:
    """
    Build CUSIP universe for all issuers.
    
    Returns:
        Dict with summary statistics
    """
    if dataset_name is None:
        dataset_name = os.getenv("TRACE_DATASET_NAME")
        if not dataset_name:
            raise ValueError("TRACE_DATASET_NAME must be set in environment")
    
    engine = get_engine()
    
    # Create ETL run
    run_id = create_etl_run(engine, "cusip_universe")
    print(f"ETL Run ID: {run_id}")
    
    try:
        # Get access token
        token = get_access_token()
        
        # Get metadata and validate CUSIP field
        print("Fetching dataset metadata...")
        metadata = get_metadata(token, GROUP, dataset_name)
        
        # Validate that dataset has CUSIP field
        if not has_cusip_field(metadata):
            error_msg = (
                f"Dataset '{dataset_name}' has no CUSIP field. "
                f"Choose a CUSIP-capable dataset or switch to CSV export.\n"
                f"Run 'python scripts/finra_discover.py' to find CUSIP-capable datasets."
            )
            raise ValueError(error_msg)
        
        # Infer field map
        field_map = infer_field_map(metadata)
        print(f"Field mapping: CUSIP={field_map['cusip_field']}, TradeDate={field_map['trade_date_field']}, Issuer={field_map['issuer_field']}")
        
        # Get issuers
        issuers = get_issuers(engine)
        print(f"Processing {len(issuers)} issuers...")
        
        all_bonds = []
        failed_issuers = []
        
        for issuer in tqdm(issuers, desc="Issuers"):
            ticker = issuer["ticker"]
            issuer_name = issuer["issuer_name"]
            issuer_id = issuer["issuer_id"]
            finra_key = issuer.get("finra_issuer_key")
            
            print(f"\nProcessing {ticker} ({issuer_name})...")
            
            try:
                # Determine issuer filter
                if not finra_key:
                    print(f"  Determining issuer filter value...")
                    finra_key = determine_issuer_filter(token, dataset_name, field_map, issuer_name, engine, run_id)
                    update_issuer_finra_key(engine, issuer_id, finra_key)
                    print(f"  Using issuer filter: '{finra_key}'")
                else:
                    print(f"  Using cached issuer filter: '{finra_key}'")
                
                # Query monthly slices
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
                
                all_data = []
                current = start
                
                while current <= end:
                    month_end = min(current + timedelta(days=31), end)
                    slice_start = current.strftime("%Y-%m-%d")
                    slice_end = month_end.strftime("%Y-%m-%d")
                    
                    data = query_monthly_slice(
                        token, dataset_name, field_map, finra_key,
                        slice_start, slice_end, engine, run_id
                    )
                    all_data.extend(data)
                    
                    current = month_end + timedelta(days=1)
                    time.sleep(0.1)  # Rate limiting
                
                # Aggregate
                if all_data:
                    bonds_df = aggregate_cusips(all_data, field_map)
                    print(f"  Found {len(bonds_df)} unique CUSIPs")
                    
                    # Upsert
                    upsert_bonds(engine, bonds_df, issuer_id, issuer_name, end_date)
                    all_bonds.append((ticker, issuer_name, len(bonds_df)))
                else:
                    print(f"  No data found for {ticker}")
                    failed_issuers.append((ticker, issuer_name, finra_key))
                
            except Exception as e:
                print(f"  Error processing {ticker}: {e}")
                failed_issuers.append((ticker, issuer_name, str(e)))
        
        # Mark ETL run as complete
        meta = {
            "total_issuers": len(issuers),
            "successful_issuers": len(all_bonds),
            "failed_issuers": len(failed_issuers),
            "failed_details": failed_issuers,
        }
        update_etl_run(engine, run_id, "OK" if not failed_issuers else "FAILED", meta)
        
        return {
            "run_id": run_id,
            "successful": all_bonds,
            "failed": failed_issuers,
        }
    
    except Exception as e:
        update_etl_run(engine, run_id, "FAILED", {"error": str(e)})
        raise


if __name__ == "__main__":
    result = build_cusip_universe()
    print("\n" + "="*60)
    print("CUSIP Universe Build Complete")
    print("="*60)
    print(f"Successful issuers: {len(result['successful'])}")
    for ticker, name, count in result["successful"]:
        print(f"  {ticker}: {count} CUSIPs")
    if result["failed"]:
        print(f"\nFailed issuers: {len(result['failed'])}")
        for item in result["failed"]:
            print(f"  {item}")

