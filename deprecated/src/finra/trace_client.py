"""FINRA TRACE Query API client."""
import json
import time
from typing import Any, Dict, Optional, Tuple

import requests
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.finra.auth import get_access_token

API_BASE = "https://api.finra.org"
GROUP = "FixedIncomeMarket"


def post_query(
    token: str,
    dataset_name: str,
    payload: Dict[str, Any],
) -> Tuple[int, Dict[str, Any]]:
    """
    POST a query to FINRA TRACE API.
    
    Args:
        token: FINRA access token
        dataset_name: Dataset name
        payload: Query payload dict
    
    Returns:
        Tuple of (status_code, json_response)
    """
    url = f"{API_BASE}/data/group/{GROUP}/name/{dataset_name}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    
    try:
        json_response = response.json()
    except Exception:
        json_response = {"error": "Failed to parse JSON", "text": response.text[:500]}
    
    return response.status_code, json_response


def request_with_retries(
    token: str,
    dataset_name: str,
    payload: Dict[str, Any],
    max_retries: int = 5,
    initial_delay: float = 1.0,
) -> Tuple[int, Dict[str, Any]]:
    """
    POST query with exponential backoff retries for 429 and 5xx errors.
    
    Args:
        token: FINRA access token
        dataset_name: Dataset name
        payload: Query payload dict
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
    
    Returns:
        Tuple of (status_code, json_response)
    """
    delay = initial_delay
    
    for attempt in range(max_retries):
        status_code, json_response = post_query(token, dataset_name, payload)
        
        # Success
        if 200 <= status_code < 300:
            return status_code, json_response
        
        # Retry on 429 (rate limit) or 5xx (server error)
        if status_code == 429 or (500 <= status_code < 600):
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries - 1} after {delay:.1f}s (status {status_code})")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue
        
        # Don't retry on other errors
        return status_code, json_response
    
    return status_code, json_response


def log_raw_payload(
    engine: Engine,
    run_id: int,
    source: str,
    endpoint: str,
    params: Dict[str, Any],
    status: int,
    payload_json: Dict[str, Any],
) -> None:
    """
    Log a raw HTTP request/response to raw_http_payload table.
    
    Args:
        engine: SQLAlchemy engine
        run_id: ETL run ID
        source: Source identifier (e.g., "FINRA_TRACE")
        endpoint: API endpoint
        params: Request parameters
        status: HTTP status code
        payload_json: Response JSON
    """
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO raw_http_payload 
                (run_id, source, endpoint, request_params, http_status, payload)
                VALUES (:run_id, :source, :endpoint, :request_params, :status, :payload)
            """),
            {
                "run_id": run_id,
                "source": source,
                "endpoint": endpoint,
                "request_params": json.dumps(params),
                "status": status,
                "payload": json.dumps(payload_json),
            }
        )

