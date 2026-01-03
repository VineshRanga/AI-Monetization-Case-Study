"""FINRA dataset discovery and metadata retrieval."""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from src.finra.auth import FinraAuthError, get_access_token

load_dotenv()

BASE_URL = "https://api.finra.org"


def list_datasets(token: str, group: str = "FixedIncomeMarket") -> Dict[str, Any]:
    """
    List datasets for a given group.
    
    Args:
        token: FINRA access token
        group: Dataset group name (default: FixedIncomeMarket)
    
    Returns:
        JSON response containing dataset list
    """
    url = f"{BASE_URL}/datasets"
    params = {"group": group}
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise FinraAuthError(f"Failed to list datasets: {e}") from e


def get_metadata(
    token: str,
    group: str,
    name: str,
) -> Dict[str, Any]:
    """
    Get metadata for a specific dataset.
    
    Args:
        token: FINRA access token
        group: Dataset group name
        name: Dataset name
    
    Returns:
        JSON response containing dataset metadata
    """
    url = f"{BASE_URL}/metadata/group/{group}/name/{name}"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise FinraAuthError(f"Failed to get metadata: {e}") from e


def trace_candidates(datasets_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Filter datasets to find TRACE-related candidates.
    
    Args:
        datasets_json: JSON response from list_datasets
    
    Returns:
        List of dataset dicts matching TRACE criteria
    """
    candidates = []
    
    # Handle different possible JSON structures
    datasets = datasets_json
    if isinstance(datasets_json, dict):
        # Try common keys
        if "datasets" in datasets_json:
            datasets = datasets_json["datasets"]
        elif "data" in datasets_json:
            datasets = datasets_json["data"]
        elif "results" in datasets_json:
            datasets = datasets_json["results"]
    
    if not isinstance(datasets, list):
        return candidates
    
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        
        name = dataset.get("name", "").lower()
        description = dataset.get("description", "").lower()
        status = dataset.get("status", "").upper()
        
        # Check if name or description contains 'trace'
        is_trace = "trace" in name or "trace" in description
        
        # If status field exists, check it's Active (case-insensitive)
        if "status" in dataset:
            if status != "ACTIVE":
                continue
        
        if is_trace:
            candidates.append(dataset)
    
    return candidates


def pretty_print_fields(metadata_json: Dict[str, Any]) -> None:
    """
    Print dataset fields in a readable format.
    
    Args:
        metadata_json: JSON response from get_metadata
    """
    # Handle different possible JSON structures
    fields = metadata_json
    if isinstance(metadata_json, dict):
        if "fields" in metadata_json:
            fields = metadata_json["fields"]
        elif "columns" in metadata_json:
            fields = metadata_json["columns"]
        elif "schema" in metadata_json:
            schema = metadata_json["schema"]
            if isinstance(schema, dict) and "fields" in schema:
                fields = schema["fields"]
    
    if not isinstance(fields, list):
        print("No fields found in metadata")
        return
    
    print("\nDataset Fields:")
    print("-" * 80)
    for field in fields:
        if not isinstance(field, dict):
            continue
        
        name = field.get("name", "N/A")
        field_type = field.get("type", field.get("dataType", "N/A"))
        searchable = field.get("searchable", field.get("isSearchable", False))
        filterable = field.get("filterable", field.get("isFilterable", False))
        
        flags = []
        if searchable:
            flags.append("searchable")
        if filterable:
            flags.append("filterable")
        
        flag_str = ", ".join(flags) if flags else "none"
        
        print(f"  {name:30} | {str(field_type):20} | flags: {flag_str}")
    print("-" * 80)


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save JSON data to file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def has_cusip_field(metadata_json: Dict[str, Any]) -> bool:
    """
    Check if metadata contains a CUSIP-like field.
    
    Args:
        metadata_json: JSON response from get_metadata
    
    Returns:
        True if a CUSIP field is found (case-insensitive)
    """
    fields = metadata_json
    if isinstance(metadata_json, dict):
        if "fields" in metadata_json:
            fields = metadata_json["fields"]
        elif "columns" in metadata_json:
            fields = metadata_json["columns"]
        elif "schema" in metadata_json:
            schema = metadata_json["schema"]
            if isinstance(schema, dict) and "fields" in schema:
                fields = schema["fields"]
    
    if not isinstance(fields, list):
        return False
    
    for field in fields:
        if not isinstance(field, dict):
            continue
        field_name = field.get("name", "").lower()
        if "cusip" in field_name:
            return True
    
    return False


def get_key_fields(metadata_json: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Extract key field names from metadata (trade date, yield, price).
    
    Args:
        metadata_json: JSON response from get_metadata
    
    Returns:
        Dict with keys: trade_date, yield, price, cusip
    """
    fields = metadata_json
    if isinstance(metadata_json, dict):
        if "fields" in metadata_json:
            fields = metadata_json["fields"]
        elif "columns" in metadata_json:
            fields = metadata_json["columns"]
        elif "schema" in metadata_json:
            schema = metadata_json["schema"]
            if isinstance(schema, dict) and "fields" in schema:
                fields = schema["fields"]
    
    if not isinstance(fields, list):
        return {"trade_date": None, "yield": None, "price": None, "cusip": None}
    
    result = {"trade_date": None, "yield": None, "price": None, "cusip": None}
    
    for field in fields:
        if not isinstance(field, dict):
            continue
        field_name = field.get("name", "").lower()
        original_name = field.get("name", "")
        
        if "cusip" in field_name:
            result["cusip"] = original_name
        elif "trade" in field_name and "date" in field_name:
            result["trade_date"] = original_name
        elif "yield" in field_name or "ytm" in field_name:
            result["yield"] = original_name
        elif "price" in field_name and "trade" not in field_name:
            result["price"] = original_name
    
    return result


