"""Metadata-driven field mapping for FINRA TRACE datasets."""
import json
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


def infer_field_map(metadata_json: Dict) -> Dict[str, Optional[str]]:
    """
    Infer field mappings from FINRA metadata.
    
    Args:
        metadata_json: JSON response from get_metadata
    
    Returns:
        Dict with keys: cusip_field, trade_date_field, issuer_field, etc.
    
    Raises:
        ValueError: If required fields cannot be inferred
    """
    # Extract fields from metadata
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
        raise ValueError("Could not find fields list in metadata")
    
    # Get field names (lowercase for matching)
    field_names = []
    for field in fields:
        if isinstance(field, dict):
            name = field.get("name", "")
        elif isinstance(field, str):
            name = field
        else:
            continue
        field_names.append(name)
    
    field_names_lower = [f.lower() for f in field_names]
    
    # Mapping rules
    mapping = {
        "cusip_field": None,
        "trade_date_field": None,
        "issuer_field": None,
        "maturity_field": None,
        "coupon_field": None,
        "issue_date_field": None,
        "security_type_field": None,
        "trade_count_field": None,
        "volume_field": None,
    }
    
    # CUSIP: exact matches
    cusip_candidates = [
        "cusip",
        "cusp",
        "cusip9",
        "cusip_id",
    ]
    for candidate in cusip_candidates:
        if candidate in field_names_lower:
            idx = field_names_lower.index(candidate)
            mapping["cusip_field"] = field_names[idx]
            break
    
    # Trade date
    trade_date_candidates = [
        "tradedate",
        "trade_date",
        "executiondate",
        "execution_date",
        "execdate",
        "exec_date",
        "date",
        "asofdate",
        "as_of_date",
    ]
    for candidate in trade_date_candidates:
        if candidate in field_names_lower:
            idx = field_names_lower.index(candidate)
            mapping["trade_date_field"] = field_names[idx]
            break
    
    # Issuer
    issuer_candidates = [
        "issuername",
        "issuer_name",
        "issuer",
        "issuer_nm",
        "issuingentity",
        "issuing_entity",
        "issuername",
    ]
    for candidate in issuer_candidates:
        if candidate in field_names_lower:
            idx = field_names_lower.index(candidate)
            mapping["issuer_field"] = field_names[idx]
            break
    
    # Maturity
    maturity_candidates = [
        "maturitydate",
        "maturity_date",
        "maturity",
        "matdate",
        "mat_date",
    ]
    for candidate in maturity_candidates:
        if candidate in field_names_lower:
            idx = field_names_lower.index(candidate)
            mapping["maturity_field"] = field_names[idx]
            break
    
    # Coupon
    coupon_candidates = [
        "coupon",
        "couponrate",
        "coupon_rate",
        "rate",
        "interestrate",
        "interest_rate",
    ]
    for candidate in coupon_candidates:
        if candidate in field_names_lower:
            idx = field_names_lower.index(candidate)
            mapping["coupon_field"] = field_names[idx]
            break
    
    # Issue date
    issue_date_candidates = [
        "issuedate",
        "issue_date",
        "issued",
        "offeringdate",
        "offering_date",
    ]
    for candidate in issue_date_candidates:
        if candidate in field_names_lower:
            idx = field_names_lower.index(candidate)
            mapping["issue_date_field"] = field_names[idx]
            break
    
    # Security type
    security_type_candidates = [
        "securitytype",
        "security_type",
        "type",
        "instrumenttype",
        "instrument_type",
    ]
    for candidate in security_type_candidates:
        if candidate in field_names_lower:
            idx = field_names_lower.index(candidate)
            mapping["security_type_field"] = field_names[idx]
            break
    
    # Trade count
    trade_count_candidates = [
        "tradecount",
        "trade_count",
        "numtrades",
        "num_trades",
    ]
    for candidate in trade_count_candidates:
        if candidate in field_names_lower:
            idx = field_names_lower.index(candidate)
            mapping["trade_count_field"] = field_names[idx]
            break
    
    # Volume
    volume_candidates = [
        "volume",
        "tradevolume",
        "trade_volume",
        "dollarvolume",
        "dollar_volume",
    ]
    for candidate in volume_candidates:
        if candidate in field_names_lower:
            idx = field_names_lower.index(candidate)
            mapping["volume_field"] = field_names[idx]
            break
    
    # Validate required fields
    if not mapping["cusip_field"]:
        raise ValueError(
            f"CUSIP field not found. Available fields: {field_names}"
        )
    if not mapping["trade_date_field"]:
        raise ValueError(
            f"Trade date field not found. Available fields: {field_names}"
        )
    if not mapping["issuer_field"]:
        raise ValueError(
            f"Issuer field not found. Available fields: {field_names}"
        )
    
    # Save mapping to file
    data_dir = Path(__file__).parent.parent.parent / "data" / "interim"
    data_dir.mkdir(parents=True, exist_ok=True)
    mapping_file = data_dir / "trace_field_map.json"
    
    with open(mapping_file, "w") as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Field mapping saved to: {mapping_file}")
    
    return mapping

