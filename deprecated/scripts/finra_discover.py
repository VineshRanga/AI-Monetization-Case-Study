#!/usr/bin/env python3
"""Discover FINRA TRACE datasets and metadata."""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.finra.auth import FinraAuthError, get_access_token
from src.finra.discovery import (
    get_key_fields,
    get_metadata,
    has_cusip_field,
    list_datasets,
    pretty_print_fields,
    save_json,
    trace_candidates,
)

if __name__ == "__main__":
    load_dotenv()
    
    # Get access token
    try:
        print("Authenticating to FINRA...")
        token = get_access_token()
        print("Authentication successful")
    except FinraAuthError as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)
    
    # List datasets
    try:
        print("\nFetching FixedIncomeMarket datasets...")
        datasets_json = list_datasets(token, group="FixedIncomeMarket")
        
        # Save raw datasets JSON
        data_dir = Path(__file__).parent.parent / "data" / "interim"
        save_json(datasets_json, data_dir / "finra_datasets.json")
        print(f"Saved datasets to: {data_dir / 'finra_datasets.json'}")
        
        # Find TRACE candidates
        candidates = trace_candidates(datasets_json)
        
        print(f"\nFound {len(candidates)} TRACE candidate dataset(s):")
        for candidate in candidates:
            name = candidate.get("name", "N/A")
            desc = candidate.get("description", "N/A")
            status = candidate.get("status", "N/A")
            print(f"  - {name} (status: {status})")
            if desc and desc != "N/A":
                print(f"    Description: {desc[:100]}...")
        
        # Scan all datasets for CUSIP fields
        print("\n" + "="*80)
        print("Scanning all FixedIncomeMarket datasets for CUSIP fields...")
        print("="*80)
        
        # Extract dataset list
        datasets = datasets_json
        if isinstance(datasets_json, dict):
            if "datasets" in datasets_json:
                datasets = datasets_json["datasets"]
            elif "data" in datasets_json:
                datasets = datasets_json["data"]
            elif "results" in datasets_json:
                datasets = datasets_json["results"]
        
        if not isinstance(datasets, list):
            print("Warning: Could not parse dataset list")
            datasets = []
        
        # Filter to datasets with 'TRACE' in description (case-insensitive) to limit API calls
        trace_datasets = []
        for ds in datasets:
            if not isinstance(ds, dict):
                continue
            desc = ds.get("description", "").lower()
            name = ds.get("name", "").lower()
            if "trace" in desc or "trace" in name:
                trace_datasets.append(ds)
        
        print(f"Checking {len(trace_datasets)} datasets with 'TRACE' in name/description...\n")
        
        cusip_capable = []
        no_cusip = []
        
        for ds in trace_datasets:
            name = ds.get("name", "N/A")
            status = ds.get("status", "N/A")
            
            # Skip if not Active
            if status and status.upper() != "ACTIVE":
                continue
            
            try:
                print(f"Checking {name}...", end=" ", flush=True)
                metadata = get_metadata(token, group="FixedIncomeMarket", name=name)
                has_cusip = has_cusip_field(metadata)
                key_fields = get_key_fields(metadata)
                
                if has_cusip:
                    print("✓ HAS CUSIP")
                    cusip_capable.append({
                        "name": name,
                        "description": ds.get("description", "N/A"),
                        "key_fields": key_fields,
                    })
                else:
                    print("✗ NO CUSIP")
                    no_cusip.append({
                        "name": name,
                        "description": ds.get("description", "N/A"),
                    })
            except Exception as e:
                print(f"✗ ERROR: {e}")
                no_cusip.append({
                    "name": name,
                    "description": ds.get("description", "N/A"),
                    "error": str(e),
                })
        
        # Print summary
        print("\n" + "="*80)
        print("CUSIP-CAPABLE DATASETS:")
        print("="*80)
        if cusip_capable:
            for ds in cusip_capable:
                print(f"\n  Dataset: {ds['name']}")
                print(f"  Description: {ds['description'][:100]}...")
                print(f"  Key Fields:")
                for field_type, field_name in ds['key_fields'].items():
                    if field_name:
                        print(f"    - {field_type}: {field_name}")
                    else:
                        print(f"    - {field_type}: (not found)")
        else:
            print("  ✗ NO CUSIP-CAPABLE DATASETS FOUND")
            print("\n  The Query API in this credential scope provides aggregate TRACE datasets only.")
            print("  Recommendation: Use FINRA CSV export for CUSIP-level EOD yields.")
        
        print("\n" + "="*80)
        print("DATASETS WITHOUT CUSIP:")
        print("="*80)
        if no_cusip:
            for ds in no_cusip[:10]:  # Limit output
                print(f"  - {ds['name']}")
            if len(no_cusip) > 10:
                print(f"  ... and {len(no_cusip) - 10} more")
        
        # If TRACE_DATASET_NAME is set, fetch full metadata
        trace_dataset_name = os.getenv("TRACE_DATASET_NAME")
        if trace_dataset_name:
            print(f"\n" + "="*80)
            print(f"Full metadata for TRACE_DATASET_NAME: {trace_dataset_name}")
            print("="*80)
            try:
                metadata = get_metadata(
                    token,
                    group="FixedIncomeMarket",
                    name=trace_dataset_name,
                )
                
                # Save metadata JSON
                safe_name = trace_dataset_name.replace("/", "_").replace(" ", "_")
                save_json(metadata, data_dir / f"finra_metadata_{safe_name}.json")
                print(f"Saved metadata to: {data_dir / f'finra_metadata_{safe_name}.json'}")
                
                # Check CUSIP
                has_cusip = has_cusip_field(metadata)
                if not has_cusip:
                    print("\n⚠️  WARNING: This dataset does NOT have a CUSIP field!")
                    print("   The pipeline will fail when building the CUSIP universe.")
                    print("   Choose a CUSIP-capable dataset from the list above.")
                else:
                    print("\n✓ This dataset has a CUSIP field.")
                
                # Print fields
                pretty_print_fields(metadata)
            except FinraAuthError as e:
                print(f"Failed to get metadata: {e}")
        else:
            print("\n(Set TRACE_DATASET_NAME in .env to fetch full metadata for a specific dataset)")
    
    except FinraAuthError as e:
        print(f"Failed to list datasets: {e}")
        sys.exit(1)

