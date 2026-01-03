"""Ingest SEC fundamentals using companyfacts API."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.config.issuer_cik import ISSUER_CIK
from src.db.db import get_engine, create_etl_run, update_etl_run, log_raw_payload

SEC_BASE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json"

# US-GAAP tag mappings
TAG_MAPPINGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "net_income": ["NetIncomeLoss"],
    "total_debt": ["Debt", "LongTermDebtAndCapitalLeaseObligations", "LongTermDebt"],
    "debt_current": ["DebtCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "interest_expense": ["InterestExpense"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "op_income": ["OperatingIncomeLoss"],
    "depreciation": ["DepreciationDepletionAndAmortization"],
    "lease_current": ["OperatingLeaseLiabilityCurrent"],
    "lease_noncurrent": ["OperatingLeaseLiabilityNoncurrent"],
}


def extract_tag_value(facts: Dict, tag_names: list, unit: str = "USD") -> Optional[float]:
    """Extract value from SEC facts for given tag names."""
    if "us-gaap" not in facts:
        return None
    
    us_gaap = facts["us-gaap"]
    
    for tag_name in tag_names:
        if tag_name in us_gaap:
            tag_data = us_gaap[tag_name]
            
            # Look for units
            if "units" in tag_data:
                if unit in tag_data["units"]:
                    units_data = tag_data["units"][unit]
                    if units_data:
                        # Get most recent value
                        latest = max(units_data, key=lambda x: x.get("end", ""))
                        return float(latest.get("val", 0))
    
    return None


def fetch_company_facts(cik: str, engine: Engine, run_id: int) -> Optional[Dict]:
    """Fetch company facts from SEC API."""
    url = SEC_BASE.format(cik.zfill(10))
    
    headers = {
        "User-Agent": "AICreditRiskAnalysis research@example.com",
        "Accept": "application/json",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Log raw payload
        log_raw_payload(
            engine, run_id, "SEC",
            url,
            {"cik": cik},
            response.status_code,
            data,
        )
        
        return data
    
    except Exception as e:
        print(f"  Warning: Failed to fetch CIK {cik}: {e}")
        return None


def parse_fundamentals(facts: Dict, issuer_id: int, engine: Engine) -> int:
    """Parse fundamentals from SEC facts and store in database."""
    rows_inserted = 0
    
    # Extract values
    revenue = extract_tag_value(facts, TAG_MAPPINGS["revenue"])
    net_income = extract_tag_value(facts, TAG_MAPPINGS["net_income"])
    total_debt = extract_tag_value(facts, TAG_MAPPINGS["total_debt"])
    debt_current = extract_tag_value(facts, TAG_MAPPINGS["debt_current"])
    cash = extract_tag_value(facts, TAG_MAPPINGS["cash"])
    interest_expense = extract_tag_value(facts, TAG_MAPPINGS["interest_expense"])
    capex = extract_tag_value(facts, TAG_MAPPINGS["capex"])
    cfo = extract_tag_value(facts, TAG_MAPPINGS["cfo"])
    op_income = extract_tag_value(facts, TAG_MAPPINGS["op_income"])
    depreciation = extract_tag_value(facts, TAG_MAPPINGS["depreciation"])
    
    # Compute derived
    if total_debt is None and debt_current is not None:
        # Approximate if only current debt available
        total_debt = debt_current
    
    net_debt = None
    if total_debt is not None and cash is not None:
        net_debt = total_debt - cash
    
    fcf = None
    if cfo is not None and capex is not None:
        fcf = cfo - capex
    
    ebitda = None
    if op_income is not None:
        if depreciation is not None:
            ebitda = op_income + depreciation
        else:
            # Use op_income as proxy
            ebitda = op_income
    
    # Extract quarterly data from facts
    if "us-gaap" in facts:
        us_gaap = facts["us-gaap"]
        
        # Get revenue quarterly data
        if TAG_MAPPINGS["revenue"][0] in us_gaap:
            revenue_tag = us_gaap[TAG_MAPPINGS["revenue"][0]]
            if "units" in revenue_tag and "USD" in revenue_tag["units"]:
                quarterly_data = revenue_tag["units"]["USD"]
                
                # Group by period
                periods = {}
                for entry in quarterly_data:
                    end_date = entry.get("end", "")
                    fiscal_year = entry.get("fy", None)
                    fiscal_quarter = entry.get("fp", None)
                    
                    if end_date and fiscal_year:
                        period_key = (end_date, fiscal_year, fiscal_quarter)
                        if period_key not in periods:
                            periods[period_key] = {}
                        
                        periods[period_key]["revenue"] = entry.get("val")
                        periods[period_key]["period_end"] = end_date
                        periods[period_key]["fiscal_year"] = fiscal_year
                        periods[period_key]["fiscal_quarter"] = fiscal_quarter
                
                # Extract other metrics for same periods
                for tag_name, tag_list in TAG_MAPPINGS.items():
                    if tag_name == "revenue":
                        continue
                    
                    for tname in tag_list:
                        if tname in us_gaap:
                            tag_data = us_gaap[tname]
                            if "units" in tag_data and "USD" in tag_data["units"]:
                                for entry in tag_data["units"]["USD"]:
                                    end_date = entry.get("end", "")
                                    fiscal_year = entry.get("fy", None)
                                    fiscal_quarter = entry.get("fp", None)
                                    
                                    if end_date and fiscal_year:
                                        period_key = (end_date, fiscal_year, fiscal_quarter)
                                        if period_key in periods:
                                            periods[period_key][tag_name] = entry.get("val")
                
                # Store periods
                with engine.begin() as conn:
                    for period_key, period_data in periods.items():
                        period_end = pd.to_datetime(period_data["period_end"]).date()
                        
                        conn.execute(
                            text("""
                                INSERT INTO fact_fundamentals_quarterly (
                                    issuer_id, period_end, fiscal_year, fiscal_quarter,
                                    revenue, ebitda, op_income, net_income,
                                    total_debt, cash, net_debt, interest_expense,
                                    capex, cfo, fcf, source
                                )
                                VALUES (
                                    :issuer_id, :period_end, :fiscal_year, :fiscal_quarter,
                                    :revenue, :ebitda, :op_income, :net_income,
                                    :total_debt, :cash, :net_debt, :interest_expense,
                                    :capex, :cfo, :fcf, 'SEC'
                                )
                                ON CONFLICT (issuer_id, period_end) DO UPDATE SET
                                    revenue = EXCLUDED.revenue,
                                    ebitda = EXCLUDED.ebitda,
                                    op_income = EXCLUDED.op_income,
                                    net_income = EXCLUDED.net_income,
                                    total_debt = EXCLUDED.total_debt,
                                    cash = EXCLUDED.cash,
                                    net_debt = EXCLUDED.net_debt,
                                    interest_expense = EXCLUDED.interest_expense,
                                    capex = EXCLUDED.capex,
                                    cfo = EXCLUDED.cfo,
                                    fcf = EXCLUDED.fcf
                            """),
                            {
                                "issuer_id": issuer_id,
                                "period_end": period_end,
                                "fiscal_year": period_data.get("fiscal_year"),
                                "fiscal_quarter": period_data.get("fiscal_quarter"),
                                "revenue": period_data.get("revenue"),
                                "ebitda": period_data.get("ebitda") or period_data.get("op_income"),
                                "op_income": period_data.get("op_income"),
                                "net_income": period_data.get("net_income"),
                                "total_debt": period_data.get("total_debt") or period_data.get("debt_current"),
                                "cash": period_data.get("cash"),
                                "net_debt": None,  # Will compute
                                "interest_expense": period_data.get("interest_expense"),
                                "capex": period_data.get("capex"),
                                "cfo": period_data.get("cfo"),
                                "fcf": None,  # Will compute
                            }
                        )
                        rows_inserted += 1
    
    return rows_inserted


def ingest_sec_fundamentals() -> Dict:
    """Ingest SEC fundamentals for all issuers."""
    engine = get_engine()
    run_id = create_etl_run(engine, "sec_fundamentals_ingest")
    
    results = {}
    
    try:
        # Get issuers with CIK
        with engine.connect() as conn:
            issuers = conn.execute(
                text("SELECT issuer_id, ticker FROM dim_issuer ORDER BY ticker")
            ).fetchall()
        
        for issuer_id, ticker in issuers:
            cik = ISSUER_CIK.get(ticker)
            
            if not cik:
                print(f"  {ticker}: No CIK available")
                results[ticker] = {"rows": 0, "status": "no_cik"}
                continue
            
            print(f"  Fetching {ticker} (CIK: {cik})...")
            
            facts = fetch_company_facts(cik, engine, run_id)
            
            if facts:
                rows = parse_fundamentals(facts, issuer_id, engine)
                results[ticker] = {"rows": rows, "status": "success"}
                print(f"    Inserted {rows} quarterly periods")
            else:
                results[ticker] = {"rows": 0, "status": "fetch_failed"}
        
        update_etl_run(engine, run_id, "OK", {"results": results})
        
        return results
    
    except Exception as e:
        update_etl_run(engine, run_id, "FAILED", {"error": str(e)})
        raise

