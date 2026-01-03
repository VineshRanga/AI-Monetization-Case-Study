#!/usr/bin/env python3
"""Ingest SEC company facts (XBRL) and parse into fundamentals."""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.db import get_engine
from src.sec.sec_client import SecClient

# XBRL tag whitelist (US-GAAP)
TAG_WHITELIST = {
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "DebtCurrent",
    "LongTermDebt",
    "Debt",
    "LongTermDebtAndCapitalLeaseObligations",
    "CashAndCashEquivalentsAtCarryingValue",
    "InterestExpense",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "NetCashProvidedByUsedInOperatingActivities",
    "CommonStockSharesOutstanding",
    "DepreciationDepletionAndAmortization",
}

# Tag mappings for fundamentals table
TAG_TO_FUNDAMENTAL = {
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "OperatingIncomeLoss": "op_income",
    "NetIncomeLoss": "net_income",
    "DebtCurrent": "debt_current",
    "LongTermDebt": "long_term_debt",
    "Debt": "total_debt",
    "LongTermDebtAndCapitalLeaseObligations": "total_debt",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "InterestExpense": "interest_expense",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    "NetCashProvidedByUsedInOperatingActivities": "cfo",
    "CommonStockSharesOutstanding": "shares_outstanding",
}


def extract_quarterly_values(facts_data: Dict, issuer_id: int, engine, debug: bool = False) -> int:
    """
    Extract quarterly values from company facts and store in fact_fundamentals_quarterly.
    
    SEC format: facts -> us-gaap -> TAG -> units -> UNIT -> list of observations
    Each observation has: end, val, fy, fp, form, filed, frame
    
    Returns:
        Number of quarters inserted
    """
    if "facts" not in facts_data:
        return 0
    
    facts = facts_data["facts"]
    if "us-gaap" not in facts:
        return 0
    
    us_gaap = facts["us-gaap"]
    
    # Also check dei for shares_outstanding
    dei = facts.get("dei", {})
    
    # Collect all quarterly values by period_end
    quarters = {}  # period_end -> {field: value, fiscal_year, fiscal_quarter, ...}
    
    # Debug: track observations per tag
    debug_counts = {}
    
    # Process US-GAAP tags
    for tag_name, tag_data in us_gaap.items():
        if tag_name not in TAG_WHITELIST:
            continue
        
        fundamental_field = TAG_TO_FUNDAMENTAL.get(tag_name)
        if not fundamental_field:
            continue
        
        # Process units
        if "units" not in tag_data:
            continue
        
        obs_count = 0
        quarterly_obs_count = 0
        
        for unit, unit_data in tag_data["units"].items():
            if unit not in ["USD", "shares"]:
                continue
            
            # Process each observation
            for fact in unit_data:
                if "end" not in fact or "val" not in fact:
                    continue
                
                obs_count += 1
                
                # Extract fields
                period_end_str = fact.get("end")
                value = fact.get("val")
                fp = fact.get("fp", "")  # Fiscal period: Q1, Q2, Q3, Q4, FY
                fy = fact.get("fy")  # Fiscal year
                form = fact.get("form", "")  # 10-Q, 10-K, etc.
                filed_date = fact.get("filed", period_end_str)
                frame = fact.get("frame", "")
                
                if value is None or period_end_str is None:
                    continue
                
                # Parse period_end date
                try:
                    period_end = datetime.strptime(period_end_str, "%Y-%m-%d").date()
                except:
                    continue
                
                # Filter: only keep 2020-2025 (with buffer)
                if period_end < datetime(2019, 1, 1).date() or period_end > datetime(2026, 12, 31).date():
                    continue
                
                # Filter: only quarterly observations
                # Keep if fp is Q1-Q4, OR if form is 10-Q (quarterly filing)
                is_quarterly = False
                if fp in ("Q1", "Q2", "Q3", "Q4"):
                    is_quarterly = True
                elif form == "10-Q":
                    # 10-Q is quarterly filing
                    is_quarterly = True
                elif form == "10-K" and fp == "FY":
                    # Skip annual (FY) observations for quarterly table
                    continue
                
                if not is_quarterly:
                    continue
                
                quarterly_obs_count += 1
                
                # Determine fiscal year/quarter from observation fields
                if fy is not None:
                    fiscal_year = int(fy)
                else:
                    fiscal_year = period_end.year
                
                if fp in ("Q1", "Q2", "Q3", "Q4"):
                    fiscal_quarter = int(fp[1])
                elif form == "10-Q":
                    # Approximate from period_end if fp not available
                    fiscal_quarter = ((period_end.month - 1) // 3) + 1
                else:
                    fiscal_quarter = None
                
                # Initialize quarter if not exists
                if period_end not in quarters:
                    quarters[period_end] = {
                        "fiscal_year": fiscal_year,
                        "fiscal_quarter": fiscal_quarter,
                    }
                
                # Store value (handle multiple values per period by taking latest filed_date)
                if fundamental_field not in quarters[period_end]:
                    quarters[period_end][fundamental_field] = value
                    quarters[period_end]["_filed_" + fundamental_field] = filed_date
                else:
                    # If multiple, take the one with later filed_date
                    existing_filed = quarters[period_end].get("_filed_" + fundamental_field, "")
                    if filed_date and filed_date > existing_filed:
                        quarters[period_end][fundamental_field] = value
                        quarters[period_end]["_filed_" + fundamental_field] = filed_date
                
                # Store raw fact in fact_xbrl_facts
                try:
                    with engine.begin() as conn:
                        conn.execute(
                            text("""
                                INSERT INTO fact_xbrl_facts (
                                    issuer_id, period_end, taxonomy, tag, unit,
                                    value, form, filed_date, frame
                                )
                                VALUES (
                                    :issuer_id, :period_end, :taxonomy, :tag, :unit,
                                    :value, :form, :filed_date, :frame
                                )
                                ON CONFLICT (issuer_id, period_end, taxonomy, tag, unit) DO UPDATE SET
                                    value = EXCLUDED.value,
                                    form = EXCLUDED.form,
                                    filed_date = EXCLUDED.filed_date,
                                    frame = EXCLUDED.frame
                            """),
                            {
                                "issuer_id": issuer_id,
                                "period_end": period_end,
                                "taxonomy": "us-gaap",
                                "tag": tag_name,
                                "unit": unit,
                                "value": float(value) if isinstance(value, (int, float)) else None,
                                "form": form,
                                "filed_date": datetime.strptime(filed_date, "%Y-%m-%d").date() if filed_date else None,
                                "frame": frame,
                            }
                        )
                except Exception:
                    # Skip duplicates or invalid data
                    pass
        
        if debug:
            debug_counts[tag_name] = {"total": obs_count, "quarterly": quarterly_obs_count}
    
    # Also check dei for shares_outstanding if not found in us-gaap
    if "CommonStockSharesOutstanding" in dei:
        shares_tag = dei["CommonStockSharesOutstanding"]
        if "units" in shares_tag and "shares" in shares_tag["units"]:
            for fact in shares_tag["units"]["shares"]:
                if "end" not in fact or "val" not in fact:
                    continue
                
                period_end_str = fact.get("end")
                value = fact.get("val")
                fp = fact.get("fp", "")
                fy = fact.get("fy")
                form = fact.get("form", "")
                
                if value is None or period_end_str is None:
                    continue
                
                try:
                    period_end = datetime.strptime(period_end_str, "%Y-%m-%d").date()
                except:
                    continue
                
                # Filter date range
                if period_end < datetime(2019, 1, 1).date() or period_end > datetime(2026, 12, 31).date():
                    continue
                
                # Filter quarterly
                is_quarterly = fp in ("Q1", "Q2", "Q3", "Q4") or form == "10-Q"
                if not is_quarterly or fp == "FY":
                    continue
                
                if period_end not in quarters:
                    quarters[period_end] = {
                        "fiscal_year": int(fy) if fy else period_end.year,
                        "fiscal_quarter": int(fp[1]) if fp in ("Q1", "Q2", "Q3", "Q4") else None,
                    }
                
                # Only set if not already set from us-gaap
                if "shares_outstanding" not in quarters[period_end]:
                    quarters[period_end]["shares_outstanding"] = value
    
    # Debug output
    if debug and debug_counts:
        print(f"    Debug - Observations per tag:")
        for tag, counts in debug_counts.items():
            print(f"      {tag}: {counts['total']} total, {counts['quarterly']} quarterly")
    
    # Compute derived fields and insert into fact_fundamentals_quarterly
    rows_inserted = 0
    
    for period_end, values in quarters.items():
        revenue = values.get("revenue")
        op_income = values.get("op_income")
        net_income = values.get("net_income")
        total_debt = values.get("total_debt")
        debt_current = values.get("debt_current")
        long_term_debt = values.get("long_term_debt")
        cash = values.get("cash")
        interest_expense = values.get("interest_expense")
        capex = values.get("capex")
        cfo = values.get("cfo")
        shares_outstanding = values.get("shares_outstanding")
        
        # Compute total_debt if not directly available
        if total_debt is None:
            if debt_current is not None and long_term_debt is not None:
                total_debt = debt_current + long_term_debt
            elif long_term_debt is not None:
                total_debt = long_term_debt
        
        # Compute net_debt
        net_debt = None
        if total_debt is not None and cash is not None:
            net_debt = total_debt - cash
        
        # Compute FCF
        fcf = None
        if cfo is not None and capex is not None:
            fcf = cfo - capex
        
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO fact_fundamentals_quarterly (
                            issuer_id, period_end, fiscal_year, fiscal_quarter,
                            revenue, op_income, net_income, total_debt, debt_current,
                            long_term_debt, cash, interest_expense, capex, cfo, fcf,
                            shares_outstanding, source
                        )
                        VALUES (
                            :issuer_id, :period_end, :fiscal_year, :fiscal_quarter,
                            :revenue, :op_income, :net_income, :total_debt, :debt_current,
                            :long_term_debt, :cash, :interest_expense, :capex, :cfo, :fcf,
                            :shares_outstanding, :source
                        )
                        ON CONFLICT (issuer_id, period_end) DO UPDATE SET
                            revenue = EXCLUDED.revenue,
                            op_income = EXCLUDED.op_income,
                            net_income = EXCLUDED.net_income,
                            total_debt = EXCLUDED.total_debt,
                            debt_current = EXCLUDED.debt_current,
                            long_term_debt = EXCLUDED.long_term_debt,
                            cash = EXCLUDED.cash,
                            interest_expense = EXCLUDED.interest_expense,
                            capex = EXCLUDED.capex,
                            cfo = EXCLUDED.cfo,
                            fcf = EXCLUDED.fcf,
                            shares_outstanding = EXCLUDED.shares_outstanding
                    """),
                    {
                        "issuer_id": issuer_id,
                        "period_end": period_end,
                        "fiscal_year": values.get("fiscal_year"),
                        "fiscal_quarter": values.get("fiscal_quarter"),
                        "revenue": float(revenue) if revenue is not None else None,
                        "op_income": float(op_income) if op_income is not None else None,
                        "net_income": float(net_income) if net_income is not None else None,
                        "total_debt": float(total_debt) if total_debt is not None else None,
                        "debt_current": float(debt_current) if debt_current is not None else None,
                        "long_term_debt": float(long_term_debt) if long_term_debt is not None else None,
                        "cash": float(cash) if cash is not None else None,
                        "interest_expense": float(interest_expense) if interest_expense is not None else None,
                        "capex": float(capex) if capex is not None else None,
                        "cfo": float(cfo) if cfo is not None else None,
                        "fcf": float(fcf) if fcf is not None else None,
                        "shares_outstanding": float(shares_outstanding) if shares_outstanding is not None else None,
                        "source": "SEC",
                    }
                )
                rows_inserted += 1
        except Exception as e:
            # Skip duplicates or invalid data
            if debug:
                print(f"    Warning: Failed to insert quarter {period_end}: {e}")
            pass
    
    return rows_inserted


if __name__ == "__main__":
    load_dotenv()
    
    print("="*80)
    print("Ingesting SEC Company Facts (XBRL)")
    print("="*80)
    
    engine = get_engine()
    client = SecClient()
    
    # Get all issuers with CIK
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT issuer_id, ticker, cik FROM dim_issuer WHERE cik IS NOT NULL")
        )
        issuers = [
            {"issuer_id": row[0], "ticker": row[1], "cik": row[2]}
            for row in result
        ]
    
    if not issuers:
        print("  ✗ No issuers with CIK found. Run seed_issuers_edgar.py first.")
        sys.exit(1)
    
    print(f"\nProcessing {len(issuers)} issuers...")
    
    total_quarters = 0
    total_facts = 0
    failed = []
    debug_mode = "--debug" in sys.argv
    
    for issuer in tqdm(issuers, desc="Issuers"):
        ticker = issuer["ticker"]
        cik = issuer["cik"]
        issuer_id = issuer["issuer_id"]
        
        try:
            # Fetch company facts
            facts_data = client.get_companyfacts(cik)
            
            # Extract and store (debug for AAPL or if --debug flag)
            issuer_debug = debug_mode or (ticker == "AAPL" and debug_mode)
            quarters_inserted = extract_quarterly_values(facts_data, issuer_id, engine, debug=issuer_debug)
            total_quarters += quarters_inserted
            
            print(f"  ✓ {ticker}: {quarters_inserted} quarters")
        
        except Exception as e:
            print(f"  ✗ {ticker}: Error - {e}")
            if debug_mode:
                import traceback
                traceback.print_exc()
            failed.append({"ticker": ticker, "error": str(e)})
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Total quarters inserted: {total_quarters}")
    print(f"  Successful issuers: {len(issuers) - len(failed)}")
    if failed:
        print(f"  Failed issuers: {len(failed)}")
        for item in failed:
            print(f"    - {item['ticker']}: {item['error']}")
    
    # Verification query
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM fact_fundamentals_quarterly")
        )
        count = result.fetchone()[0]
        print(f"  Total quarters in fact_fundamentals_quarterly: {count}")
        
        if count > 0:
            # Get date range
            result = conn.execute(
                text("""
                    SELECT MIN(period_end), MAX(period_end), COUNT(DISTINCT issuer_id)
                    FROM fact_fundamentals_quarterly
                """)
            )
            min_date, max_date, issuer_count = result.fetchone()
            print(f"  Date range: {min_date} to {max_date}")
            print(f"  Issuers with data: {issuer_count}")
        else:
            print("  ⚠ No quarters found in fact_fundamentals_quarterly!")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
