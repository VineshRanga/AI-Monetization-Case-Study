#!/usr/bin/env python3
"""Ingest SEC submissions (filing index) for all issuers."""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.db import get_engine
from src.sec.sec_client import SecClient

if __name__ == "__main__":
    load_dotenv()
    
    print("="*80)
    print("Ingesting SEC Submissions (Filing Index)")
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
    
    total_inserted = 0
    failed = []
    
    for issuer in tqdm(issuers, desc="Issuers"):
        ticker = issuer["ticker"]
        cik = issuer["cik"]
        
        try:
            # Fetch submissions
            submissions_data = client.get_submissions(cik)
            
            # Parse structure: {"cik": "...", "name": "...", "filings": {"recent": {...}, "files": [...]}}
            if "filings" not in submissions_data:
                print(f"  ⚠ {ticker}: No 'filings' key in response")
                continue
            
            filings = submissions_data["filings"]
            if "recent" not in filings:
                print(f"  ⚠ {ticker}: No 'recent' key in filings")
                continue
            
            recent = filings["recent"]
            
            # Extract arrays
            accession_numbers = recent.get("accessionNumber", [])
            forms = recent.get("form", [])
            filing_dates = recent.get("filingDate", [])
            report_dates = recent.get("reportDate", [])
            primary_docs = recent.get("primaryDocument", [])
            
            # Insert into database
            inserted = 0
            with engine.begin() as conn:
                for i in range(len(accession_numbers)):
                    accession_no = accession_numbers[i]
                    form = forms[i] if i < len(forms) else None
                    filing_date = filing_dates[i] if i < len(filing_dates) else None
                    report_date = report_dates[i] if i < len(report_dates) else None
                    primary_doc = primary_docs[i] if i < len(primary_docs) else None
                    
                    # Build filing URL
                    filing_url = None
                    if accession_no:
                        # Format: https://www.sec.gov/cgi-bin/viewer?action=view&cik=...&accession_number=...&xbrl_type=v
                        filing_url = f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik}&accession_number={accession_no}&xbrl_type=v"
                    
                    try:
                        conn.execute(
                            text("""
                                INSERT INTO fact_sec_submissions (
                                    cik, accession_no, form, filing_date, report_date,
                                    primary_doc, filing_url
                                )
                                VALUES (
                                    :cik, :accession_no, :form, :filing_date, :report_date,
                                    :primary_doc, :filing_url
                                )
                                ON CONFLICT (cik, accession_no) DO NOTHING
                            """),
                            {
                                "cik": cik,
                                "accession_no": accession_no,
                                "form": form,
                                "filing_date": filing_date,
                                "report_date": report_date,
                                "primary_doc": primary_doc,
                                "filing_url": filing_url,
                            }
                        )
                        inserted += 1
                    except Exception as e:
                        # Skip duplicates or invalid dates
                        pass
            
            total_inserted += inserted
            if inserted > 0:
                print(f"  ✓ {ticker}: {inserted} filings")
            else:
                print(f"  ⚠ {ticker}: No new filings inserted")
        
        except Exception as e:
            print(f"  ✗ {ticker}: Error - {e}")
            failed.append({"ticker": ticker, "error": str(e)})
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Total filings inserted: {total_inserted}")
    print(f"  Successful issuers: {len(issuers) - len(failed)}")
    if failed:
        print(f"  Failed issuers: {len(failed)}")
        for item in failed:
            print(f"    - {item['ticker']}: {item['error']}")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)

