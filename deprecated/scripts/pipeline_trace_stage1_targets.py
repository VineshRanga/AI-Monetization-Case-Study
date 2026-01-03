#!/usr/bin/env python3
"""End-to-end pipeline for building target dataset."""
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.db import get_engine
from sqlalchemy import text

if __name__ == "__main__":
    load_dotenv()
    
    print("="*80)
    print("PIPELINE STAGE 1: Building Target Dataset")
    print("="*80)
    
    engine = get_engine()
    
    # Step 1: Run migrations
    print("\n[1/7] Running migrations...")
    try:
        migrations_dir = Path(__file__).parent.parent / "src" / "db" / "migrations"
        if migrations_dir.exists():
            migration_files = sorted(migrations_dir.glob("*.sql"))
            for migration_file in migration_files:
                print(f"  Running {migration_file.name}...")
                with open(migration_file, "r") as f:
                    sql_content = f.read()
                with engine.begin() as conn:
                    for statement in sql_content.split(";"):
                        statement = statement.strip()
                        if statement:
                            conn.execute(text(statement))
            print(f"  ✓ Completed {len(migration_files)} migration(s)")
    except Exception as e:
        print(f"  Warning: {e}")
    
    # Step 2: Seed issuers
    print("\n[2/7] Seeding issuers...")
    try:
        from src.config.universe import ISSUER_UNIVERSE
        
        with engine.begin() as conn:
            for issuer in ISSUER_UNIVERSE:
                conn.execute(
                    text("""
                        INSERT INTO dim_issuer (ticker, issuer_name, bucket)
                        VALUES (:ticker, :issuer_name, :bucket)
                        ON CONFLICT (ticker) DO UPDATE SET
                            issuer_name = EXCLUDED.issuer_name,
                            bucket = EXCLUDED.bucket
                    """),
                    {
                        "ticker": issuer["ticker"],
                        "issuer_name": issuer["issuer_name"],
                        "bucket": issuer["bucket"]
                    }
                )
        print(f"  ✓ Seeded {len(ISSUER_UNIVERSE)} issuers")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        sys.exit(1)
    
    # Step 3: Select bonds
    print("\n[3/7] Selecting representative bonds...")
    try:
        # Check if dim_bond has any bonds
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM dim_bond"))
            bond_count = result.fetchone()[0]
        
        if bond_count == 0:
            print("  ✗ Error: dim_bond is empty")
            print("  Please run: python3 scripts/build_cusips.py first")
            sys.exit(1)
        
        print(f"  Found {bond_count} bonds in dim_bond")
        
        from src.transform.select_bonds import select_bonds_initial, update_bond_selection, save_selection_csv
        
        selected_df = select_bonds_initial(engine)
        if not selected_df.empty:
            update_bond_selection(engine, selected_df)
            data_dir = Path(__file__).parent.parent / "data" / "processed"
            save_selection_csv(selected_df, data_dir)
            print(f"  ✓ Selected {len(selected_df)} bonds")
            
            # Print summary per issuer
            print("\n  Selection summary by issuer:")
            print("  " + "-" * 70)
            for ticker in sorted(selected_df['ticker'].unique()):
                issuer_bonds = selected_df[selected_df['ticker'] == ticker]
                print(f"  {ticker}:")
                for _, bond in issuer_bonds.iterrows():
                    maturity_str = str(bond['maturity_date'])[:10] if pd.notna(bond.get('maturity_date')) else "N/A"
                    first_seen = str(bond.get('first_seen_trade_date', 'N/A'))[:10] if pd.notna(bond.get('first_seen_trade_date')) else "N/A"
                    last_seen = str(bond.get('last_seen_trade_date', 'N/A'))[:10] if pd.notna(bond.get('last_seen_trade_date')) else "N/A"
                    print(f"    Rank {bond['selected_rank']}: {bond['cusip']} | "
                          f"Maturity: {maturity_str} | "
                          f"First: {first_seen} | Last: {last_seen} | "
                          f"Reason: {bond['selected_reason']}")
        else:
            print("  ✗ No bonds selected")
            sys.exit(1)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Ingest TRACE
    print("\n[4/7] Ingesting TRACE time series...")
    try:
        from src.ingest.ingest_trace_timeseries import ingest_trace_timeseries
        
        result = ingest_trace_timeseries()
        summaries = result.get("summaries", [])
        print(f"  ✓ Ingested data for {len(summaries)} CUSIPs")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: Ingest UST yields
    print("\n[5/7] Ingesting U.S. Treasury yields...")
    try:
        from src.ingest.ingest_ust import ingest_ust_yields
        
        result = ingest_ust_yields()
        print(f"  ✓ Ingested {result['rows_inserted']} UST yield rows via {result['method']}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 6: Build spreads
    print("\n[6/7] Computing bond spreads...")
    try:
        from src.transform.build_spreads import build_spreads
        
        result = build_spreads()
        print(f"  ✓ Computed spreads: {result['rows_matched']}/{result['rows_total']} matched ({result['match_rate']:.1f}%)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 7: Build weekly targets
    print("\n[7/7] Building weekly issuer series...")
    try:
        from src.transform.aggregate_weekly_issuer import aggregate_weekly_issuer
        
        result = aggregate_weekly_issuer(method="TOP2_VOL_WEIGHTED")
        print(f"  ✓ Created {result['weeks_created']} weekly records for {result['issuers']} issuers")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Final validation
    print("\n" + "="*80)
    print("VALIDATION QUERIES")
    print("="*80)
    
    with engine.connect() as conn:
        # Weeks per issuer
        print("\nWeeks per issuer:")
        print("-" * 60)
        result = conn.execute(
            text("""
                SELECT issuer_id, COUNT(*) as weeks
                FROM fact_issuer_spread_weekly
                GROUP BY issuer_id
                ORDER BY issuer_id
            """)
        )
        for row in result:
            issuer_id, weeks = row
            ticker_result = conn.execute(
                text("SELECT ticker FROM dim_issuer WHERE issuer_id = :id"),
                {"id": issuer_id}
            ).fetchone()
            ticker = ticker_result[0] if ticker_result else f"Issuer {issuer_id}"
            print(f"  {ticker}: {weeks} weeks")
        
        # Min/max week per issuer
        print("\nWeek range per issuer:")
        print("-" * 60)
        result = conn.execute(
            text("""
                SELECT 
                    i.ticker,
                    MIN(w.week_start) as min_week,
                    MAX(w.week_start) as max_week
                FROM fact_issuer_spread_weekly w
                JOIN dim_issuer i ON w.issuer_id = i.issuer_id
                GROUP BY i.ticker
                ORDER BY i.ticker
            """)
        )
        for row in result:
            print(f"  {row[0]:6} {row[1]} to {row[2]}")
        
        # Sample recent weeks for 3 issuers
        print("\nSample recent weeks (10 most recent for 3 issuers):")
        print("-" * 60)
        issuers = conn.execute(
            text("SELECT ticker FROM dim_issuer ORDER BY ticker LIMIT 3")
        ).fetchall()
        
        for (ticker,) in issuers:
            samples = conn.execute(
                text("""
                    SELECT week_start, spread_bps, dspread_bps
                    FROM fact_issuer_spread_weekly w
                    JOIN dim_issuer i ON w.issuer_id = i.issuer_id
                    WHERE i.ticker = :ticker
                    ORDER BY week_start DESC
                    LIMIT 10
                """),
                {"ticker": ticker}
            ).fetchall()
            
            if samples:
                print(f"\n  {ticker}:")
                for row in samples:
                    dspread_str = f"{row[2]:.1f}" if row[2] is not None else "N/A"
                    print(f"    {row[0]} | spread: {row[1]:.1f} bps | Δspread: {dspread_str} bps")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)

