#!/usr/bin/env python3
"""Run scenario engine: stress test scenarios for AI credit crisis."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Path bootstrap
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.db.db import get_engine
from src.scenario.run_scenarios import run_all_scenarios
from src.reporting.figures import (
    plot_scenario_system_path,
    plot_scenario_top10_uplift_bar,
    plot_scenario_bucket_impact,
    plot_scenario_heatmap_issuer_week,
    plot_scenario_spillover_groups,
)
from src.reporting.plot_style import apply_style
from sqlalchemy import text


def print_summary(engine):
    """Print scenario summary tables."""
    print("\n" + "="*80)
    print("SCENARIO SUMMARY")
    print("="*80)
    
    with engine.connect() as conn:
        # Top 10 issuers by cumulative uplift (scenario 3)
        print("\n[1] TOP 10 ISSUERS BY CUMULATIVE UPLIFT (AI Shock + Funding Freeze)")
        print("-" * 80)
        result = conn.execute(text("""
            SELECT 
                i.ticker,
                i.issuer_name,
                SUM(sr.uplift) as cumulative_uplift,
                AVG(sr.scenario_pred_dcredit_proxy) as avg_scenario_pred
            FROM scenario_results_issuer_weekly sr
            JOIN dim_issuer i ON sr.issuer_id = i.issuer_id
            JOIN scenario_definition sd ON sr.scenario_id = sd.scenario_id
            WHERE sd.scenario_name = 'AI Shock + Funding Freeze'
            GROUP BY i.ticker, i.issuer_name
            ORDER BY cumulative_uplift DESC
            LIMIT 10
        """))
        
        rows = result.fetchall()
        for i, (ticker, name, cum_uplift, avg_pred) in enumerate(rows, 1):
            print(f"{i:2d}. {ticker:6s} ({name[:40]:40s}): "
                  f"Cumulative uplift = {cum_uplift:.4f}, Avg scenario pred = {avg_pred:.4f}")
        
        # Top 10 weeks system-wide by total scenario deterioration
        print("\n[2] TOP 10 WEEKS BY SYSTEM-WIDE DETERIORATION (AI Shock + Funding Freeze)")
        print("-" * 80)
        result = conn.execute(text("""
            SELECT 
                sr.week_start,
                SUM(sr.scenario_pred_dcredit_proxy) as total_deterioration,
                COUNT(DISTINCT sr.issuer_id) as issuer_count
            FROM scenario_results_issuer_weekly sr
            JOIN scenario_definition sd ON sr.scenario_id = sd.scenario_id
            WHERE sd.scenario_name = 'AI Shock + Funding Freeze'
            GROUP BY sr.week_start
            ORDER BY total_deterioration DESC
            LIMIT 10
        """))
        
        rows = result.fetchall()
        for i, (week_start, total_det, issuer_count) in enumerate(rows, 1):
            print(f"{i:2d}. {week_start}: Total deterioration = {total_det:.4f} "
                  f"({issuer_count} issuers)")
        
        # Contagion lens: bucket-level aggregation
        print("\n[3] BUCKET-LEVEL IMPACT (Average Uplift by Bucket)")
        print("-" * 80)
        result = conn.execute(text("""
            SELECT 
                sd.scenario_name,
                d.bucket,
                AVG(sr.uplift) as avg_uplift,
                COUNT(DISTINCT sr.issuer_id) as issuer_count
            FROM scenario_results_issuer_weekly sr
            JOIN scenario_definition sd ON sr.scenario_id = sd.scenario_id
            JOIN model_dataset_weekly d ON sr.issuer_id = d.issuer_id AND sr.week_start = d.week_start
            WHERE sd.scenario_name != 'Base Risk-Off'
            GROUP BY sd.scenario_name, d.bucket
            ORDER BY sd.scenario_name, avg_uplift DESC
        """))
        
        rows = result.fetchall()
        current_scenario = None
        for scenario_name, bucket, avg_uplift, issuer_count in rows:
            if scenario_name != current_scenario:
                print(f"\n{scenario_name}:")
                current_scenario = scenario_name
            print(f"  {bucket:15s}: Avg uplift = {avg_uplift:.4f} ({issuer_count} issuers)")
        
        # Spillover groups
        print("\n[4] SPILLOVER TO BANKS / ASSET MANAGERS / TECH SUPPLY CHAIN")
        print("-" * 80)
        result = conn.execute(text("""
            SELECT 
                sd.scenario_name,
                ssg.group_name,
                ssg.spillover_index
            FROM scenario_spillover_groups ssg
            JOIN scenario_definition sd ON ssg.scenario_id = sd.scenario_id
            WHERE sd.scenario_name != 'Base Risk-Off'
            ORDER BY sd.scenario_name, ssg.spillover_index DESC
        """))
        
        rows = result.fetchall()
        current_scenario = None
        for scenario_name, group_name, spillover_index in rows:
            if scenario_name != current_scenario:
                print(f"\n{scenario_name}:")
                current_scenario = scenario_name
            print(f"  {group_name:20s}: Spillover index = {spillover_index:.4f}")


if __name__ == "__main__":
    load_dotenv()
    
    print("="*80)
    print("STAGE 4: SCENARIO ENGINE")
    print("="*80)
    
    # Step 0: Run migrations
    print("\n[0/3] Running migrations...")
    try:
        engine = get_engine()
        migrations_dir = Path(__file__).parent.parent / "src" / "db" / "migrations"
        if migrations_dir.exists():
            migration_files = sorted(migrations_dir.glob("*.sql"))
            for migration_file in migration_files:
                print(f"  Running {migration_file.name}...")
                try:
                    with open(migration_file, "r") as f:
                        sql_content = f.read()
                    with engine.begin() as conn:
                        conn.exec_driver_sql(sql_content)
                    print(f"    ✓ {migration_file.name} completed")
                except Exception as e:
                    error_str = str(e).lower()
                    if ("012" in migration_file.name and 
                        ("column" in error_str and "does not exist" in error_str and "y_dspread_bps" in error_str)):
                        print(f"    ⚠ {migration_file.name}: y_dspread_bps column doesn't exist (OK for EDGAR-first schema)")
                        continue
                    print(f"    ✗ {migration_file.name} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            print(f"  ✓ Completed {len(migration_files)} migration(s)")
        else:
            print(f"  ⚠ Migrations directory not found: {migrations_dir}")
    except Exception as e:
        print(f"  ✗ Error running migrations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 1: Run scenarios
    print("\n[1/3] Running scenarios...")
    try:
        apply_style()
        results = run_all_scenarios(engine, year=2025)
        print(f"  ✓ Completed {len(results)} scenario(s)")
        for name, result in results.items():
            print(f"    - {name}: scenario_id={result['scenario_id']}, "
                  f"rows={result['rows_created']}")
    except Exception as e:
        print(f"  ✗ Error running scenarios: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Print summary
    print("\n[2/3] Generating summary...")
    try:
        print_summary(engine)
        print("\n  ✓ Summary printed")
    except Exception as e:
        print(f"  ✗ Error generating summary: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 3: Generate charts
    print("\n[3/3] Generating charts...")
    chart_paths = []
    
    try:
        path = plot_scenario_system_path(engine)
        if path:
            chart_paths.append(path.name)
            print(f"  ✓ {path.name}")
    except Exception as e:
        print(f"  ⚠ System path chart failed: {e}")
    
    try:
        path = plot_scenario_top10_uplift_bar(engine)
        if path:
            chart_paths.append(path.name)
            print(f"  ✓ {path.name}")
    except Exception as e:
        print(f"  ⚠ Top 10 uplift chart failed: {e}")
    
    try:
        path = plot_scenario_bucket_impact(engine)
        if path:
            chart_paths.append(path.name)
            print(f"  ✓ {path.name}")
    except Exception as e:
        print(f"  ⚠ Bucket impact chart failed: {e}")
    
    try:
        path = plot_scenario_heatmap_issuer_week(engine)
        if path:
            chart_paths.append(path.name)
            print(f"  ✓ {path.name}")
    except Exception as e:
        print(f"  ⚠ Heatmap chart failed: {e}")
    
    try:
        path = plot_scenario_spillover_groups(engine)
        if path:
            chart_paths.append(path.name)
            print(f"  ✓ {path.name}")
    except Exception as e:
        print(f"  ⚠ Spillover groups chart failed: {e}")
    
    print(f"\n  Generated {len(chart_paths)} chart(s)")
    
    output_dir = Path(__file__).parent.parent / "reports" / "figures"
    print(f"\nAll charts saved to: {output_dir}")
    
    print("\n" + "="*80)
    print("SCENARIO ENGINE COMPLETE!")
    print("="*80)

