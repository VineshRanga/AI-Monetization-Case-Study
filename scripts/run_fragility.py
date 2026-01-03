#!/usr/bin/env python3
"""Build and visualize Fragility Scores."""
import os
import sys

# Path bootstrap: add repo root to sys.path so 'src' imports work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sqlalchemy import text
from dotenv import load_dotenv

from src.db.db import get_engine
from src.transform.build_fragility_score import build_fragility_score
from src.reporting.plot_style import (
    apply_style, add_title_subtitle, add_source, save_fig, COLORS
)

# Apply global style
apply_style()

# Source text
SOURCE_TEXT = "Source: SEC EDGAR (fundamentals), Stooq (equities). Macro FRED not loaded. Regime: QQQ/SMH fallback."


def print_ranked_table():
    """Print ranked fragility score table."""
    engine = get_engine()
    
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT 
                    i.ticker,
                    fs.total_score_0_100,
                    MAX(CASE WHEN fps.pillar_name = 'refinancing_wall' THEN fps.score_0_100 END) as p1_refinancing,
                    MAX(CASE WHEN fps.pillar_name = 'cash_generation' THEN fps.score_0_100 END) as p2_cash,
                    MAX(CASE WHEN fps.pillar_name = 'leverage_coverage' THEN fps.score_0_100 END) as p3_leverage,
                    MAX(CASE WHEN fps.pillar_name = 'cyclicality_ai' THEN fps.score_0_100 END) as p4_cyclicality,
                    MAX(CASE WHEN fps.pillar_name = 'structure_opacity' THEN fps.score_0_100 END) as p5_structure
                FROM fragility_scores fs
                JOIN dim_issuer i ON fs.issuer_id = i.issuer_id
                LEFT JOIN fragility_pillar_scores fps ON fs.issuer_id = fps.issuer_id AND fs.asof_date = fps.asof_date
                WHERE fs.asof_date = (SELECT MAX(asof_date) FROM fragility_scores)
                GROUP BY i.ticker, fs.total_score_0_100
                ORDER BY fs.total_score_0_100 DESC
            """),
            conn
        )
    
    if df.empty:
        print("No fragility scores found. Run build_fragility_score() first.")
        return
    
    print("\n" + "="*100)
    print("FRAGILITY SCORE RANKING")
    print("="*100)
    print(f"{'Rank':<6} {'Ticker':<10} {'Total':<8} {'P1:Refin':<10} {'P2:Cash':<10} {'P3:Leverage':<12} {'P4:Cyclical':<12} {'P5:Structure':<12}")
    print("-"*100)
    
    for idx, row in df.iterrows():
        rank = idx + 1
        print(f"{rank:<6} {row['ticker']:<10} {row['total_score_0_100']:>7.1f}  "
              f"{row['p1_refinancing']:>9.1f}  {row['p2_cash']:>9.1f}  "
              f"{row['p3_leverage']:>11.1f}  {row['p4_cyclicality']:>11.1f}  "
              f"{row['p5_structure']:>11.1f}")
    
    print("="*100)


def make_fragility_rank_chart() -> Path:
    """Create ranked bar chart of fragility scores."""
    engine = get_engine()
    
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT MAX(asof_date) as latest_date
                FROM fragility_scores
            """)
        )
        row = result.fetchone()
        asof_date_str = str(row[0]) if row and row[0] else "N/A"
        
        df = pd.read_sql(
            text("""
                SELECT i.ticker, fs.total_score_0_100
                FROM fragility_scores fs
                JOIN dim_issuer i ON fs.issuer_id = i.issuer_id
                WHERE fs.asof_date = (SELECT MAX(asof_date) FROM fragility_scores)
                ORDER BY fs.total_score_0_100 DESC
            """),
            conn
        )
    
    if df.empty:
        print("  Warning: No fragility scores found for rank chart")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.barh(range(len(df)), df["total_score_0_100"], color=COLORS['steelblue'])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["ticker"], fontsize=9)
    ax.invert_yaxis()
    
    ax.set_xlabel("Fragility Score (0-100)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Issuer (Ticker)", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25, axis='x')
    
    add_title_subtitle(
        fig,
        "Fragility Score Ranking",
        f"As of {asof_date_str}. Higher score = more fragile. 5-pillar weighted sum (0-100). Macro FRED not loaded."
    )
    
    add_source(fig, SOURCE_TEXT)
    
    output_dir = Path(__file__).parent.parent / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fragility_rank_bar.png"
    save_fig(fig, output_path)
    
    print(f"  Saved fragility rank chart to {output_path}")
    return output_path


def make_fragility_pillars_heatmap() -> Path:
    """Create heatmap of issuer x pillars."""
    engine = get_engine()
    
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT 
                    i.ticker,
                    fps.pillar_name,
                    fps.score_0_100
                FROM fragility_pillar_scores fps
                JOIN dim_issuer i ON fps.issuer_id = i.issuer_id
                WHERE fps.asof_date = (SELECT MAX(asof_date) FROM fragility_scores)
                ORDER BY i.ticker, fps.pillar_name
            """),
            conn
        )
    
    if df.empty:
        print("  Warning: No pillar scores found for heatmap")
        return None
    
    # Pivot to matrix
    heatmap_df = df.pivot(index='ticker', columns='pillar_name', values='score_0_100')
    
    # Sort by total score
    with engine.connect() as conn:
        totals_df = pd.read_sql(
            text("""
                SELECT i.ticker, fs.total_score_0_100
                FROM fragility_scores fs
                JOIN dim_issuer i ON fs.issuer_id = i.issuer_id
                WHERE fs.asof_date = (SELECT MAX(asof_date) FROM fragility_scores)
                ORDER BY fs.total_score_0_100 DESC
            """),
            conn
        )
    
    if not totals_df.empty:
        heatmap_df = heatmap_df.loc[totals_df["ticker"]]
    
    # Rename columns for display
    pillar_labels = {
        'refinancing_wall': 'P1: Refinancing',
        'cash_generation': 'P2: Cash Gen',
        'leverage_coverage': 'P3: Leverage',
        'cyclicality_ai': 'P4: Cyclicality',
        'structure_opacity': 'P5: Structure'
    }
    heatmap_df.columns = [pillar_labels.get(col, col.replace('_', ' ').title()) for col in heatmap_df.columns]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set colormap with "bad" color for NaN (grey)
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad(color=COLORS['lightgrey'])
    
    # Mask NaNs explicitly
    masked_data = np.ma.masked_invalid(heatmap_df.values)
    
    im = ax.imshow(masked_data, aspect='auto', cmap=cmap, interpolation='nearest')
    
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index, fontsize=9)
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=45, ha='right', fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pillar Score (0-100)', fontsize=10, fontweight='bold')
    
    ax.set_xlabel("Pillar", fontsize=10, fontweight='bold')
    ax.set_ylabel("Issuer (Ticker)", fontsize=10, fontweight='bold')
    ax.grid(False)
    
    add_title_subtitle(
        fig,
        "Fragility Pillars Heatmap",
        "5-pillar breakdown per issuer. Red = higher risk. Grey = insufficient data."
    )
    
    add_source(fig, SOURCE_TEXT)
    
    output_dir = Path(__file__).parent.parent / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fragility_pillars_heatmap.png"
    save_fig(fig, output_path)
    
    print(f"  Saved pillars heatmap to {output_path}")
    return output_path


def make_fragility_vs_predicted_chart() -> Path:
    """Create scatter: fragility score vs predicted dcredit_proxy in RISK_OFF 2025."""
    engine = get_engine()
    
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT 
                    i.ticker,
                    fs.total_score_0_100 as fragility_score,
                    AVG(p.y_pred) as avg_predicted_deterioration
                FROM fragility_scores fs
                JOIN dim_issuer i ON fs.issuer_id = i.issuer_id
                JOIN model_predictions_weekly p ON fs.issuer_id = p.issuer_id
                JOIN model_dataset_weekly d ON p.issuer_id = d.issuer_id AND p.week_start = d.week_start
                WHERE fs.asof_date = (SELECT MAX(asof_date) FROM fragility_scores)
                    AND p.split = 'test'
                    AND p.week_start >= '2025-01-01'
                    AND d.regime_label = 'RISK_OFF'
                    AND p.y_pred IS NOT NULL
                GROUP BY i.ticker, fs.total_score_0_100
            """),
            conn
        )
    
    if df.empty:
        print("  Warning: No data found for fragility vs predicted chart")
        return None
    
    # Calculate correlation
    corr = df["fragility_score"].corr(df["avg_predicted_deterioration"])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(df["fragility_score"], df["avg_predicted_deterioration"], 
              alpha=0.6, s=80, edgecolors='black', linewidth=0.5, color=COLORS['steelblue'])
    
    # Add ticker labels with small text and adjustment
    for _, row in df.iterrows():
        ax.annotate(row["ticker"], 
                   (row["fragility_score"], row["avg_predicted_deterioration"]),
                   fontsize=8, alpha=0.8, 
                   xytext=(5, 5), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax.set_xlabel("Fragility Score (0-100)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Avg Predicted ΔPD proxy (RISK_OFF 2025)", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25)
    
    add_title_subtitle(
        fig,
        "Fragility Score vs Predicted Deterioration (RISK_OFF 2025)",
        f"Correlation: {corr:.3f}. Higher fragility should predict higher deterioration."
    )
    
    add_source(fig, SOURCE_TEXT)
    
    output_dir = Path(__file__).parent.parent / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fragility_vs_predicted_deterioration.png"
    save_fig(fig, output_path)
    
    print(f"  Saved fragility vs predicted chart to {output_path}")
    return output_path


if __name__ == "__main__":
    load_dotenv()
    print("="*80)
    print("FRAGILITY SCORE BUILDER")
    print("="*80)
    
    # Run migrations
    print("\nStep 1: Running migrations...")
    try:
        engine = get_engine()
        migrations_dir = Path(__file__).parent.parent / "src" / "db" / "migrations"
        if migrations_dir.exists():
            migration_files = sorted(migrations_dir.glob("*.sql"))
            for migration_file in migration_files:
                print(f"  Running {migration_file.name}...")
                try:
                    # Each migration file runs in its own transaction
                    # Execute entire file as one script (handles DO blocks, dollar-quoted strings, etc.)
                    with open(migration_file, "r") as f:
                        sql_content = f.read()
                    
                    with engine.begin() as conn:
                        conn.exec_driver_sql(sql_content)
                    
                    print(f"    ✓ {migration_file.name} completed")
                except Exception as e:
                    error_str = str(e).lower()
                    # Gracefully handle migration 012: if y_dspread_bps column doesn't exist,
                    # that's OK (table might have been created from EDGAR-first migration 007)
                    if ("012" in migration_file.name and 
                        ("column" in error_str and "does not exist" in error_str and "y_dspread_bps" in error_str)):
                        print(f"    ⚠ {migration_file.name}: y_dspread_bps column doesn't exist (OK for EDGAR-first schema)")
                        continue
                    print(f"    ✗ {migration_file.name} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            print("  All migrations completed successfully")
        else:
            print(f"  ⚠ Migrations directory not found: {migrations_dir}")
    except Exception as e:
        print(f"  ✗ Error running migrations: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Build scores
    print("\nStep 2: Building fragility scores...")
    result = build_fragility_score()
    print(f"  Result: {result}")
    
    # Print ranked table
    print("\nStep 3: Printing ranked table...")
    print_ranked_table()
    
    # Generate charts
    print("\nStep 4: Generating charts...")
    make_fragility_rank_chart()
    make_fragility_pillars_heatmap()
    make_fragility_vs_predicted_chart()
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)
