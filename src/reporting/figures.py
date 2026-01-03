"""Generate all report figures with consistent director-level styling."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import Optional, List, Dict
from datetime import datetime

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from src.reporting.plot_style import (
    apply_style, add_title_subtitle, add_source, save_fig,
    format_date_axis, risk_off_spans, finalize_figure, COLORS
)

# Apply global style
apply_style()

# Source text
SOURCE_TEXT = "Source: SEC EDGAR (fundamentals), Stooq (equities). Macro FRED not loaded. Regime: QQQ/SMH fallback."


def plot_backtest_mean_timeseries(engine: Engine) -> Path:
    """Plot backtest: weekly mean actual vs predicted with RISK_OFF shading."""
    pred_df = pd.read_sql(
        text("""
            SELECT 
                p.week_start,
                p.y_true,
                p.y_pred,
                d.regime_label
            FROM model_predictions_weekly p
            JOIN model_dataset_weekly d ON p.issuer_id = d.issuer_id AND p.week_start = d.week_start
            WHERE p.split = 'test'
            AND p.week_start >= '2025-01-01'
            AND p.week_start <= '2025-12-31'
            AND p.y_true IS NOT NULL
            AND p.y_pred IS NOT NULL
            ORDER BY p.week_start
        """),
        engine
    )
    
    if pred_df.empty:
        print("  Warning: No test predictions found for backtest chart")
        return None
    
    # Aggregate by week (mean across issuers)
    weekly_agg = pred_df.groupby("week_start").agg({
        "y_true": "mean",
        "y_pred": "mean",
        "regime_label": "first"
    }).reset_index()
    
    weekly_agg["week_start"] = pd.to_datetime(weekly_agg["week_start"])
    
    # Get RISK_OFF spans
    regime_df = weekly_agg[["week_start", "regime_label"]].copy()
    spans = risk_off_spans(regime_df, date_col="week_start", label_col="regime_label")
    
    # Calculate metrics
    mae = np.mean(np.abs(weekly_agg["y_pred"] - weekly_agg["y_true"]))
    rmse = np.sqrt(np.mean((weekly_agg["y_pred"] - weekly_agg["y_true"])**2))
    n_weeks = len(weekly_agg)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Shade RISK_OFF spans (contiguous, not weekly stripes)
    for start, end in spans:
        ax.axvspan(start, end, alpha=0.12, color=COLORS['red'], zorder=0)
    
    # Plot actual and predicted
    ax.plot(weekly_agg["week_start"], weekly_agg["y_true"], 
            marker='o', label='Actual (mean)', linewidth=2, markersize=4, 
            color=COLORS['steelblue'], zorder=3)
    ax.plot(weekly_agg["week_start"], weekly_agg["y_pred"], 
            marker='x', label='Predicted (mean)', linewidth=2, markersize=4, 
            linestyle='--', color=COLORS['darkorange'], zorder=3)
    
    ax.axhline(0, color=COLORS['gray'], linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    
    # Format date axis (quarterly/monthly)
    format_date_axis(ax, min_tick="monthly")
    
    ax.set_xlabel('Week Start', fontsize=10, fontweight='bold')
    ax.set_ylabel('Δ PD proxy (weekly change, unitless probability)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25)
    
    # Title/subtitle (1 line max)
    add_title_subtitle(
        fig,
        "Backtest: Weekly Change in PD Proxy (2025)",
        "Mean across issuers. Test window: 2025. Regime: QQQ/SMH fallback."
    )
    
    # Metrics box
    metrics_text = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nN: {n_weeks}"
    ax.text(0.98, 0.85, metrics_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add RISK_OFF regime legend entry
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color=COLORS['steelblue'], label='Actual (mean)', linestyle='-', linewidth=2, markersize=4),
        plt.Line2D([0], [0], marker='x', color=COLORS['darkorange'], label='Predicted (mean)', linestyle='--', linewidth=2, markersize=4),
        Patch(facecolor=COLORS['red'], alpha=0.12, edgecolor='none', label='RISK_OFF regime (equity-only fallback)')
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left', framealpha=0.9)
    add_source(fig, SOURCE_TEXT)
    
    output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "backtest_actual_vs_pred.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    
    print(f"  Saved: {output_path.name}")
    return output_path


def plot_issuer_heatmap_2025(engine: Engine) -> Path:
    """Plot issuer-level risk heatmap for 2025 (aggregated to monthly)."""
    pred_df = pd.read_sql(
        text("""
            SELECT 
                i.ticker,
                p.week_start,
                p.y_pred
            FROM model_predictions_weekly p
            JOIN dim_issuer i ON p.issuer_id = i.issuer_id
            WHERE p.split = 'test'
            AND p.week_start >= '2025-01-01'
            AND p.week_start <= '2025-12-31'
            ORDER BY i.ticker, p.week_start
        """),
        engine
    )
    
    if pred_df.empty:
        print("  Warning: No test predictions found for heatmap")
        return None
    
    # Convert to datetime and aggregate to monthly mean
    pred_df["week_start"] = pd.to_datetime(pred_df["week_start"])
    pred_df["year_month"] = pred_df["week_start"].dt.to_period('M')
    
    monthly_df = pred_df.groupby(['ticker', 'year_month'])['y_pred'].mean().reset_index()
    monthly_df["year_month"] = monthly_df["year_month"].astype(str)
    
    # Pivot to matrix
    heatmap_df = monthly_df.pivot(index='ticker', columns='year_month', values='y_pred')
    
    # Sort issuers by average predicted deterioration
    issuer_avg = heatmap_df.mean(axis=1).sort_values(ascending=False)
    heatmap_df = heatmap_df.loc[issuer_avg.index]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set colormap with "bad" color for NaN (grey)
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad(color=COLORS['lightgrey'])
    
    # Mask NaNs explicitly
    masked_data = np.ma.masked_invalid(heatmap_df.values)
    
    im = ax.imshow(masked_data, aspect='auto', cmap=cmap, interpolation='nearest')
    
    # Set ticks
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index, fontsize=9)
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=45, ha='right', fontsize=8)
    
    # Colorbar with proper spacing
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('Predicted Δ PD proxy (weekly change, unitless probability)', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Month (2025)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Issuer (Ticker)', fontsize=10, fontweight='bold')
    ax.grid(False)
    
    # Ensure proper margins for rotated labels and long ticker names
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.20, top=0.88)
    
    # Title/subtitle (note about grey)
    add_title_subtitle(
        fig,
        "Issuer-Level Predicted Deterioration (2025 Test)",
        "Sorted by average predicted deterioration. Grey = insufficient data."
    )
    
    add_source(fig, SOURCE_TEXT)
    
    output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "issuer_risk_heatmap_2025.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    
    print(f"  Saved: {output_path.name}")
    return output_path


def plot_predicted_vs_actual_scatter(engine: Engine) -> Path:
    """Plot predicted vs actual scatter with symmetric axis limits."""
    pred_df = pd.read_sql(
        text("""
            SELECT 
                p.y_true, 
                p.y_pred,
                d.regime_label
            FROM model_predictions_weekly p
            JOIN model_dataset_weekly d ON p.issuer_id = d.issuer_id AND p.week_start = d.week_start
            WHERE p.split = 'test'
            AND p.y_true IS NOT NULL
            AND p.y_pred IS NOT NULL
        """),
        engine
    )
    
    if pred_df.empty:
        print("  Warning: No test predictions found for scatter chart")
        return None
    
    # Calculate metrics
    mae = np.mean(np.abs(pred_df["y_pred"] - pred_df["y_true"]))
    rmse = np.sqrt(np.mean((pred_df["y_pred"] - pred_df["y_true"])**2))
    r2 = 1 - np.sum((pred_df["y_true"] - pred_df["y_pred"])**2) / np.sum((pred_df["y_true"] - pred_df["y_true"].mean())**2)
    n = len(pred_df)
    
    # Robust axis limits (1-99% quantiles, symmetric around 0)
    all_vals = np.r_[pred_df["y_true"], pred_df["y_pred"]]
    lo, hi = np.quantile(all_vals, [0.01, 0.99])
    m = max(abs(lo), abs(hi))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color points by regime
    risk_on_df = pred_df[pred_df["regime_label"] == "RISK_ON"]
    risk_off_df = pred_df[pred_df["regime_label"] == "RISK_OFF"]
    
    if not risk_on_df.empty:
        ax.scatter(risk_on_df["y_true"], risk_on_df["y_pred"], alpha=0.5, s=30, 
                  edgecolors='black', linewidth=0.5, color=COLORS['steelblue'], 
                  label='RISK_ON', zorder=3)
    if not risk_off_df.empty:
        ax.scatter(risk_off_df["y_true"], risk_off_df["y_pred"], alpha=0.5, s=30, 
                  edgecolors='black', linewidth=0.5, color=COLORS['darkorange'], 
                  label='RISK_OFF', zorder=3)
    
    # Perfect prediction line (45-degree)
    ax.plot([-m, m], [-m, m], 'r--', linewidth=2, label='Perfect prediction', zorder=2)
    
    # Equal aspect and symmetric axis limits
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-m * 1.05, m * 1.05)
    ax.set_ylim(-m * 1.05, m * 1.05)
    
    ax.set_xlabel('Actual Δ PD proxy (weekly change, unitless probability)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Predicted Δ PD proxy (weekly change, unitless probability)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25)
    
    add_title_subtitle(
        fig,
        "Predicted vs Actual (2025 Test)",
        "Color-coded by regime. 45° line indicates perfect prediction."
    )
    
    # Stats box (upper-left, with padding)
    metrics_text = f"N: {n}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}"
    ax.text(0.05, 0.85, metrics_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    add_source(fig, SOURCE_TEXT)
    
    output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "predicted_vs_actual_scatter.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    
    print(f"  Saved: {output_path.name}")
    return output_path


def plot_regime_timeline(engine: Engine) -> Path:
    """Plot regime probability timeline with contiguous RISK_OFF shading."""
    regime_df = pd.read_sql(
        text("""
            SELECT week_start, prob_risk_off, regime_label
            FROM model_regime_weekly
            WHERE week_start >= '2020-01-01'
            AND week_start <= '2025-12-31'
            ORDER BY week_start
        """),
        engine
    )
    
    if regime_df.empty:
        print("  Warning: No regime data found for timeline chart")
        return None
    
    regime_df["week_start"] = pd.to_datetime(regime_df["week_start"])
    
    # Compute 4-week rolling mean
    regime_df["prob_risk_off_roll4"] = regime_df["prob_risk_off"].rolling(window=4, min_periods=1).mean()
    
    # Get RISK_OFF spans
    spans = risk_off_spans(regime_df, date_col="week_start", label_col="regime_label")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Shade RISK_OFF spans (contiguous, not weekly stripes)
    for start, end in spans:
        ax.axvspan(start, end, alpha=0.12, color=COLORS['red'], zorder=0)
    
    # Plot probability (with rolling mean)
    ax.plot(regime_df["week_start"], regime_df["prob_risk_off"], 
           linewidth=1.5, color=COLORS['steelblue'], alpha=0.5, label='P(Risk-Off)', zorder=2)
    ax.plot(regime_df["week_start"], regime_df["prob_risk_off_roll4"], 
           linewidth=2, color=COLORS['steelblue'], label='P(Risk-Off, 4w rolling)', zorder=3)
    
    ax.axhline(0.5, color=COLORS['gray'], linestyle='--', linewidth=1, alpha=0.5, label='50% threshold', zorder=1)
    
    # Set y-limits to [0, 1]
    ax.set_ylim(0, 1)
    
    # Format date axis (quarterly)
    format_date_axis(ax, min_tick="quarterly")
    
    ax.set_xlabel('Week Start', fontsize=10, fontweight='bold')
    ax.set_ylabel('P(Risk-Off)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25)
    
    add_title_subtitle(
        fig,
        "Risk Regime Probability (2020-2025)",
        "Probability of RISK_OFF regime. Shaded regions indicate RISK_OFF periods."
    )
    
    # Add RISK_OFF regime legend entry
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color=COLORS['steelblue'], alpha=0.5, linewidth=1.5, label='P(Risk-Off)'),
        plt.Line2D([0], [0], color=COLORS['steelblue'], linewidth=2, label='P(Risk-Off, 4w rolling)'),
        plt.Line2D([0], [0], color=COLORS['gray'], linestyle='--', linewidth=1, alpha=0.5, label='50% threshold'),
        Patch(facecolor=COLORS['red'], alpha=0.12, edgecolor='none', label='RISK_OFF regime (equity-only fallback)')
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='best', framealpha=0.9)
    add_source(fig, SOURCE_TEXT)
    
    output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "regime_timeline_2020_2025.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    
    print(f"  Saved: {output_path.name}")
    return output_path


def plot_residual_hist(engine: Engine) -> Path:
    """Plot residual histogram with robust axis limits."""
    pred_df = pd.read_sql(
        text("""
            SELECT y_true, y_pred
            FROM model_predictions_weekly
            WHERE split = 'test'
            AND y_true IS NOT NULL
            AND y_pred IS NOT NULL
        """),
        engine
    )
    
    if pred_df.empty:
        print("  Warning: No test predictions found for residuals chart")
        return None
    
    pred_df["residual"] = pred_df["y_pred"] - pred_df["y_true"]
    mean_residual = pred_df["residual"].mean()
    std_residual = pred_df["residual"].std()
    n = len(pred_df)
    
    # Robust axis limits (1-99% quantiles)
    lo, hi = np.quantile(pred_df["residual"], [0.01, 0.99])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(pred_df["residual"], bins=50, edgecolor='black', alpha=0.7, 
           color=COLORS['steelblue'], range=(lo, hi))
    ax.axvline(0, color=COLORS['red'], linestyle='--', linewidth=2, label='Zero residual', zorder=2)
    ax.axvline(mean_residual, color=COLORS['green'], linestyle='--', linewidth=2, 
               label=f'Mean: {mean_residual:.4f}', zorder=2)
    
    # Set robust x-limits
    ax.set_xlim(lo, hi)
    
    ax.set_xlabel('Residual (Predicted − Actual Δ PD proxy, unitless probability)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25)
    
    # Title/subtitle (separate, not stacked)
    add_title_subtitle(
        fig,
        "Residual Distribution (2025 Test)",
        "Predicted minus Actual ΔPD proxy. X-limits = 1-99% quantiles."
    )
    
    # Stats box (upper right, with padding)
    stats_text = f"Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}\nN: {n}"
    ax.text(0.98, 0.85, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    add_source(fig, SOURCE_TEXT)
    
    output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "residuals_hist.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    
    print(f"  Saved: {output_path.name}")
    return output_path


def plot_feature_importance_bar(engine: Engine, regime: str = "RISK_OFF") -> Path:
    """Plot feature importance bar chart (top 10 features)."""
    # Find the latest model_id for regime (SQLAlchemy 2.0 compatible)
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT model_id
                FROM model_registry
                WHERE regime = :regime AND model_name = :model_name
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"regime": regime, "model_name": f"xgb_credit_{regime.lower()}_v1"}
        )
        model_row = result.fetchone()
    
    if model_row is None:
        print(f"  Warning: No {regime} model found in model_registry. Skipping feature importance chart.")
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(0.5, 0.5, f'No feature importance available.\nRe-run Stage 3 to populate model_feature_importance.', 
               transform=ax.transAxes, fontsize=12, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        add_title_subtitle(fig, f"Top Risk Drivers in {regime} Regime", "No model found in model_registry.")
        ax.axis('off')
        add_source(fig, SOURCE_TEXT)
        output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "shap_summary_bar.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_fig(fig, output_path)
        return output_path
    
    model_id = model_row[0]
    
    # Load feature importances (top 10) - pd.read_sql can use Engine directly
    importance_df = pd.read_sql(
        text("""
            SELECT feature, 
                   COALESCE(importance_norm, importance_raw, importance) as importance_val
            FROM model_feature_importance
            WHERE model_id = :model_id
            ORDER BY COALESCE(importance_norm, importance_raw, importance) DESC
            LIMIT 10
        """),
        engine,
        params={"model_id": model_id}
    )
    
    if importance_df.empty:
        print(f"  Warning: No feature importance data found for {regime} model (model_id={model_id}). Skipping chart.")
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(0.5, 0.5, 'No feature importance found.\nRe-run Stage 3 to populate model_feature_importance.', 
               transform=ax.transAxes, fontsize=12, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        add_title_subtitle(fig, f"Top Risk Drivers in {regime} Regime", "No feature importance data available.")
        ax.axis('off')
        add_source(fig, SOURCE_TEXT)
        output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "shap_summary_bar.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_fig(fig, output_path)
        return output_path
    
    # Normalize to percentage (ensure sum = 100)
    total = importance_df["importance_val"].sum()
    if total > 0:
        importance_df["importance_pct"] = (importance_df["importance_val"] / total) * 100
    else:
        importance_df["importance_pct"] = 0.0
    
    # Extract features and importances
    features = importance_df["feature"].tolist()
    importances_pct = importance_df["importance_pct"].tolist()
    
    # Create figure with director-level formatting
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.barh(range(len(features)), importances_pct, color=COLORS['steelblue'])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.invert_yaxis()
    
    # Format x-axis as percentage
    max_pct = max(importances_pct) if importances_pct else 10
    ax.set_xticks(np.arange(0, max_pct + 5, 5))
    
    # Strong axis labels
    ax.set_xlabel('% of total importance (normalized)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25, axis='x')
    
    # Title/subtitle (no overlap) - using helper
    add_title_subtitle(
        fig,
        f"Top Risk Drivers in {regime} Regime",
        "XGBoost feature importance (normalized gain)."
    )
    
    # Increase left margin for feature names and ensure source footer is visible
    fig.subplots_adjust(left=0.25, bottom=0.12)
    
    # Source footer (consistent with other plots, outside axes)
    add_source(fig, SOURCE_TEXT)
    
    # Save with proper settings
    output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "shap_summary_bar.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    
    print(f"  Saved: {output_path.name}")
    return output_path


def plot_regime_vol_surface(engine: Engine) -> Path:
    """Plot regime volatility surface map (2D heatmap of rolling std dev of prob_risk_off)."""
    regime_df = pd.read_sql(
        text("""
            SELECT week_start, prob_risk_off, regime_label
            FROM model_regime_weekly
            WHERE week_start >= '2020-01-01'
            AND week_start <= '2025-12-31'
            ORDER BY week_start
        """),
        engine
    )
    
    if regime_df.empty:
        print("  Warning: No regime data found for vol surface chart")
        return None
    
    regime_df["week_start"] = pd.to_datetime(regime_df["week_start"])
    
    # Compute rolling volatility across multiple window sizes (2 to 26 weeks)
    window_sizes = range(2, 27)  # 2 to 26 weeks
    weeks = regime_df["week_start"].values
    prob_values = regime_df["prob_risk_off"].values
    
    # Build 2D matrix: rows = window sizes, cols = week_start
    vol_matrix = np.full((len(window_sizes), len(weeks)), np.nan)
    
    for i, window_size in enumerate(window_sizes):
        rolling_std = regime_df["prob_risk_off"].rolling(window=window_size, min_periods=window_size).std()
        vol_matrix[i, :] = rolling_std.values
    
    # Get RISK_OFF spans for overlay
    spans = risk_off_spans(regime_df, date_col="week_start", label_col="regime_label")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot heatmap
    # Convert dates to numeric for extent
    extent = [
        mdates.date2num(weeks[0]),
        mdates.date2num(weeks[-1]),
        window_sizes[0] - 0.5,  # Center on integer values
        window_sizes[-1] + 0.5
    ]
    
    im = ax.imshow(vol_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest',
                   extent=extent, origin='lower')
    
    # Overlay RISK_OFF spans as thin bands at top
    for start, end in spans:
        ax.axvspan(start, end, ymin=0.95, ymax=1.0, alpha=0.5, color=COLORS['red'], zorder=3)
    
    # Format x-axis (dates) - need to set locator/formatter after imshow
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.MonthLocator(interval=3)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha='right')
    ax.tick_params(labelsize=9)
    
    # Format y-axis (window sizes)
    ax.set_yticks(range(2, 27, 4))  # Every 4 weeks
    ax.set_ylabel('Rolling Window Size (weeks)', fontsize=10, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Rolling σ of P(Risk-Off)', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Week Start', fontsize=10, fontweight='bold')
    ax.grid(False)
    
    add_title_subtitle(
        fig,
        "Regime Volatility Surface Map (2020-2025)",
        "Realized volatility by week and lookback window, conditioned on regime probability. Rolling σ of P(Risk-Off)."
    )
    
    # Add RISK_OFF regime legend entry
    from matplotlib.patches import Patch
    # Create a custom legend for the red bands
    legend_elements = [
        Patch(facecolor=COLORS['red'], alpha=0.5, edgecolor='none', label='RISK_OFF regime (equity-only fallback)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    add_source(fig, SOURCE_TEXT)
    
    output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "regime_vol_surface.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    
    print(f"  Saved: {output_path.name}")
    return output_path


def plot_scenario_system_path(engine: Engine) -> Path:
    """Plot weekly system-wide uplift: scenario 2 vs scenario 3 (Option 1: plot uplift)."""
    df = pd.read_sql(
        text("""
            SELECT 
                sr.week_start,
                sd.scenario_name,
                SUM(sr.uplift) as total_uplift
            FROM scenario_results_issuer_weekly sr
            JOIN scenario_definition sd ON sr.scenario_id = sd.scenario_id
            WHERE sd.scenario_name IN ('AI Monetization Shock', 'AI Shock + Funding Freeze')
            GROUP BY sr.week_start, sd.scenario_name
            ORDER BY sr.week_start, sd.scenario_name
        """),
        engine
    )
    
    if df.empty:
        print("  Warning: No scenario data found for system path chart")
        return None
    
    df["week_start"] = pd.to_datetime(df["week_start"])
    
    # Get regime spans for shading
    regime_df = pd.read_sql(
        text("""
            SELECT DISTINCT week_start, regime_label
            FROM model_dataset_weekly
            WHERE week_start >= '2025-01-01' AND week_start <= '2025-12-31'
            ORDER BY week_start
        """),
        engine
    )
    regime_df["week_start"] = pd.to_datetime(regime_df["week_start"])
    spans = risk_off_spans(regime_df, date_col="week_start", label_col="regime_label")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Shade RISK_OFF periods
    for start, end in spans:
        ax.axvspan(start, end, alpha=0.15, color=COLORS['red'], zorder=1)
    
    # Plot each scenario (uplift only, baseline is 0)
    scenarios = ["AI Monetization Shock", "AI Shock + Funding Freeze"]
    colors = [COLORS['darkorange'], COLORS['red']]
    
    for scenario, color in zip(scenarios, colors):
        scenario_data = df[df["scenario_name"] == scenario].sort_values("week_start")
        if not scenario_data.empty:
            ax.plot(scenario_data["week_start"], scenario_data["total_uplift"],
                   linewidth=2, color=color, label=scenario, zorder=2)
    
    # Baseline line at 0
    ax.axhline(0, color=COLORS['gray'], linestyle='--', linewidth=1, alpha=0.5, label='Baseline (no shock)', zorder=1)
    
    format_date_axis(ax, min_tick="monthly", rotation=30)
    ax.set_ylabel('System-Wide Uplift (Sum of Δ PD proxy uplift, unitless probability)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Week Start', fontsize=10, fontweight='bold')
    
    # Add RISK_OFF regime legend entry
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color=COLORS['darkorange'], linewidth=2, label='AI Monetization Shock'),
        plt.Line2D([0], [0], color=COLORS['red'], linewidth=2, label='AI Shock + Funding Freeze'),
        plt.Line2D([0], [0], color=COLORS['gray'], linestyle='--', linewidth=1, alpha=0.5, label='Baseline (no shock)'),
        Patch(facecolor=COLORS['red'], alpha=0.15, edgecolor='none', label='RISK_OFF regime (equity-only fallback)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.25)
    
    add_title_subtitle(
        fig,
        "Scenario System Path: System-Wide Uplift (2025)",
        "Weekly sum of scenario uplift across all issuers. Baseline = 0 (no shock). Red shading = RISK_OFF periods."
    )
    
    add_source(fig, SOURCE_TEXT)
    
    output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "scenario_system_path.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    
    print(f"  Saved: {output_path.name}")
    return output_path


def plot_scenario_top10_uplift_bar(engine: Engine) -> Path:
    """Plot top 10 issuers by cumulative uplift (scenario 3)."""
    df = pd.read_sql(
        text("""
            SELECT 
                i.ticker,
                i.issuer_name,
                SUM(sr.uplift) as cumulative_uplift
            FROM scenario_results_issuer_weekly sr
            JOIN dim_issuer i ON sr.issuer_id = i.issuer_id
            JOIN scenario_definition sd ON sr.scenario_id = sd.scenario_id
            WHERE sd.scenario_name = 'AI Shock + Funding Freeze'
            GROUP BY i.ticker, i.issuer_name
            ORDER BY cumulative_uplift DESC
            LIMIT 10
        """),
        engine
    )
    
    if df.empty:
        print("  Warning: No scenario data found for top 10 uplift chart")
        return None
    
    # Sort by uplift (ascending for barh)
    df = df.sort_values("cumulative_uplift", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create labels
    labels = [f"{row['ticker']} ({row['issuer_name'][:30]})" for _, row in df.iterrows()]
    
    ax.barh(range(len(df)), df["cumulative_uplift"], color=COLORS['red'], alpha=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Cumulative Uplift (Δ PD proxy, unitless probability)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25, axis='x')
    
    add_title_subtitle(
        fig,
        "Top 10 Issuers by Cumulative Uplift (AI Shock + Funding Freeze)",
        "Sum of scenario uplift across all 2025 weeks. Higher = more vulnerable to AI shock + funding freeze."
    )
    
    add_source(fig, SOURCE_TEXT)
    
    output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "scenario_top10_uplift_bar.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    
    print(f"  Saved: {output_path.name}")
    return output_path


def plot_scenario_bucket_impact(engine: Engine) -> Path:
    """Plot average uplift by bucket for each scenario (grouped bars)."""
    df = pd.read_sql(
        text("""
            SELECT 
                sd.scenario_name,
                d.bucket,
                AVG(sr.uplift) as avg_uplift
            FROM scenario_results_issuer_weekly sr
            JOIN scenario_definition sd ON sr.scenario_id = sd.scenario_id
            JOIN model_dataset_weekly d ON sr.issuer_id = d.issuer_id AND sr.week_start = d.week_start
            WHERE sd.scenario_name != 'Base Risk-Off'
            GROUP BY sd.scenario_name, d.bucket
            ORDER BY sd.scenario_name, d.bucket
        """),
        engine
    )
    
    if df.empty:
        print("  Warning: No scenario data found for bucket impact chart")
        return None
    
    # Pivot for grouped bars
    pivot_df = df.pivot(index="bucket", columns="scenario_name", values="avg_uplift")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(pivot_df.index))
    width = 0.35
    
    scenarios = pivot_df.columns.tolist()
    colors = [COLORS['darkorange'], COLORS['red']]
    
    for i, (scenario, color) in enumerate(zip(scenarios, colors)):
        offset = (i - len(scenarios)/2 + 0.5) * width
        ax.bar(x + offset, pivot_df[scenario], width, label=scenario, color=color, alpha=0.7)
    
    ax.set_xlabel('Bucket', fontsize=10, fontweight='bold')
    ax.set_ylabel('Average Uplift (Δ PD proxy, unitless probability)', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, fontsize=9)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.25, axis='y')
    
    add_title_subtitle(
        fig,
        "Bucket-Level Impact by Scenario",
        "Average scenario uplift by issuer bucket. Higher = more vulnerable to AI shock."
    )
    
    add_source(fig, SOURCE_TEXT)
    
    output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "scenario_bucket_impact.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    
    print(f"  Saved: {output_path.name}")
    return output_path


def plot_scenario_heatmap_issuer_week(engine: Engine) -> Path:
    """Plot heatmap (issuer x week) of scenario uplift for scenario 3."""
    df = pd.read_sql(
        text("""
            SELECT 
                i.ticker,
                sr.week_start,
                sr.uplift
            FROM scenario_results_issuer_weekly sr
            JOIN dim_issuer i ON sr.issuer_id = i.issuer_id
            JOIN scenario_definition sd ON sr.scenario_id = sd.scenario_id
            WHERE sd.scenario_name = 'AI Shock + Funding Freeze'
            ORDER BY i.ticker, sr.week_start
        """),
        engine
    )
    
    if df.empty:
        print("  Warning: No scenario data found for heatmap chart")
        return None
    
    df["week_start"] = pd.to_datetime(df["week_start"])
    
    # Pivot: issuers (rows) x weeks (cols)
    pivot_df = df.pivot(index="ticker", columns="week_start", values="uplift")
    
    # Sort by total uplift (descending)
    pivot_df["total_uplift"] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values("total_uplift", ascending=False)
    pivot_df = pivot_df.drop(columns=["total_uplift"])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set colormap with "bad" color for NaN (light grey)
    cmap = plt.get_cmap('YlOrRd').copy()
    cmap.set_bad(color=COLORS['lightgrey'])
    
    # Mask NaNs explicitly
    masked_data = np.ma.masked_invalid(pivot_df.values)
    
    # Plot heatmap
    im = ax.imshow(masked_data, aspect='auto', cmap=cmap, interpolation='nearest')
    
    # Set ticks (monthly for weeks, all issuers)
    week_dates = pivot_df.columns
    month_indices = []
    month_labels = []
    for i, date in enumerate(week_dates):
        if date.day <= 7:  # First week of month
            month_indices.append(i)
            month_labels.append(date.strftime('%b %Y'))
    
    ax.set_xticks(month_indices)
    ax.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index, fontsize=8)
    
    ax.set_xlabel('Week Start', fontsize=10, fontweight='bold')
    ax.set_ylabel('Issuer (Ticker)', fontsize=10, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Uplift (Δ PD proxy, unitless probability)', fontsize=10, fontweight='bold')
    
    add_title_subtitle(
        fig,
        "Scenario Uplift Heatmap: Issuer x Week (AI Shock + Funding Freeze)",
        "Higher values (red) = more vulnerable weeks. Sorted by total cumulative uplift. Grey = insufficient data."
    )
    
    add_source(fig, SOURCE_TEXT)
    
    output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "scenario_heatmap_issuer_week.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    
    print(f"  Saved: {output_path.name}")
    return output_path


def plot_scenario_spillover_groups(engine: Engine) -> Path:
    """Plot spillover indices for exposure groups."""
    df = pd.read_sql(
        text("""
            SELECT 
                sd.scenario_name,
                ssg.group_name,
                ssg.spillover_index
            FROM scenario_spillover_groups ssg
            JOIN scenario_definition sd ON ssg.scenario_id = sd.scenario_id
            WHERE sd.scenario_name != 'Base Risk-Off'
            ORDER BY sd.scenario_name, ssg.spillover_index DESC
        """),
        engine
    )
    
    if df.empty:
        print("  Warning: No spillover data found for spillover groups chart")
        return None
    
    # Pivot for grouped bars
    pivot_df = df.pivot(index="group_name", columns="scenario_name", values="spillover_index")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(pivot_df.index))
    width = 0.35
    
    scenarios = pivot_df.columns.tolist()
    colors = [COLORS['darkorange'], COLORS['red']]
    
    for i, (scenario, color) in enumerate(zip(scenarios, colors)):
        offset = (i - len(scenarios)/2 + 0.5) * width
        ax.bar(x + offset, pivot_df[scenario], width, label=scenario, color=color, alpha=0.7)
    
    ax.set_xlabel('Exposure Group', fontsize=10, fontweight='bold')
    ax.set_ylabel('Spillover Index', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    # Fix x tick labels with spaces
    label_map = {
        'Banks': 'Banks',
        'AssetManagers': 'Asset Managers',
        'TechSupplyChain': 'Tech Supply Chain'
    }
    labels = [label_map.get(idx, idx) for idx in pivot_df.index]
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.25, axis='y')
    
    add_title_subtitle(
        fig,
        "Spillover to Banks / Asset Managers / Tech Supply Chain",
        "Weighted average of bucket-level uplift. Higher = more exposure to AI shock."
    )
    
    add_source(fig, SOURCE_TEXT)
    
    output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "scenario_spillover_groups.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    
    print(f"  Saved: {output_path.name}")
    return output_path


def _parse_yaml_fallback(content: str) -> List[Dict]:
    """
    Fallback YAML parser for simple event structure.
    Parses events with date, label, category, priority fields.
    """
    events = []
    current_event = None
    in_events = False
    
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Check if we're in events section
        if line.startswith('events:'):
            in_events = True
            continue
        
        if not in_events:
            continue
        
        # Start of new event
        if line == '-' or line.startswith('- '):
            if current_event and 'date' in current_event:
                events.append(current_event)
            current_event = {}
            continue
        
        # Parse key-value pairs
        if ':' in line and current_event is not None:
            key, value = line.split(':', 1)
            key = key.strip().strip('"').strip("'")
            value = value.strip().strip('"').strip("'")
            
            if key == 'date':
                current_event['date'] = value
            elif key == 'label':
                current_event['label'] = value
            elif key == 'category':
                current_event['category'] = value
            elif key == 'priority':
                try:
                    current_event['priority'] = int(value)
                except ValueError:
                    current_event['priority'] = 1
    
    # Add last event
    if current_event and 'date' in current_event:
        events.append(current_event)
    
    return events


def load_macro_events_yaml() -> List[Dict]:
    """
    Load macro events from YAML file with robust fallback.
    Always returns a list (never empty if file exists).
    """
    # Hardcoded default events (always available) - matches YAML
    default_events = [
        {"date": "2025-01-20", "label": "POTUS Inauguration", "category": "ELECTION", "priority": 1},
        {"date": "2025-04-02", "label": "US reciprocal tariffs", "category": "TARIFFS", "priority": 1},
        {"date": "2025-04-28", "label": "Canada federal election", "category": "ELECTION", "priority": 1},
        {"date": "2025-06-04", "label": "US metals tariffs 50%", "category": "TARIFFS", "priority": 1},
        {"date": "2025-09-17", "label": "FOMC cut: 4.00–4.25%", "category": "RATES", "priority": 1},
        {"date": "2025-10-29", "label": "FOMC cut: 3.75–4.00%", "category": "RATES", "priority": 1},
        {"date": "2025-11-10", "label": "Aluminum premium spike", "category": "SUPPLY_CHAIN", "priority": 2},
        {"date": "2025-12-11", "label": "FOMC cut: 3.50–3.75%", "category": "RATES", "priority": 1},
        {"date": "2025-12-15", "label": "AI HBM/DRAM tightness", "category": "SUPPLY_CHAIN", "priority": 2},
    ]
    
    yaml_path = Path(__file__).parent.parent.parent / "reports" / "macro_events_2025.yaml"
    
    if not yaml_path.exists():
        return default_events
    
    # Try PyYAML first
    if YAML_AVAILABLE:
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                events = data.get('events', [])
                if events:
                    # Validate and convert dates
                    validated = []
                    for e in events:
                        if 'date' in e and 'label' in e:
                            validated.append({
                                'date': str(e['date']),
                                'label': str(e['label']),
                                'category': str(e.get('category', 'RATES')),
                                'priority': int(e.get('priority', 1))
                            })
                    if validated:
                        return validated
        except Exception as e:
            print(f"  Warning: YAML parse failed ({e}), using fallback parser")
    
    # Fallback: simple line-by-line parser
    try:
        with open(yaml_path, 'r') as f:
            content = f.read()
        events = _parse_yaml_fallback(content)
        if events:
            return events
    except Exception as e:
        print(f"  Warning: Fallback YAML parse failed ({e}), using hardcoded defaults")
    
    # Final fallback: hardcoded defaults
    return default_events


def _place_event_labels_no_overlap(ax, events_by_category, lanes, renderer, min_gap_days=28, max_levels=4, max_label_len=24):
    """
    Place event labels with deterministic lane-based collision avoidance.
    Uses date-based spacing (min_gap_days) to prevent overlap.
    
    Args:
        ax: Matplotlib axes (event rail)
        events_by_category: Dict {category: [(date, y_pos, label, color, category), ...]}
        lanes: Dict {category: base_y_position}
        renderer: Figure renderer (from fig.canvas.get_renderer())
        min_gap_days: Minimum days between events in same lane+level (default 28)
        max_levels: Maximum stack levels per lane (default 4)
        max_label_len: Maximum label length before truncation
    
    Returns:
        List of text_annotation objects that were successfully placed
    """
    placed_labels = []
    # Track last placed date and right edge per lane+level: {(category, level): (last_date, right_edge_px)}
    lane_level_last_info = {}
    
    # Stack offsets (vertical offsets within each lane, in data coordinates)
    stack_offsets = [0.00, -0.10, -0.20, -0.30]  # Up to 4 levels
    
    fig = ax.figure
    
    for category, events in events_by_category.items():
        if not events:
            continue
        
        base_y = lanes.get(category, 0.5)
        
        for event_date, y_pos, label, color, cat in events:
            # Truncate label if too long (with wrapping support)
            display_label = label
            if len(label) > max_label_len:
                # Try to wrap at word boundary
                words = label.split()
                wrapped = []
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) <= max_label_len:
                        current_line = current_line + " " + word if current_line else word
                    else:
                        if current_line:
                            wrapped.append(current_line)
                        current_line = word
                if current_line:
                    wrapped.append(current_line)
                if len(wrapped) > 2:
                    # Too long, truncate
                    display_label = label[:max_label_len-3] + "..."
                else:
                    display_label = "\n".join(wrapped[:2])  # Max 2 lines
            
            # Try each stack level (deterministic: check date spacing)
            placed = False
            temp_text = None
            
            for level in range(max_levels):
                key = (category, level)
                last_info = lane_level_last_info.get(key, None)
                last_date = last_info[0] if last_info else None
                
                # Check if enough days have passed (or first event in this lane+level)
                if last_date is None or (event_date - last_date).days >= min_gap_days:
                    stack_y = base_y + stack_offsets[level]
                    
                    # Create temporary text to measure bbox
                    # Use 'left' alignment to prevent labels from extending backward in time
                    temp_text = ax.text(
                        event_date, stack_y, display_label,
                        fontsize=8, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=color, lw=1, alpha=0.95),
                        clip_on=False, zorder=4, visible=False
                    )
                    
                    # Get bbox in display coordinates for final check
                    fig.canvas.draw()
                    bbox = temp_text.get_window_extent(renderer)
                    
                    # Additional pixel-based check (min 15px gap)
                    if level == 0 or last_date is None:
                        # First in level, always place
                        temp_text.set_visible(True)
                        placed_labels.append(temp_text)
                        # Store both date and right edge of bbox
                        lane_level_last_info[key] = (event_date, bbox.x1)
                        placed = True
                        temp_text = None
                        break
                    else:
                        # Check pixel overlap with previous label in same level
                        # Since labels are left-aligned, check if current label's left edge
                        # is far enough from previous label's right edge
                        last_right_edge = last_info[1] if last_info else None
                        if last_right_edge is None:
                            # Fallback: estimate from date
                            last_x_display = ax.transData.transform((mdates.date2num(last_date), stack_y))[0]
                            estimated_label_width = 100
                            last_right_edge = last_x_display + estimated_label_width
                        
                        current_x_display = bbox.x0
                        
                        if current_x_display >= last_right_edge + 15:  # 15px minimum gap between labels
                            temp_text.set_visible(True)
                            placed_labels.append(temp_text)
                            # Store both date and right edge of bbox
                            lane_level_last_info[key] = (event_date, bbox.x1)
                            placed = True
                            temp_text = None
                            break
                        else:
                            # Still too close, remove and try next level
                            temp_text.remove()
                            temp_text = None
            
            if not placed:
                # Try truncating more aggressively and force on level 0
                if len(label) > 15:
                    display_label = label[:12] + "..."
                    stack_y = base_y + stack_offsets[0]
                    key = (category, 0)
                    temp_text = ax.text(
                        event_date, stack_y, display_label,
                        fontsize=8, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=color, lw=1, alpha=0.95),
                        clip_on=False, zorder=4, visible=False
                    )
                    fig.canvas.draw()
                    bbox_fallback = temp_text.get_window_extent(renderer)
                    temp_text.set_visible(True)
                    placed_labels.append(temp_text)
                    lane_level_last_info[key] = (event_date, bbox_fallback.x1)
                # If still can't place, marker-only (already drawn)
    
    return placed_labels


def plot_macro_events_overlay_2025(engine: Engine) -> Path:
    """Create publication-grade 2025 overlay chart with collision-proof event rail."""
    # Load actual values from model_dataset_weekly
    actual_df = pd.read_sql(
        text("""
            SELECT 
                week_start,
                AVG(dcredit_proxy) as mean_actual
            FROM model_dataset_weekly
            WHERE week_start >= '2025-01-01'
            AND week_start <= '2025-12-31'
            AND dcredit_proxy IS NOT NULL
            GROUP BY week_start
            ORDER BY week_start
        """),
        engine
    )
    
    # Load predictions from model_predictions_weekly
    pred_df = pd.read_sql(
        text("""
            SELECT 
                p.week_start,
                AVG(p.y_pred) as mean_predicted
            FROM model_predictions_weekly p
            WHERE p.week_start >= '2025-01-01'
            AND p.week_start <= '2025-12-31'
            AND p.split = 'test'
            AND p.y_pred IS NOT NULL
            GROUP BY p.week_start
            ORDER BY p.week_start
        """),
        engine
    )
    
    if actual_df.empty or pred_df.empty:
        print("  Warning: No data found for macro events overlay chart")
        return None
    
    # Convert to datetime
    actual_df["week_start"] = pd.to_datetime(actual_df["week_start"])
    pred_df["week_start"] = pd.to_datetime(pred_df["week_start"])
    
    # Merge
    df = actual_df.merge(pred_df, on="week_start", how="outer").sort_values("week_start")
    
    # Load macro events (always returns list, never empty)
    events = load_macro_events_yaml()
    
    # Category colors and markers (SUPPLY_CHAIN removed)
    category_colors = {
        "RATES": COLORS['red'],
        "TARIFFS": COLORS['darkorange'],
        "ELECTION": COLORS['green'],
    }
    category_markers = {
        "RATES": "|",
        "TARIFFS": "|",
        "ELECTION": "|",
    }
    
    # Lane y-positions (spaced evenly, 3 lanes only)
    lanes = {
        "RATES": 0.78,
        "TARIFFS": 0.56,
        "ELECTION": 0.34,
    }
    
    # Create two-panel figure with proper ratios
    fig = plt.figure(figsize=(16, 7), dpi=150)
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[5.5, 1.4], hspace=0.05, figure=fig)
    
    # Main plot
    ax_main = fig.add_subplot(gs[0])
    
    # Plot lines
    ax_main.plot(df["week_start"], df["mean_actual"], 
                linewidth=2, color=COLORS['steelblue'], label='Actual', zorder=2)
    ax_main.plot(df["week_start"], df["mean_predicted"], 
                linewidth=2, color=COLORS['darkorange'], label='Predicted', linestyle='--', zorder=2)
    
    # Process events: priority=1 get subtle vertical lines in main plot
    priority1_events = [e for e in events if e.get('priority', 1) == 1]
    for event in priority1_events:
        if 'date' in event:
            event_date = pd.to_datetime(event['date'])
            if df["week_start"].min() <= event_date <= df["week_start"].max():
                category = event.get('category', 'RATES')
                color = category_colors.get(category, COLORS['gray'])
                ax_main.axvline(event_date, color=color, linestyle=':', lw=1.2, alpha=0.35, zorder=0)
    
    # Format main plot
    format_date_axis(ax_main, min_tick="monthly", rotation=30)
    ax_main.set_ylabel('Δ PD proxy (weekly change, unitless probability)', fontsize=10)
    ax_main.set_xlabel('')  # Remove xlabel (will be on event rail)
    ax_main.legend(loc='upper left', fontsize=9, framealpha=0.8)
    ax_main.grid(True, alpha=0.25)
    ax_main.tick_params(labelbottom=False)  # Hide x-axis labels on main plot
    
    # Event rail (bottom subplot)
    ax_rail = fig.add_subplot(gs[1], sharex=ax_main)
    
    # Filter events to date range and group by category
    events_in_range = []
    for event in events:
        if 'date' not in event:
            continue
        event_date = pd.to_datetime(event['date'])
        if df["week_start"].min() <= event_date <= df["week_start"].max():
            events_in_range.append(event)
    
    # Group events by category and sort by date (filter out SUPPLY_CHAIN)
    events_by_category = {}
    priority2_events = []
    
    for event in events_in_range:
        category = event.get('category', 'RATES')
        # Skip SUPPLY_CHAIN events
        if category == 'SUPPLY_CHAIN':
            continue
        priority = event.get('priority', 1)
        event_date = pd.to_datetime(event['date'])
        label = event.get('label', 'Event')
        color = category_colors.get(category, COLORS['gray'])
        marker = category_markers.get(category, '|')
        y_pos = lanes.get(category, 0.5)
        
        if priority == 1:
            if category not in events_by_category:
                events_by_category[category] = []
            events_by_category[category].append((event_date, y_pos, label, color, category))
        elif priority == 2:
            priority2_events.append((event_date, y_pos, label, color, category, marker))
    
    # Sort events within each category by date
    for category in events_by_category:
        events_by_category[category].sort(key=lambda x: x[0])
    
    # Draw markers for all events (priority 1 and 2)
    for category, event_list in events_by_category.items():
        for event_date, y_pos, label, color, cat in event_list:
            marker = category_markers.get(category, '|')
            ax_rail.scatter([event_date], [y_pos], marker=marker, s=80, color=color, zorder=3, clip_on=False)
    
    for event_date, y_pos, label, color, category, marker in priority2_events:
        ax_rail.scatter([event_date], [y_pos], marker=marker, s=50, color=color, alpha=0.6, zorder=2, clip_on=False)
    
    # Get renderer for collision detection (must draw first)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    
    # Place labels with deterministic lane-based collision detection
    placed_labels = _place_event_labels_no_overlap(ax_rail, events_by_category, lanes, renderer, min_gap_days=28, max_levels=4)
    
    # Format event rail
    ax_rail.set_ylim(0, 1)
    ax_rail.set_yticks([])
    ax_rail.set_ylabel('')
    ax_rail.spines['top'].set_visible(False)
    ax_rail.spines['left'].set_visible(False)
    ax_rail.spines['right'].set_visible(False)
    ax_rail.set_xlabel('Month (2025)', fontsize=10)
    format_date_axis(ax_rail, min_tick="monthly", rotation=30)
    
    # Add lane headers (category names) on the left, outside axes
    for category, y_pos in lanes.items():
        color = category_colors.get(category, COLORS['gray'])
        ax_rail.text(-0.02, y_pos, category, transform=ax_rail.get_yaxis_transform(),
                    ha='right', va='center', fontsize=8, color=color, fontweight='bold',
                    clip_on=False)
    
    # Category legend removed - lane headers on the left are sufficient
    
    # Show note if no events
    if not events_in_range:
        ax_rail.text(0.5, 0.5, "No macro events loaded", 
                    ha='center', va='center', fontsize=9, 
                    style='italic', alpha=0.5, transform=ax_rail.transAxes)
    
    # Title and subtitle
    add_title_subtitle(
        fig,
        "2025 Credit Risk Proxy: Actual vs Predicted with Macro Events",
        "Major 2025 Macro Catalysts (policy, rates). Event rail shows priority-1 labels; priority-2 shown as markers."
    )
    
    # Source footer
    source_text = "Source: SEC EDGAR (fundamentals), Stooq (equities). Macro events: YAML (reports/macro_events_2025.yaml)."
    add_source(fig, source_text)
    
    # Final spacing adjustments
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.12)
    
    output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "macro_events_overlay_2025.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    
    print(f"  Saved: {output_path.name}")
    return output_path


def plot_fragility_pillars_heatmap(engine: Engine) -> Path:
    """Create heatmap of issuer x pillars (P1-P5 ordered)."""
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
        engine
    )
    
    if df.empty:
        print("  Warning: No pillar scores found for heatmap")
        return None
    
    # Pivot to matrix
    heatmap_df = df.pivot(index='ticker', columns='pillar_name', values='score_0_100')
    
    # Sort by total score
    totals_df = pd.read_sql(
        text("""
            SELECT i.ticker, fs.total_score_0_100
            FROM fragility_scores fs
            JOIN dim_issuer i ON fs.issuer_id = i.issuer_id
            WHERE fs.asof_date = (SELECT MAX(asof_date) FROM fragility_scores)
            ORDER BY fs.total_score_0_100 DESC
        """),
        engine
    )
    
    if not totals_df.empty:
        heatmap_df = heatmap_df.loc[totals_df["ticker"]]
    
    # Reorder columns P1-P5
    pillar_order = ['refinancing_wall', 'cash_generation', 'leverage_coverage', 'cyclicality_ai', 'structure_opacity']
    existing_pillars = [p for p in pillar_order if p in heatmap_df.columns]
    heatmap_df = heatmap_df[existing_pillars]
    
    # Rename columns for display (shorter labels to prevent overlap)
    pillar_labels = {
        'refinancing_wall': 'P1: Refin.',
        'cash_generation': 'P2: Cash',
        'leverage_coverage': 'P3: Leverage',
        'cyclicality_ai': 'P4: Cyclic',
        'structure_opacity': 'P5: Struct'
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
    # Use horizontal labels for 5 pillars (no rotation needed, prevents overlap)
    ax.set_xticklabels(heatmap_df.columns, rotation=0, ha='center', fontsize=9)
    
    # Colorbar with proper spacing
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('Pillar Score (0-100)', fontsize=10, fontweight='bold')
    
    # Increased bottom margin to prevent label/axis title overlap with source footer
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.25, top=0.88)
    
    ax.set_xlabel("Pillar", fontsize=10, fontweight='bold')
    ax.set_ylabel("Issuer (Ticker)", fontsize=10, fontweight='bold')
    ax.grid(False)
    
    add_title_subtitle(
        fig,
        "Fragility Pillars Heatmap",
        "5-pillar breakdown per issuer (P1-P5). Red = higher risk. Grey = insufficient data."
    )
    
    add_source(fig, SOURCE_TEXT)
    
    output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "fragility_pillars_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    
    print(f"  Saved: {output_path.name}")
    return output_path


def plot_fragility_vs_predicted_chart(engine: Engine) -> Path:
    """Create scatter: fragility score vs predicted dcredit_proxy in RISK_OFF 2025."""
    df = pd.read_sql(
        text("""
            SELECT 
                i.ticker,
                fs.total_score_0_100 as fragility_score,
                AVG(p.y_pred) as avg_predicted_dpd
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
        engine
    )
    
    if df.empty:
        print("  Warning: No data found for fragility vs predicted chart")
        return None
    
    # Ensure numeric conversion
    df["avg_predicted_dpd"] = pd.to_numeric(df["avg_predicted_dpd"], errors="coerce")
    df["fragility_score"] = pd.to_numeric(df["fragility_score"], errors="coerce")
    df = df.dropna(subset=["avg_predicted_dpd", "fragility_score"])
    
    if df.empty:
        print("  Warning: No valid numeric data after conversion")
        return None
    
    # Calculate correlation
    corr = df["fragility_score"].corr(df["avg_predicted_dpd"])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(df["fragility_score"], df["avg_predicted_dpd"], 
              alpha=0.6, s=80, edgecolors='black', linewidth=0.5, color=COLORS['steelblue'])
    
    # Create absolute column and use it in nlargest (handle edge case: fewer than 5 rows)
    df2 = df.assign(abs_predicted_dpd=df["avg_predicted_dpd"].abs())
    n_top = min(5, len(df2))
    df_sorted_dpd_abs = df2.nlargest(n_top, "abs_predicted_dpd") if n_top > 0 else pd.DataFrame()
    
    # Also compute top 5 by fragility (handle edge case)
    n_top_frag = min(5, len(df))
    df_sorted_frag = df.nlargest(n_top_frag, "fragility_score") if n_top_frag > 0 else pd.DataFrame()
    
    # Build a union set of tickers to label
    tickers_dpd = set(df_sorted_dpd_abs["ticker"]) if not df_sorted_dpd_abs.empty else set()
    tickers_frag = set(df_sorted_frag["ticker"]) if not df_sorted_frag.empty else set()
    label_tickers = sorted(tickers_frag.union(tickers_dpd))
    
    # Add ticker labels with deterministic offsets to avoid collisions
    offsets = [(6, 6), (6, -8), (-14, 6), (-14, -8), (10, 0), (-18, 0), (0, 10), (0, -12), (14, 10), (-22, 10)]
    
    # Filter df to only rows with tickers in label_tickers
    if label_tickers:
        labels_df = df[df["ticker"].isin(label_tickers)].copy()
        
        for idx, ticker in enumerate(label_tickers):
            ticker_rows = labels_df[labels_df["ticker"] == ticker]
            if not ticker_rows.empty:
                row = ticker_rows.iloc[0]
                offset = offsets[idx % len(offsets)]
                ax.annotate(
                    ticker, 
                    (row["fragility_score"], row["avg_predicted_dpd"]),
                    fontsize=8, alpha=0.8, 
                    xytext=offset, textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
                )
    
    ax.set_xlabel("Fragility Score (0-100)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Avg predicted Δ PD proxy (weekly change, unitless probability)", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25)
    
    add_title_subtitle(
        fig,
        "Fragility Score vs Predicted ΔPD (RISK_OFF 2025)",
        f"Correlation: {corr:.3f}. Higher fragility should predict higher deterioration. Labels: top-5 fragility ∪ top-5 |ΔPD|."
    )
    
    add_source(fig, SOURCE_TEXT)
    
    output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "fragility_vs_predicted_deterioration.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    
    print(f"  Saved: {output_path.name}")
    return output_path


def plot_project_logic_diagram() -> Path:
    """Draw a simple architecture/logic diagram using matplotlib patches."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Hide axes
    ax.axis('off')
    
    # Define box positions (left to right flow)
    box_width = 1.2
    box_height = 0.8
    x_start = 0.5
    y_center = 0.5
    x_spacing = 2.0
    
    boxes = [
        (x_start, y_center, "EDGAR\nIngestion", ["company_tickers.json", "submissions index", "companyfacts fundamentals"]),
        (x_start + x_spacing, y_center, "Equity\nIngestion", ["prices (Stooq)"]),
        (x_start + 2*x_spacing, y_center, "FRED Macro\n(optional)", ["VIX, DGS2, DGS10, OAS", "(currently missing)"]),
        (x_start + 3*x_spacing, y_center, "Weekly Feature\nStore", ["feat_market_weekly", "feat_issuer_weekly", "(equity + fundamentals lags)"]),
        (x_start + 4*x_spacing, y_center, "Target\nConstruction", ["credit proxy", "(Merton-style PD proxy)", "dcredit_proxy weekly change"]),
        (x_start + 5*x_spacing, y_center, "Regime\nLabeling", ["risk_on/risk_off", "(currently equity-only", "fallback QQQ + SMH)"]),
        (x_start + 6*x_spacing, y_center, "Model\nTraining", ["regime-gated XGBoost", "predictions saved", "feature importance saved"]),
        (x_start + 7*x_spacing, y_center, "Reporting", ["backtests, residuals", "heatmaps, macro overlay", "vol surface"]),
        (x_start + 8*x_spacing, y_center, "Fragility +\nScenarios", ["fragility pillars", "scenario shocks", "contagion narrative"]),
    ]
    
    # Draw boxes
    box_patches = []
    for x, y, title, details in boxes:
        # Main box
        box = mpatches.FancyBboxPatch(
            (x - box_width/2, y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor='lightblue',
            edgecolor='black',
            linewidth=1.5,
            alpha=0.7
        )
        ax.add_patch(box)
        box_patches.append((x, y, title, details))
        
        # Title text
        ax.text(x, y + 0.25, title, ha='center', va='center', 
               fontsize=10, fontweight='bold')
        
        # Details text (smaller, below title)
        detail_text = '\n'.join(details)
        ax.text(x, y - 0.15, detail_text, ha='center', va='top',
               fontsize=7)
    
    # Draw arrows between boxes
    arrow_style = mpatches.ArrowStyle('->', head_length=0.3, head_width=0.2)
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + box_width/2
        x2 = boxes[i+1][0] - box_width/2
        y = boxes[i][1]
        
        arrow = mpatches.FancyArrowPatch(
            (x1, y), (x2, y),
            arrowstyle=arrow_style,
            color='black',
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(arrow)
    
    # Special styling for FRED Macro (optional/dotted)
    fred_idx = 2
    fred_x, fred_y, _, _ = boxes[fred_idx]
    # Make FRED box dotted border
    fred_box = mpatches.FancyBboxPatch(
        (fred_x - box_width/2, fred_y - box_height/2),
        box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor='lightyellow',
        edgecolor='gray',
        linewidth=1.5,
        linestyle='--',
        alpha=0.7
    )
    ax.add_patch(fred_box)
    
    # Set limits
    ax.set_xlim(0, x_start + 9*x_spacing)
    ax.set_ylim(0, 1)
    
    add_title_subtitle(
        fig,
        "AI Credit Crisis Case Study: End-to-End Pipeline Logic",
        "Data flow from EDGAR/equity ingestion through feature engineering, regime-gated modeling, to reporting and scenario analysis"
    )
    
    add_source(fig, "Source: Project architecture. EDGAR-first schema. Macro FRED optional.")
    
    output_path = Path(__file__).parent.parent.parent / "reports" / "figures" / "project_logic_diagram.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    
    print(f"  Saved: {output_path.name}")
    return output_path

