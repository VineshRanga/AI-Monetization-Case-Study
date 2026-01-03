"""Generate model evaluation charts (director-level quant quality)."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sqlalchemy import text

from src.db.db import get_engine
from src.reports.plot_style import (
    apply_mpl_style, add_title_subtitle, add_source_footnote, save_fig,
    format_date_axis_robust, shade_regime_contiguous, FOOTNOTE_TEXT
)

# Apply global style
apply_mpl_style()


def make_backtest_chart() -> Path:
    """Create actual vs predicted chart for backtest (weekly aggregate with regime shading)."""
    engine = get_engine()
    
    try:
        # Get predictions for 2025 test period only
        with engine.connect() as conn:
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
                conn
            )
            
            # Get regime data for shading
            regime_df = pd.read_sql(
                text("""
                    SELECT week_start, regime_label
                    FROM model_regime_weekly
                    WHERE week_start >= '2025-01-01'
                    AND week_start <= '2025-12-31'
                    ORDER BY week_start
                """),
                conn
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
        
        # Count issuers per week
        issuer_counts = pred_df.groupby("week_start")["y_true"].count().reset_index()
        issuer_counts.columns = ["week_start", "issuer_count"]
        weekly_agg = weekly_agg.merge(issuer_counts, on="week_start", how="left")
        
        # Convert week_start to datetime if needed
        weekly_agg["week_start"] = pd.to_datetime(weekly_agg["week_start"])
        if not regime_df.empty:
            regime_df["week_start"] = pd.to_datetime(regime_df["week_start"])
        
        # Calculate metrics
        mae = np.mean(np.abs(weekly_agg["y_pred"] - weekly_agg["y_true"]))
        rmse = np.sqrt(np.mean((weekly_agg["y_pred"] - weekly_agg["y_true"])**2))
        n_weeks = len(weekly_agg)
        n_issuers = int(weekly_agg["issuer_count"].max()) if "issuer_count" in weekly_agg.columns else 0
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Shade RISK_OFF periods as contiguous blocks
        if not regime_df.empty:
            shade_regime_contiguous(ax, regime_df, week_col='week_start', 
                                   regime_col='regime_label', regime_value='RISK_OFF')
        
        # Plot actual and predicted (mean)
        ax.plot(weekly_agg["week_start"], weekly_agg["y_true"], 
                marker='o', label='Actual (mean)', linewidth=2, markersize=4, color='steelblue', zorder=3)
        ax.plot(weekly_agg["week_start"], weekly_agg["y_pred"], 
                marker='x', label='Predicted (mean)', linewidth=2, markersize=4, linestyle='--', color='darkorange', zorder=3)
        
        # Add zero line
        ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
        
        # Format date axis (monthly ticks, not weekly)
        format_date_axis_robust(ax, rotation=45, interval_months=3)
        
        # Axis labels
        ax.set_xlabel('Week Start', fontsize=10, fontweight='bold')
        ax.set_ylabel('ΔPD proxy (mean across issuers)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.25)
        
        # Title and subtitle via helper (NO ax.set_title)
        add_title_subtitle(
            fig,
            "Backtest: Weekly Change in PD Proxy (2025)",
            f"Mean across {n_issuers} issuers. Test window: 2025. Regime: QQQ/SMH fallback."
        )
        
        # Metrics box (upper right, below subtitle)
        metrics_text = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nN: {n_weeks}"
        ax.text(0.98, 0.85, metrics_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
        
        # Footnote via helper
        add_source_footnote(fig, FOOTNOTE_TEXT)
        
        # Save
        output_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "backtest_actual_vs_pred.png"
        save_fig(fig, output_path)
        
        print(f"  Saved backtest chart to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"  Warning: Failed to generate backtest chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def make_residuals_chart() -> Path:
    """Create residuals histogram with robust axis limits."""
    engine = get_engine()
    
    try:
        with engine.connect() as conn:
            pred_df = pd.read_sql(
                text("""
                    SELECT y_true, y_pred
                    FROM model_predictions_weekly
                    WHERE split = 'test'
                    AND y_true IS NOT NULL
                    AND y_pred IS NOT NULL
                """),
                conn
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
        
        ax.hist(pred_df["residual"], bins=50, edgecolor='black', alpha=0.7, color='steelblue', range=(lo, hi))
        ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero residual', zorder=2)
        ax.axvline(mean_residual, color='g', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_residual:.4f}', zorder=2)
        
        # Set robust x-limits
        ax.set_xlim(lo, hi)
        
        # Axis labels
        ax.set_xlabel('Residual (Predicted − Actual ΔPD proxy)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.25)
        
        # Title and subtitle via helper
        add_title_subtitle(
            fig,
            "Residual Distribution (2025 Test)",
            "Predicted minus Actual ΔPD proxy. X-limits = 1-99% quantiles."
        )
        
        # Stats box (upper right)
        stats_text = f"Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}\nN: {n}"
        ax.text(0.98, 0.85, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend(fontsize=9, framealpha=0.9)
        
        # Footnote via helper
        add_source_footnote(fig, FOOTNOTE_TEXT)
        
        output_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "residuals_hist.png"
        save_fig(fig, output_path)
        
        print(f"  Saved residuals chart to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"  Warning: Failed to generate residuals chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def make_predicted_vs_actual_scatter() -> Path:
    """Create predicted vs actual scatter plot with robust axis limits."""
    engine = get_engine()
    
    try:
        with engine.connect() as conn:
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
                conn
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
                      edgecolors='black', linewidth=0.5, color='steelblue', label='RISK_ON', zorder=3)
        if not risk_off_df.empty:
            ax.scatter(risk_off_df["y_true"], risk_off_df["y_pred"], alpha=0.5, s=30, 
                      edgecolors='black', linewidth=0.5, color='darkorange', label='RISK_OFF', zorder=3)
        
        # Perfect prediction line (45-degree)
        ax.plot([-m, m], [-m, m], 'r--', linewidth=2, label='Perfect prediction', zorder=2)
        
        # Equal aspect and symmetric axis limits
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-m * 1.05, m * 1.05)
        ax.set_ylim(-m * 1.05, m * 1.05)
        
        # Axis labels
        ax.set_xlabel('Actual ΔPD proxy', fontsize=10, fontweight='bold')
        ax.set_ylabel('Predicted ΔPD proxy', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.25)
        
        # Title and subtitle via helper
        add_title_subtitle(
            fig,
            "Predicted vs Actual (2025 Test)",
            "Color-coded by regime. 45° line indicates perfect prediction."
        )
        
        # Metrics box (upper left, below subtitle)
        metrics_text = f"N: {n}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}"
        ax.text(0.02, 0.85, metrics_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
        
        # Footnote via helper
        add_source_footnote(fig, FOOTNOTE_TEXT)
        
        output_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "predicted_vs_actual_scatter.png"
        save_fig(fig, output_path)
        
        print(f"  Saved scatter chart to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"  Warning: Failed to generate scatter chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def make_shap_summary_chart() -> Path:
    """Create feature importance summary chart (top 15 features)."""
    engine = get_engine()
    
    try:
        with engine.connect() as conn:
            # Find the latest RISK_OFF model_id
            model_result = conn.execute(
                text("""
                    SELECT model_id
                    FROM model_registry
                    WHERE regime = 'RISK_OFF' AND model_name = 'xgb_credit_risk_off_v1'
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
            )
            model_row = model_result.fetchone()
            
            if model_row is None:
                print("  Warning: No RISK_OFF model found in model_registry")
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.text(0.5, 0.5, 'No feature importance available.\nRe-run Stage 3 to populate model_feature_importance.', 
                       transform=ax.transAxes, fontsize=12, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                add_title_subtitle(fig, "Top Risk Drivers in RISK_OFF Regime")
                ax.axis('off')
                add_source_footnote(fig, FOOTNOTE_TEXT)
                output_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "shap_summary_bar.png"
                save_fig(fig, output_path)
                return output_path
            
            model_id = model_row[0]
            
            # Load feature importances
            importance_df = pd.read_sql(
                text("""
                    SELECT feature, 
                           COALESCE(importance_norm, importance_raw, importance) as importance_val,
                           importance_raw,
                           importance_norm
                    FROM model_feature_importance
                    WHERE model_id = :model_id
                    ORDER BY COALESCE(importance_norm, importance_raw, importance) DESC
                    LIMIT 15
                """),
                conn,
                params={"model_id": model_id}
            )
        
        if importance_df.empty:
            print("  Warning: No feature importance data found for RISK_OFF model")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No feature importance found.\nRe-run Stage 3 to populate model_feature_importance.', 
                   transform=ax.transAxes, fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            add_title_subtitle(fig, "Top Risk Drivers in RISK_OFF Regime")
            ax.axis('off')
            add_source_footnote(fig, FOOTNOTE_TEXT)
            output_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "shap_summary_bar.png"
            save_fig(fig, output_path)
            return output_path
        
        # Normalize to percentage (ensure sum = 100)
        if "importance_norm" in importance_df.columns and importance_df["importance_norm"].notna().any():
            total = importance_df["importance_norm"].sum()
            if total > 0:
                importance_df["importance_pct"] = (importance_df["importance_norm"] / total) * 100
            else:
                importance_df["importance_pct"] = 0.0
        else:
            total = importance_df["importance_val"].sum()
            if total > 0:
                importance_df["importance_pct"] = (importance_df["importance_val"] / total) * 100
            else:
                importance_df["importance_pct"] = 0.0
        
        # Extract features and importances (as percentage)
        features = importance_df["feature"].tolist()
        importances_pct = importance_df["importance_pct"].tolist()
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(features)), importances_pct, color='steelblue')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.invert_yaxis()
        
        # Format x-axis as percentage
        ax.set_xlabel('% of Total Importance', fontsize=10, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.25, axis='x')
        
        # Set x-ticks to round numbers (0, 5, 10, ...)
        max_pct = max(importances_pct) if importances_pct else 10
        ax.set_xticks(np.arange(0, max_pct + 5, 5))
        
        # Title and subtitle via helper (NO ax.set_title)
        add_title_subtitle(
            fig,
            "Top Risk Drivers in RISK_OFF Regime",
            "XGBoost feature importance (normalized)."
        )
        
        # Increase left margin for feature names
        fig.subplots_adjust(left=0.25)
        
        # Footnote via helper
        add_source_footnote(fig, FOOTNOTE_TEXT)
        
        output_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "shap_summary_bar.png"
        save_fig(fig, output_path)
        
        print(f"  Saved feature importance chart to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"  Warning: Failed to generate feature importance chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def make_regime_timeline_chart() -> Path:
    """Create regime probability timeline (2020-2025) with contiguous shading."""
    engine = get_engine()
    
    try:
        with engine.connect() as conn:
            regime_df = pd.read_sql(
                text("""
                    SELECT week_start, prob_risk_off, regime_label
                    FROM model_regime_weekly
                    WHERE week_start >= '2020-01-01'
                    AND week_start <= '2025-12-31'
                    ORDER BY week_start
                """),
                conn
            )
        
        if regime_df.empty:
            print("  Warning: No regime data found for timeline chart")
            return None
        
        # Convert to datetime
        regime_df["week_start"] = pd.to_datetime(regime_df["week_start"])
        
        # Compute 4-week rolling mean for readability
        regime_df["prob_risk_off_roll4"] = regime_df["prob_risk_off"].rolling(window=4, min_periods=1).mean()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Shade RISK_OFF periods as contiguous blocks (not stripes)
        shade_regime_contiguous(ax, regime_df, week_col='week_start', 
                               regime_col='regime_label', regime_value='RISK_OFF')
        
        # Plot probability (with rolling mean)
        ax.plot(regime_df["week_start"], regime_df["prob_risk_off"], 
               linewidth=1.5, color='steelblue', alpha=0.5, label='P(Risk-Off)', zorder=2)
        ax.plot(regime_df["week_start"], regime_df["prob_risk_off_roll4"], 
               linewidth=2, color='steelblue', label='P(Risk-Off, 4w rolling)', zorder=3)
        
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% threshold', zorder=1)
        
        # Set y-limits to [0, 1]
        ax.set_ylim(0, 1)
        
        # Format date axis (monthly ticks, not weekly)
        format_date_axis_robust(ax, rotation=45, interval_months=3)
        
        # Axis labels
        ax.set_xlabel('Week Start', fontsize=10, fontweight='bold')
        ax.set_ylabel('P(Risk-Off)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.25)
        
        # Title and subtitle via helper
        add_title_subtitle(
            fig,
            "Risk Regime Probability (2020-2025)",
            "Probability of RISK_OFF regime. Shaded regions indicate RISK_OFF periods. Regime: QQQ/SMH fallback."
        )
        
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        
        # Footnote via helper
        add_source_footnote(fig, FOOTNOTE_TEXT)
        
        output_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "regime_timeline_2020_2025.png"
        save_fig(fig, output_path)
        
        print(f"  Saved regime timeline chart to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"  Warning: Failed to generate regime timeline chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def make_issuer_risk_heatmap() -> Path:
    """Create issuer-level risk heatmap for 2025 with proper NaN handling."""
    engine = get_engine()
    
    try:
        with engine.connect() as conn:
            # Get predictions for 2025 test set
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
                conn
            )
        
        if pred_df.empty:
            print("  Warning: No test predictions found for heatmap")
            return None
        
        # Pivot to create heatmap matrix (mean per issuer-week if duplicates)
        heatmap_df = pred_df.groupby(['ticker', 'week_start'])['y_pred'].mean().reset_index()
        heatmap_df = heatmap_df.pivot(index='ticker', columns='week_start', values='y_pred')
        
        # Sort issuers by average predicted deterioration in 2025 test (descending)
        issuer_avg = heatmap_df.mean(axis=1).sort_values(ascending=False)
        heatmap_df = heatmap_df.loc[issuer_avg.index]
        
        # Get top 3 worst issuers
        top_3_worst = issuer_avg.head(3).index.tolist()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set colormap with "bad" color for NaN (grey, not white)
        cmap = plt.get_cmap("RdYlGn_r").copy()
        cmap.set_bad(color="lightgrey")
        
        # Mask NaNs explicitly
        masked_data = np.ma.masked_invalid(heatmap_df.values)
        
        im = ax.imshow(masked_data, aspect='auto', cmap=cmap, interpolation='nearest')
        
        # Set ticks
        ax.set_yticks(range(len(heatmap_df.index)))
        ax.set_yticklabels(heatmap_df.index, fontsize=9)
        
        # Format date labels (monthly only, not weekly)
        week_dates = [pd.to_datetime(str(d)) for d in heatmap_df.columns]
        # Get first week of each month
        month_starts = []
        month_labels = []
        last_month = None
        for i, d in enumerate(week_dates):
            month_key = (d.year, d.month)
            if month_key != last_month:
                month_starts.append(i)
                month_labels.append(d.strftime('%Y-%m'))
                last_month = month_key
        
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=8)
        
        # Add colorbar with proper label
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Predicted ΔPD proxy', fontsize=10, fontweight='bold')
        
        # Axis labels
        ax.set_xlabel('Week Start', fontsize=10, fontweight='bold')
        ax.set_ylabel('Issuer (Ticker)', fontsize=10, fontweight='bold')
        ax.grid(False)  # No grid for heatmap
        
        # Title and subtitle via helper
        add_title_subtitle(
            fig,
            "Issuer-Level Predicted Deterioration (2025 Test)",
            f"Sorted by average predicted deterioration. Top 3 worst: {', '.join(top_3_worst[:3])}. Grey = no prediction / insufficient data."
        )
        
        # Footnote via helper
        add_source_footnote(fig, FOOTNOTE_TEXT)
        
        output_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "issuer_risk_heatmap_2025.png"
        save_fig(fig, output_path)
        
        print(f"  Saved issuer risk heatmap to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"  Warning: Failed to generate issuer risk heatmap: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("Generating model charts...")
    make_backtest_chart()
    make_residuals_chart()
    make_predicted_vs_actual_scatter()
    make_shap_summary_chart()
    make_regime_timeline_chart()
    make_issuer_risk_heatmap()
    print("Done!")
