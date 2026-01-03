#!/usr/bin/env python3
"""
Generate all report figures with consistent director-level styling.

NOTE: Preferred execution method:
  python3 -m scripts.generate_report_figures

OR ensure PYTHONPATH includes project root:
  export PYTHONPATH=/path/to/AICreditRiskAnalysis:$PYTHONPATH
  python3 scripts/generate_report_figures.py
"""
import os
import sys

# Path bootstrap: add repo root to sys.path so 'src' imports work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pathlib import Path
from dotenv import load_dotenv

from src.db.db import get_engine
from src.reporting.figures import (
    plot_backtest_mean_timeseries,
    plot_issuer_heatmap_2025,
    plot_predicted_vs_actual_scatter,
    plot_regime_timeline,
    plot_residual_hist,
    plot_feature_importance_bar,
    plot_regime_vol_surface,
    plot_macro_events_overlay_2025,
    plot_project_logic_diagram,
    plot_fragility_pillars_heatmap,
    plot_fragility_vs_predicted_chart,
)

if __name__ == "__main__":
    load_dotenv()
    
    print("="*80)
    print("GENERATING REPORT FIGURES")
    print("="*80)
    
    engine = get_engine()
    
    # Generate all figures in fixed order
    figures = []
    
    print("\n[1/7] Backtest mean timeseries...")
    try:
        path = plot_backtest_mean_timeseries(engine)
        if path:
            figures.append(path.name)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[2/7] Issuer heatmap 2025...")
    try:
        path = plot_issuer_heatmap_2025(engine)
        if path:
            figures.append(path.name)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[3/7] Predicted vs actual scatter...")
    try:
        path = plot_predicted_vs_actual_scatter(engine)
        if path:
            figures.append(path.name)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[4/7] Regime timeline...")
    try:
        path = plot_regime_timeline(engine)
        if path:
            figures.append(path.name)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[5/7] Residual histogram...")
    try:
        path = plot_residual_hist(engine)
        if path:
            figures.append(path.name)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[6/7] Feature importance bar...")
    try:
        path = plot_feature_importance_bar(engine, regime="RISK_OFF")
        if path:
            figures.append(path.name)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[7/9] Regime volatility surface...")
    try:
        path = plot_regime_vol_surface(engine)
        if path:
            figures.append(path.name)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[8/9] Macro events overlay 2025...")
    try:
        path = plot_macro_events_overlay_2025(engine)
        if path:
            figures.append(path.name)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[9/11] Project logic diagram...")
    try:
        path = plot_project_logic_diagram()
        if path:
            figures.append(path.name)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[10/11] Fragility pillars heatmap...")
    try:
        path = plot_fragility_pillars_heatmap(engine)
        if path:
            figures.append(path.name)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[11/11] Fragility vs predicted chart...")
    try:
        path = plot_fragility_vs_predicted_chart(engine)
        if path:
            figures.append(path.name)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Print checklist
    print("\n" + "="*80)
    print("FIGURE GENERATION COMPLETE")
    print("="*80)
    print(f"\nGenerated {len(figures)} figure(s):")
    for i, fig_name in enumerate(figures, 1):
        print(f"  [{i}] ✓ {fig_name}")
    
    output_dir = Path(__file__).parent.parent / "reports" / "figures"
    print(f"\nAll figures saved to: {output_dir}")

