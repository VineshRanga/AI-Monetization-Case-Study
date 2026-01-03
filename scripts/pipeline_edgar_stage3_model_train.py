#!/usr/bin/env python3
"""Master pipeline: Stage 3 - Model training."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    load_dotenv()
    
    print("="*80)
    print("PIPELINE STAGE 3: Model Training")
    print("="*80)
    
    # Step 0: Run migrations (ensure schema is up to date)
    print("\n[0/4] Running migrations...")
    try:
        from src.db.db import get_engine
        from pathlib import Path
        
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
                    print(f"\n  Migration {migration_file.name} failed. Stopping.")
                    sys.exit(1)
            
            print(f"  ✓ Completed {len(migration_files)} migration(s)")
    except Exception as e:
        print(f"  ✗ Error running migrations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 1: Train RISK_ON model
    print("\n[1/4] Training RISK_ON model...")
    try:
        from src.models.train_credit_proxy_xgb import train_regime_model
        result = train_regime_model("RISK_ON")
        if result:
            print(f"  ✓ RISK_ON model trained (ID: {result.get('model_id')})")
            print(f"    Test RMSE: {result.get('metrics', {}).get('test_rmse', 0):.4f}, MAE: {result.get('metrics', {}).get('test_mae', 0):.4f}, R2: {result.get('metrics', {}).get('test_r2', 0):.4f}")
        else:
            print("  ⚠ No RISK_ON model trained (insufficient data)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Train RISK_OFF model
    print("\n[2/4] Training RISK_OFF model...")
    try:
        from src.models.train_credit_proxy_xgb import train_regime_model
        result = train_regime_model("RISK_OFF")
        if result:
            print(f"  ✓ RISK_OFF model trained (ID: {result.get('model_id')})")
            print(f"    Test RMSE: {result.get('metrics', {}).get('test_rmse', 0):.4f}, MAE: {result.get('metrics', {}).get('test_mae', 0):.4f}, R2: {result.get('metrics', {}).get('test_r2', 0):.4f}")
        else:
            print("  ⚠ No RISK_OFF model trained (insufficient data)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Compute SHAP explanations
    print("\n[3/4] Computing SHAP explanations...")
    try:
        from src.models.shap_explain import compute_shap_for_models
        result = compute_shap_for_models()
        print(f"  ✓ SHAP computation attempted ({result.get('rows_created', 0)} records)")
        if result.get('rows_created', 0) == 0:
            print("  ⚠ Note: SHAP requires saved model objects. Using feature importance fallback.")
    except Exception as e:
        print(f"  ⚠ Warning: SHAP computation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 4: Generate model charts
    print("\n[4/4] Generating model charts...")
    try:
        from src.viz.make_model_charts import (
            make_backtest_chart, make_residuals_chart, make_predicted_vs_actual_scatter,
            make_shap_summary_chart, make_regime_timeline_chart, make_issuer_risk_heatmap
        )
        chart_paths = []
        
        path = make_backtest_chart()
        if path:
            chart_paths.append(path)
        
        path = make_residuals_chart()
        if path:
            chart_paths.append(path)
        
        path = make_predicted_vs_actual_scatter()
        if path:
            chart_paths.append(path)
        
        path = make_shap_summary_chart()
        if path:
            chart_paths.append(path)
        else:
            print("  ⚠ Feature importance chart skipped (no data available)")
        
        path = make_regime_timeline_chart()
        if path:
            chart_paths.append(path)
        
        path = make_issuer_risk_heatmap()
        if path:
            chart_paths.append(path)
        
        print(f"  ✓ Generated {len(chart_paths)} chart(s) in reports/figures/")
    except Exception as e:
        print(f"  ⚠ Warning: Chart generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("PIPELINE STAGE 3 COMPLETE!")
    print("="*80)

