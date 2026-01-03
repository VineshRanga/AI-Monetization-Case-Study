#!/usr/bin/env python3
"""End-to-end script to train regime-gated XGBoost models."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.db import get_engine
from src.models.save_predictions import save_predictions
from src.models.shap_explain import explain_model
from src.models.train_spread_xgb import FEATURE_COLS, train_regime_model
from src.transform.build_model_dataset import build_model_dataset
from sqlalchemy import text

if __name__ == "__main__":
    load_dotenv()
    
    print("="*80)
    print("TRAINING REGIME-GATED XGBOOST MODELS")
    print("="*80)
    
    engine = get_engine()
    
    # Step 1: Run migrations
    print("\n[1/5] Running migrations...")
    try:
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
                    print(f"    ✗ {migration_file.name} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"\n  Migration {migration_file.name} failed. Stopping.")
                    sys.exit(1)
            print(f"  ✓ Completed {len(migration_files)} migration(s)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Build modeling dataset
    print("\n[2/5] Building modeling dataset...")
    try:
        result = build_model_dataset()
        print(f"  ✓ Created {result['rows_created']} rows")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Train RISK_ON model
    print("\n[3/5] Training RISK_ON model...")
    try:
        result_on = train_regime_model("RISK_ON")
        
        if result_on:
            model_id_on = result_on["model_id"]
            model_on = result_on["model"]
            train_df_on = result_on["train_df"]
            test_df_on = result_on["test_df"]
            
            # Save predictions
            if not train_df_on.empty:
                # Prepare features for prediction (same as training)
                from src.models.train_spread_xgb import compute_issuer_target_encoding
                
                X_train = train_df_on[FEATURE_COLS].copy()
                issuer_means = compute_issuer_target_encoding(train_df_on)
                X_train["issuer_mean_dspread"] = train_df_on["issuer_id"].map(issuer_means)
                global_mean = X_train["issuer_mean_dspread"].mean()
                X_train["issuer_mean_dspread"] = X_train["issuer_mean_dspread"].fillna(global_mean)
                
                if "bucket" in X_train.columns:
                    X_train["bucket"] = X_train["bucket"].astype("category")
                for col in X_train.columns:
                    if col != "bucket":
                        median_val = X_train[col].median()
                        X_train[col] = X_train[col].fillna(median_val)
                
                train_pred = model_on.predict(X_train)
                
                # Prepare test features
                test_pred = []
                if not test_df_on.empty:
                    X_test = test_df_on[FEATURE_COLS].copy()
                    X_test["issuer_mean_dspread"] = test_df_on["issuer_id"].map(issuer_means).fillna(global_mean)
                    if "bucket" in X_test.columns:
                        X_test["bucket"] = X_test["bucket"].astype("category")
                    for col in X_test.columns:
                        if col != "bucket":
                            X_test[col] = X_test[col].fillna(X_train[col].median())
                    test_pred = model_on.predict(X_test)
                
                save_predictions(
                    model_id_on,
                    train_df_on,
                    train_pred.tolist(),
                    test_df_on,
                    test_pred.tolist() if test_pred else [],
                    FEATURE_COLS,
                )
                print(f"  ✓ Saved predictions")
            
            # Compute SHAP
            explain_model(model_id_on, "RISK_ON", model_on, test_df_on)
            print(f"  ✓ Computed SHAP explanations")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 4: Train RISK_OFF model
    print("\n[4/5] Training RISK_OFF model...")
    try:
        result_off = train_regime_model("RISK_OFF")
        
        if result_off:
            model_id_off = result_off["model_id"]
            model_off = result_off["model"]
            train_df_off = result_off["train_df"]
            test_df_off = result_off["test_df"]
            
            # Save predictions
            if not train_df_off.empty:
                # Prepare features for prediction (same as training)
                from src.models.train_spread_xgb import compute_issuer_target_encoding
                
                X_train = train_df_off[FEATURE_COLS].copy()
                issuer_means = compute_issuer_target_encoding(train_df_off)
                X_train["issuer_mean_dspread"] = train_df_off["issuer_id"].map(issuer_means)
                global_mean = X_train["issuer_mean_dspread"].mean()
                X_train["issuer_mean_dspread"] = X_train["issuer_mean_dspread"].fillna(global_mean)
                
                if "bucket" in X_train.columns:
                    X_train["bucket"] = X_train["bucket"].astype("category")
                for col in X_train.columns:
                    if col != "bucket":
                        median_val = X_train[col].median()
                        X_train[col] = X_train[col].fillna(median_val)
                
                train_pred = model_off.predict(X_train)
                
                # Prepare test features
                test_pred = []
                if not test_df_off.empty:
                    X_test = test_df_off[FEATURE_COLS].copy()
                    X_test["issuer_mean_dspread"] = test_df_off["issuer_id"].map(issuer_means).fillna(global_mean)
                    if "bucket" in X_test.columns:
                        X_test["bucket"] = X_test["bucket"].astype("category")
                    for col in X_test.columns:
                        if col != "bucket":
                            X_test[col] = X_test[col].fillna(X_train[col].median())
                    test_pred = model_off.predict(X_test)
                
                save_predictions(
                    model_id_off,
                    train_df_off,
                    train_pred.tolist(),
                    test_df_off,
                    test_pred.tolist() if test_pred else [],
                    FEATURE_COLS,
                )
                print(f"  ✓ Saved predictions")
            
            # Compute SHAP
            explain_model(model_id_off, "RISK_OFF", model_off, test_df_off)
            print(f"  ✓ Computed SHAP explanations")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Print summary
    print("\n[5/5] Model Summary")
    print("="*80)
    
    with engine.connect() as conn:
        models = conn.execute(
            text("""
                SELECT model_id, name, regime_label, metrics
                FROM model_registry
                ORDER BY created_at DESC
                LIMIT 2
            """)
        ).fetchall()
        
        for model_id, name, regime, metrics_json in models:
            import json
            metrics = json.loads(metrics_json)
            print(f"\n{name} ({regime}):")
            if isinstance(metrics, dict):
                if "test_rmse" in metrics:
                    print(f"  Test RMSE: {metrics['test_rmse']:.2f}")
                if "test_mae" in metrics:
                    print(f"  Test MAE: {metrics['test_mae']:.2f}")
                if "test_dir_acc" in metrics:
                    print(f"  Test Directional Accuracy: {metrics['test_dir_acc']:.2%}")
                if "test_corr" in metrics:
                    print(f"  Test Correlation: {metrics['test_corr']:.3f}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)

