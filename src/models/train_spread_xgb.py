"""Train regime-gated XGBoost models for credit proxy change prediction."""
import json
import numpy as np
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db.db import get_engine, now_utc

# Feature list (stable)
FEATURE_COLS = [
    "eq_ret",
    "eq_vol_21d",
    "qqq_ret",
    "spx_ret",
    "vix_chg",
    "dgs2_chg",
    "dgs10_chg",
    "ig_oas_chg",
    "net_debt",
    "net_debt_ebitda",
    "int_coverage",
    "fcf_margin",
    "capex_intensity",
    "issuer_mean_target",
    "bucket",  # Categorical
]

# Hyperparameters
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 800,
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "random_state": 42,
}


def compute_issuer_target_encoding(train_df: pd.DataFrame) -> pd.Series:
    """
    Compute issuer mean dcredit_proxy for target encoding.
    
    Returns Series indexed by issuer_id.
    """
    return train_df.groupby("issuer_id")["dcredit_proxy"].mean()


def create_time_splits(
    df: pd.DataFrame,
    start_year: int = 2020,
    end_year: int = 2025,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create expanding window time-series splits.
    
    Returns list of (train_df, test_df) tuples.
    """
    splits = []
    
    # Define fold boundaries
    fold_boundaries = [
        (f"{start_year}-01-01", f"{start_year+1}-12-31", f"{start_year+2}-01-01", f"{start_year+2}-12-31"),  # Fold 1
        (f"{start_year}-01-01", f"{start_year+2}-12-31", f"{start_year+3}-01-01", f"{start_year+3}-12-31"),  # Fold 2
        (f"{start_year}-01-01", f"{start_year+3}-12-31", f"{start_year+4}-01-01", f"{start_year+4}-12-31"),  # Fold 3
        (f"{start_year}-01-01", f"{start_year+4}-12-31", f"{start_year+5}-01-01", f"{start_year+5}-12-31"),  # Fold 4
    ]
    
    for train_start, train_end, test_start, test_end in fold_boundaries:
        train_df = df[
            (df["week_start"] >= pd.to_datetime(train_start)) &
            (df["week_start"] <= pd.to_datetime(train_end))
        ].copy()
        
        test_df = df[
            (df["week_start"] >= pd.to_datetime(test_start)) &
            (df["week_start"] <= pd.to_datetime(test_end))
        ].copy()
        
        if len(train_df) > 50 and len(test_df) > 10:  # Minimum data requirements
            splits.append((train_df, test_df))
    
    return splits


def train_xgb_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_num: int,
    regime_label: str,
) -> Tuple[xgb.XGBRegressor, Dict]:
    """
    Train XGBoost model with target encoding.
    
    Returns (model, metrics_dict).
    """
    # Prepare features
    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df["dcredit_proxy"].values
    
    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df["dcredit_proxy"].values
    
    # Handle categorical bucket
    if "bucket" in X_train.columns:
        X_train["bucket"] = X_train["bucket"].astype("category")
        X_test["bucket"] = X_test["bucket"].astype("category")
    
    # Compute issuer target encoding on train only
    issuer_means = compute_issuer_target_encoding(train_df)
    
    # Fill target encoding
    X_train["issuer_mean_target"] = train_df["issuer_id"].map(issuer_means)
    X_test["issuer_mean_target"] = test_df["issuer_id"].map(issuer_means)
    
    # Fill missing with global mean
    global_mean = X_train["issuer_mean_target"].mean()
    X_train["issuer_mean_target"] = X_train["issuer_mean_target"].fillna(global_mean)
    X_test["issuer_mean_target"] = X_test["issuer_mean_target"].fillna(global_mean)
    
    # Fill other missing values with median
    for col in X_train.columns:
        if col != "bucket":
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
    
    # Create validation split (last 20% of training data)
    val_size = int(len(X_train) * 0.2)
    X_train_fit = X_train.iloc[:-val_size]
    y_train_fit = y_train[:-val_size]
    X_val = X_train.iloc[-val_size:]
    y_val = y_train[-val_size:]
    
    # Train model
    model = xgb.XGBRegressor(**XGB_PARAMS)
    
    model.fit(
        X_train_fit,
        y_train_fit,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Directional accuracy
    train_dir_acc = np.mean(np.sign(y_train) == np.sign(y_pred_train))
    test_dir_acc = np.mean(np.sign(y_test) == np.sign(y_pred_test))
    
    # Correlation
    train_corr = np.corrcoef(y_train, y_pred_train)[0, 1]
    test_corr = np.corrcoef(y_test, y_pred_test)[0, 1]
    
    metrics = {
        "fold": fold_num,
        "train_rmse": float(train_rmse),
        "train_mae": float(train_mae),
        "train_dir_acc": float(train_dir_acc),
        "train_corr": float(train_corr),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "test_dir_acc": float(test_dir_acc),
        "test_corr": float(test_corr),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    
    return model, metrics


def train_regime_model(regime_label: str) -> Dict:
    """Train model for a specific regime."""
    engine = get_engine()
    
    # Load data for regime
    with engine.connect() as conn:
        query = text("""
            SELECT *
            FROM model_dataset_weekly
            WHERE regime_label = :regime
            AND dcredit_proxy IS NOT NULL
            ORDER BY week_start
        """)
        
        df = pd.read_sql(query, conn, params={"regime": regime_label})
    
    if df.empty:
        print(f"No data found for regime {regime_label}")
        return {}
    
    print(f"\nTraining {regime_label} model...")
    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df['week_start'].min()} to {df['week_start'].max()}")
    
    # Create time splits
    splits = create_time_splits(df)
    print(f"  Created {len(splits)} time-series folds")
    
    if not splits:
        print("  Not enough data for time-series splits")
        return {}
    
    # Train on each fold
    all_metrics = []
    models = []
    
    for fold_num, (train_df, test_df) in enumerate(splits, 1):
        print(f"\n  Fold {fold_num}:")
        print(f"    Train: {train_df['week_start'].min()} to {train_df['week_start'].max()} ({len(train_df)} rows)")
        print(f"    Test: {test_df['week_start'].min()} to {test_df['week_start'].max()} ({len(test_df)} rows)")
        
        model, metrics = train_xgb_model(train_df, test_df, fold_num, regime_label)
        models.append(model)
        all_metrics.append(metrics)
        
        print(f"    Test RMSE: {metrics['test_rmse']:.2f}, MAE: {metrics['test_mae']:.2f}, Dir Acc: {metrics['test_dir_acc']:.2%}")
    
    # Train final model on full 2020-2024, test on 2025
    train_final = df[df["week_start"] < pd.to_datetime("2025-01-01")].copy()
    test_final = df[df["week_start"] >= pd.to_datetime("2025-01-01")].copy()
    
    if len(train_final) > 50 and len(test_final) > 10:
        print(f"\n  Final model:")
        print(f"    Train: {train_final['week_start'].min()} to {train_final['week_start'].max()} ({len(train_final)} rows)")
        print(f"    Test: {test_final['week_start'].min()} to {test_final['week_start'].max()} ({len(test_final)} rows)")
        
        final_model, final_metrics = train_xgb_model(train_final, test_final, 0, regime_label)
        final_metrics["fold"] = "final"
        all_metrics.append(final_metrics)
        
        print(f"    Test RMSE: {final_metrics['test_rmse']:.2f}, MAE: {final_metrics['test_mae']:.2f}, Dir Acc: {final_metrics['test_dir_acc']:.2%}")
        
        # Use final model for registration
        best_model = final_model
        best_metrics = final_metrics
    else:
        # Use best fold model
        best_fold = max(all_metrics, key=lambda m: m["test_dir_acc"])
        best_model = models[best_fold["fold"] - 1]
        best_metrics = best_fold
    
    # Register model
    model_name = f"xgb_credit_{regime_label.lower()}_v1"
    
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO model_registry (
                    model_name, model_type, regime, train_start, train_end,
                    hyperparameters, metrics, feature_importance
                )
                VALUES (
                    :model_name, :model_type, :regime, :train_start, :train_end,
                    :hyperparameters, :metrics, :feature_importance
                )
                RETURNING model_id
            """),
            {
                "model_name": model_name,
                "model_type": "XGBoost",
                "regime": regime_label,
                "train_start": df["week_start"].min(),
                "train_end": train_final["week_start"].max() if len(train_final) > 0 else df["week_start"].max(),
                "hyperparameters": json.dumps(XGB_PARAMS),
                "metrics": json.dumps(best_metrics),
                "feature_importance": json.dumps({}),  # Will be filled by SHAP
            }
        )
        model_id = result.fetchone()[0]
    
    print(f"\n  Model registered: {model_name} (ID: {model_id})")
    
    # Save predictions if we have final model
    if len(train_final) > 0 and len(test_final) > 0:
        from src.models.save_predictions import save_predictions
        
        # Prepare features for prediction
        X_train_final = train_final[FEATURE_COLS].copy()
        X_test_final = test_final[FEATURE_COLS].copy()
        
        # Handle categorical and fill missing
        if "bucket" in X_train_final.columns:
            X_train_final["bucket"] = X_train_final["bucket"].astype("category")
            X_test_final["bucket"] = X_test_final["bucket"].astype("category")
        
        for col in X_train_final.columns:
            if col != "bucket":
                median_val = X_train_final[col].median()
                X_train_final[col] = X_train_final[col].fillna(median_val)
                X_test_final[col] = X_test_final[col].fillna(median_val)
        
        train_pred = best_model.predict(X_train_final)
        test_pred = best_model.predict(X_test_final)
        
        save_predictions(
            model_id=model_id,
            train_df=train_final,
            train_pred=train_pred,
            test_df=test_final,
            test_pred=test_pred,
            feature_cols=FEATURE_COLS,
        )
        print(f"  Saved predictions for model {model_id}")
    
    return {
        "model_id": model_id,
        "model_name": model_name,
        "model": best_model,
        "all_metrics": all_metrics,
        "final_metrics": best_metrics,
        "train_df": train_final if len(train_final) > 0 else df,
        "test_df": test_final if len(test_final) > 0 else pd.DataFrame(),
    }

