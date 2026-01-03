"""Train regime-gated XGBoost models for credit proxy change prediction."""
import json
import numpy as np
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not available, using sklearn GradientBoostingRegressor")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db.db import get_engine, now_utc

# Feature list (exclude week_start, issuer_id, ticker-like, targets)
# Note: issuer_mean_target is computed dynamically, not in this list
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
    "bucket",  # Categorical
]

# Features that are computed dynamically (not in model_dataset_weekly)
DERIVED_FEATURES = ["issuer_mean_target"]

# Hyperparameters
if XGBOOST_AVAILABLE:
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
else:
    XGB_PARAMS = {
        "n_estimators": 800,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.8,
        "random_state": 42,
    }


def compute_issuer_target_encoding(train_df: pd.DataFrame) -> pd.Series:
    """
    Compute issuer mean dcredit_proxy for target encoding.
    
    Returns Series indexed by issuer_id.
    """
    return train_df.groupby("issuer_id")["dcredit_proxy"].mean()


def prepare_features(
    df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Prepare features with target encoding and missing value imputation.
    
    If train_df is provided, use its statistics for imputation.
    Otherwise, use df's own statistics.
    
    Returns DataFrame with ONLY numeric columns (XGBoost requirement).
    """
    # Drop non-feature columns explicitly
    drop_cols = {
        "week_start", "issuer_id", "ticker", "issuer_name", 
        "bucket", "regime_label", "y_dspread_bps", "dcredit_proxy"
    }
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    
    # Force all columns to numeric (XGBoost requirement)
    X = X.apply(pd.to_numeric, errors="coerce")
    
    # Compute issuer target encoding (derived feature, not in original dataframe)
    if train_df is not None:
        issuer_means = compute_issuer_target_encoding(train_df)
        global_mean = train_df["dcredit_proxy"].mean()
        # Use train_df to identify all-NaN columns
        train_X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns]).copy()
        train_X = train_X.apply(pd.to_numeric, errors="coerce")
    else:
        issuer_means = compute_issuer_target_encoding(df)
        global_mean = df["dcredit_proxy"].mean()
        train_X = X
    
    # Drop columns that are all-NaN in TRAIN (before adding derived features)
    all_nan_cols = [c for c in train_X.columns if train_X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
        print(f"  Dropped {len(all_nan_cols)} all-NaN columns: {all_nan_cols[:10]}")
    
    # Add issuer_mean_target as derived feature
    X["issuer_mean_target"] = df["issuer_id"].map(issuer_means).fillna(global_mean)
    
    # Fill missing values with median (compute on TRAIN only, apply to both)
    for col in X.columns:
        if col != "issuer_mean_target":  # issuer_mean_target already filled
            if train_df is not None:
                # Use train_df statistics (train_X already computed above)
                if col in train_X.columns:
                    median_val = train_X[col].median()
                else:
                    median_val = 0.0  # Fallback if column doesn't exist in train
            else:
                median_val = X[col].median()
            
            # Guard: if median is NaN (empty slice), fill with 0
            if pd.isna(median_val):
                median_val = 0.0
            
            X[col] = X[col].fillna(median_val)
    
    # Debug print
    print(f"  Features kept: {len(X.columns)} columns")
    
    return X


def train_xgb_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    regime_label: str,
) -> Tuple[object, Dict]:
    """
    Train XGBoost (or sklearn GradientBoosting) model.
    
    Returns (model, metrics_dict).
    """
    # Prepare features
    X_train = prepare_features(train_df)
    y_train = train_df["dcredit_proxy"].values
    
    X_test = prepare_features(test_df, train_df=train_df)
    y_test = test_df["dcredit_proxy"].values
    
    # Train model
    # Note: early_stopping_rounds is deprecated in newer XGBoost versions
    # We'll fit on full training set and evaluate on test set
    if XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(**XGB_PARAMS)
        try:
            # Try with early stopping (older XGBoost versions)
            val_size = int(len(X_train) * 0.2)
            if val_size > 0:
                X_train_fit = X_train.iloc[:-val_size]
                y_train_fit = y_train[:-val_size]
                X_val = X_train.iloc[-val_size:]
                y_val = y_train[-val_size:]
                model.fit(
                    X_train_fit,
                    y_train_fit,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False,
                )
                print("  Fit used early stopping: True")
            else:
                model.fit(X_train, y_train, verbose=False)
                print("  Fit used early stopping: False (no validation split)")
        except (TypeError, ValueError) as e:
            # Newer XGBoost versions don't support early_stopping_rounds in fit()
            # ValueError can occur if eval_set format is wrong
            model.fit(X_train, y_train, verbose=False)
            print(f"  Fit used early stopping: False (not supported: {type(e).__name__})")
    else:
        model = GradientBoostingRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)
        print("  Fit used early stopping: False (sklearn GradientBoosting)")
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    metrics = {
        "train_rmse": float(train_rmse),
        "train_mae": float(train_mae),
        "train_r2": float(train_r2),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
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
    
    # Force week_start to pandas datetime (PostgreSQL returns date objects)
    df["week_start"] = pd.to_datetime(df["week_start"])
    
    print(f"\nTraining {regime_label} model...")
    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df['week_start'].min()} to {df['week_start'].max()}")
    print(f"  week_start dtype: {df['week_start'].dtype}")
    
    # Time split: train < 2025-01-01, test >= 2025-01-01 and <= 2025-12-31
    # Use pd.Timestamp for consistent type comparison
    split_date = pd.Timestamp("2025-01-01")
    end_date = pd.Timestamp("2025-12-31")
    
    train_df = df[df["week_start"] < split_date].copy()
    test_df = df[
        (df["week_start"] >= split_date) &
        (df["week_start"] <= end_date)
    ].copy()
    
    print(f"  Train: {train_df['week_start'].min()} to {train_df['week_start'].max()} ({len(train_df)} rows)")
    print(f"  Test: {test_df['week_start'].min()} to {test_df['week_start'].max()} ({len(test_df)} rows)")
    
    if len(train_df) < 50:
        print(f"  Not enough training data ({len(train_df)} rows, need at least 50)")
        return {}
    
    if len(test_df) == 0:
        print(f"  No test data for 2025")
        # Use last 20% of training data as test
        test_size = int(len(train_df) * 0.2)
        test_df = train_df.tail(test_size).copy()
        train_df = train_df.iloc[:-test_size].copy()
        print(f"  Using last 20% of training data as test: {len(test_df)} rows")
    
    # Train model
    model, metrics = train_xgb_model(train_df, test_df, regime_label)
    
    print(f"  Train RMSE: {metrics['train_rmse']:.4f}, MAE: {metrics['train_mae']:.4f}, R2: {metrics['train_r2']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}, MAE: {metrics['test_mae']:.4f}, R2: {metrics['test_r2']:.4f}")
    
    # Register model
    model_name = f"xgb_credit_{regime_label.lower()}_v1"
    
    # Convert Timestamp to date for PostgreSQL DATE columns
    train_start_date = train_df["week_start"].min()
    if hasattr(train_start_date, 'date'):
        train_start_date = train_start_date.date()
    elif isinstance(train_start_date, pd.Timestamp):
        train_start_date = train_start_date.date()
    
    train_end_date = train_df["week_start"].max()
    if hasattr(train_end_date, 'date'):
        train_end_date = train_end_date.date()
    elif isinstance(train_end_date, pd.Timestamp):
        train_end_date = train_end_date.date()
    
    test_start_date = None
    if len(test_df) > 0:
        test_start_date = test_df["week_start"].min()
        if hasattr(test_start_date, 'date'):
            test_start_date = test_start_date.date()
        elif isinstance(test_start_date, pd.Timestamp):
            test_start_date = test_start_date.date()
    
    test_end_date = None
    if len(test_df) > 0:
        test_end_date = test_df["week_start"].max()
        if hasattr(test_end_date, 'date'):
            test_end_date = test_end_date.date()
        elif isinstance(test_end_date, pd.Timestamp):
            test_end_date = test_end_date.date()
    
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO model_registry (
                    name,
                    model_name, model_type,
                    regime, regime_label,
                    target, train_start, train_end, test_start, test_end,
                    hyperparameters, metrics
                )
                VALUES (
                    :name,
                    :model_name, :model_type,
                    :regime, :regime_label,
                    :target, :train_start, :train_end, :test_start, :test_end,
                    :hyperparameters, :metrics
                )
                ON CONFLICT (model_name) DO UPDATE SET
                    name = EXCLUDED.name,
                    model_type = EXCLUDED.model_type,
                    regime = EXCLUDED.regime,
                    regime_label = EXCLUDED.regime_label,
                    target = EXCLUDED.target,
                    train_start = EXCLUDED.train_start,
                    train_end = EXCLUDED.train_end,
                    test_start = EXCLUDED.test_start,
                    test_end = EXCLUDED.test_end,
                    hyperparameters = EXCLUDED.hyperparameters,
                    metrics = EXCLUDED.metrics,
                    created_at = NOW()
                RETURNING model_id
            """),
            {
                "name": model_name,  # Legacy column: set equal to model_name for backwards compatibility
                "model_name": model_name,
                "model_type": "XGBoost" if XGBOOST_AVAILABLE else "GradientBoosting",
                "regime": regime_label,
                "regime_label": regime_label,  # Legacy column: set equal to regime for backwards compatibility
                "target": "dcredit_proxy",
                "train_start": train_start_date,
                "train_end": train_end_date,
                "test_start": test_start_date,
                "test_end": test_end_date,
                "hyperparameters": json.dumps(XGB_PARAMS),
                "metrics": json.dumps(metrics),
            }
        )
        model_id = result.fetchone()[0]
    
    print(f"\n  Model registered: {model_name} (ID: {model_id})")
    
    # Prepare features (needed for feature importance and predictions)
    # Compute once and reuse
    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df, train_df=train_df)
    y_train = train_df["dcredit_proxy"].values
    y_test = test_df["dcredit_proxy"].values
    
    # Save feature importances
    try:
        if XGBOOST_AVAILABLE:
            # Extract feature importances (gain)
            booster = model.get_booster()
            gain_dict = booster.get_score(importance_type="gain")
            weight_dict = booster.get_score(importance_type="weight")
            
            # Use gain if available, fallback to weight
            if gain_dict:
                importance_dict = gain_dict
            elif weight_dict:
                importance_dict = weight_dict
            else:
                print("  Warning: No feature importance scores available from XGBoost")
                importance_dict = {}
            
            # Get actual feature names from X_train
            feature_names = list(X_train.columns)
            
            # Map XGBoost feature names (f0, f1, etc.) to actual feature names
            feature_map = {f"f{i}": name for i, name in enumerate(feature_names)}
            
            # Build importance dict aligned to feature_names
            # XGBoost returns f0, f1, etc., we map to actual names
            imp_dict = {}
            for xgb_feat, imp_val in importance_dict.items():
                # Map f0, f1, etc. to actual feature names
                actual_feat = feature_map.get(xgb_feat, xgb_feat)
                imp_dict[actual_feat] = float(imp_val)
            
            # Ensure all features in X_train have an importance value (default 0.0)
            for feat in feature_names:
                if feat not in imp_dict:
                    imp_dict[feat] = 0.0
            
            # Build DataFrame with all features
            importance_raw = [imp_dict.get(f, 0.0) for f in feature_names]
            df_imp = pd.DataFrame({
                "feature": feature_names,
                "importance_raw": importance_raw
            })
            
            # Normalize
            total_imp = df_imp["importance_raw"].sum()
            if total_imp > 0:
                df_imp["importance_norm"] = df_imp["importance_raw"] / total_imp
            else:
                df_imp["importance_norm"] = 0.0
            
            importances = df_imp.to_dict('records')
        else:
            # sklearn GradientBoostingRegressor
            feature_names = list(X_train.columns)
            raw_importances = [float(imp) for imp in model.feature_importances_]
            total_imp = sum(raw_importances)
            
            importances = []
            for feat, imp_raw in zip(feature_names, raw_importances):
                imp_norm = imp_raw / total_imp if total_imp > 0 else 0.0
                importances.append({
                    "feature": feat,
                    "importance_raw": imp_raw,
                    "importance_norm": imp_norm
                })
        
        # Upsert feature importances
        if importances:
            with engine.begin() as conn:
                # Clear old rows for this model_id first
                conn.execute(
                    text("DELETE FROM model_feature_importance WHERE model_id = :model_id"),
                    {"model_id": model_id}
                )
                
                # Insert new importances
                for imp_row in importances:
                    conn.execute(
                        text("""
                            INSERT INTO model_feature_importance (
                                model_id, feature, importance, importance_raw, importance_norm
                            )
                            VALUES (
                                :model_id, :feature, :importance, :importance_raw, :importance_norm
                            )
                        """),
                        {
                            "model_id": model_id,
                            "feature": str(imp_row["feature"]),
                            "importance": float(imp_row.get("importance_norm", 0.0)),  # Store normalized in importance for backwards compat
                            "importance_raw": float(imp_row.get("importance_raw", 0.0)),
                            "importance_norm": float(imp_row.get("importance_norm", 0.0)),
                        }
                    )
            max_imp_raw = max([r.get("importance_raw", 0.0) for r in importances])
            max_imp_norm = max([r.get("importance_norm", 0.0) for r in importances])
            print(f"  Saved {len(importances)} feature importances (max raw: {max_imp_raw:.6f}, max norm: {max_imp_norm:.6f})")
        else:
            print("  Warning: No feature importances to save")
    except Exception as e:
        print(f"  Warning: Failed to save feature importances: {e}")
        import traceback
        traceback.print_exc()
    
    # Save predictions
    from src.models.save_predictions import save_predictions
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    save_predictions(
        model_id=model_id,
        train_df=train_df,
        train_pred=train_pred.tolist(),
        test_df=test_df,
        test_pred=test_pred.tolist(),
        feature_cols=FEATURE_COLS,
    )
    print(f"  Saved predictions for model {model_id}")
    
    return {
        "model_id": model_id,
        "model_name": model_name,
        "model": model,
        "metrics": metrics,
        "train_df": train_df,
        "test_df": test_df,
    }

