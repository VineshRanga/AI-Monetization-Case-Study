"""Save model predictions to Postgres."""
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import List

from src.db.db import get_engine


def save_predictions(
    model_id: int,
    train_df: pd.DataFrame,
    train_pred: List[float],
    test_df: pd.DataFrame,
    test_pred: List[float],
    feature_cols: List[str],
):
    """Save predictions for train and test sets."""
    engine = get_engine()
    
    if model_id is None:
        print("Warning: model_id is None, cannot save predictions")
        return
    
    with engine.begin() as conn:
        # Save train predictions
        for idx, (_, row) in enumerate(train_df.iterrows()):
            if idx >= len(train_pred):
                break
            conn.execute(
                text("""
                    INSERT INTO model_predictions_weekly (
                        issuer_id, week_start, model_id, split, y_true, y_pred
                    )
                    VALUES (
                        :issuer_id, :week_start, :model_id, 'train', :y_true, :y_pred
                    )
                    ON CONFLICT (issuer_id, week_start, model_id, split) DO UPDATE SET
                        y_true = EXCLUDED.y_true,
                        y_pred = EXCLUDED.y_pred
                """),
                {
                    "model_id": model_id,
                    "issuer_id": int(row["issuer_id"]),
                    "week_start": row["week_start"],
                    "y_true": float(row["dcredit_proxy"]) if pd.notna(row["dcredit_proxy"]) else None,
                    "y_pred": float(train_pred[idx]) if pd.notna(train_pred[idx]) else None,
                }
            )
        
        # Save test predictions
        for idx, (_, row) in enumerate(test_df.iterrows()):
            conn.execute(
                text("""
                    INSERT INTO model_predictions_weekly (
                        issuer_id, week_start, model_id, split, y_true, y_pred
                    )
                    VALUES (
                        :issuer_id, :week_start, :model_id, 'test', :y_true, :y_pred
                    )
                    ON CONFLICT (issuer_id, week_start, model_id, split) DO UPDATE SET
                        y_true = EXCLUDED.y_true,
                        y_pred = EXCLUDED.y_pred
                """),
                {
                    "model_id": model_id,
                    "issuer_id": int(row["issuer_id"]),
                    "week_start": row["week_start"],
                    "y_true": float(row["dcredit_proxy"]) if pd.notna(row["dcredit_proxy"]) else None,
                    "y_pred": float(test_pred[idx]) if pd.notna(test_pred[idx]) else None,
                }
            )

