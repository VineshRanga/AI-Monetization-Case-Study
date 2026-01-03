#!/usr/bin/env python3
"""Quick model performance report."""
import sys
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.db import get_engine
from sqlalchemy import text

if __name__ == "__main__":
    load_dotenv()
    
    print("="*80)
    print("MODEL PERFORMANCE REPORT")
    print("="*80)
    
    engine = get_engine()
    
    # Metrics by issuer for 2025
    print("\n1. Metrics by Issuer (2025 Test Period):")
    print("-" * 80)
    
    with engine.connect() as conn:
        issuer_metrics = pd.read_sql(
            text("""
                SELECT 
                    i.ticker,
                    i.issuer_name,
                    COUNT(*) as n_weeks,
                    AVG(ABS(p.y_true - p.y_pred)) as mae,
                    SQRT(AVG(POWER(p.y_true - p.y_pred, 2))) as rmse,
                    AVG(CASE WHEN SIGN(p.y_true) = SIGN(p.y_pred) THEN 1 ELSE 0 END) as dir_acc
                FROM model_predictions_weekly p
                JOIN dim_issuer i ON p.issuer_id = i.issuer_id
                JOIN model_registry m ON p.model_id = m.model_id
                WHERE p.split_tag = 'test'
                    AND p.week_start >= '2025-01-01'
                GROUP BY i.ticker, i.issuer_name
                ORDER BY mae DESC
            """),
            conn
        )
        
        if not issuer_metrics.empty:
            print(issuer_metrics.to_string(index=False))
        else:
            print("  No test predictions found for 2025")
    
    # Top errors
    print("\n2. Top 10 Largest Errors (2025):")
    print("-" * 80)
    
    with engine.connect() as conn:
        top_errors = pd.read_sql(
            text("""
                SELECT 
                    i.ticker,
                    p.week_start,
                    p.y_true,
                    p.y_pred,
                    ABS(p.y_true - p.y_pred) as error,
                    m.regime_label
                FROM model_predictions_weekly p
                JOIN dim_issuer i ON p.issuer_id = i.issuer_id
                JOIN model_registry m ON p.model_id = m.model_id
                WHERE p.split_tag = 'test'
                    AND p.week_start >= '2025-01-01'
                ORDER BY ABS(p.y_true - p.y_pred) DESC
                LIMIT 10
            """),
            conn
        )
        
        if not top_errors.empty:
            print(top_errors.to_string(index=False))
        else:
            print("  No test predictions found")
    
    # Sample predictions for 2 issuers
    print("\n3. Sample Predictions vs Actual (Last 10 Weeks for 2 Issuers):")
    print("-" * 80)
    
    with engine.connect() as conn:
        issuers = conn.execute(
            text("""
                SELECT DISTINCT i.ticker
                FROM model_predictions_weekly p
                JOIN dim_issuer i ON p.issuer_id = i.issuer_id
                WHERE p.split_tag = 'test'
                LIMIT 2
            """)
        ).fetchall()
        
        for (ticker,) in issuers:
            samples = pd.read_sql(
                text("""
                    SELECT 
                        p.week_start,
                        p.y_true,
                        p.y_pred,
                        p.y_true - p.y_pred as error,
                        m.regime_label
                    FROM model_predictions_weekly p
                    JOIN dim_issuer i ON p.issuer_id = i.issuer_id
                    JOIN model_registry m ON p.model_id = m.model_id
                    WHERE i.ticker = :ticker
                        AND p.split_tag = 'test'
                    ORDER BY p.week_start DESC
                    LIMIT 10
                """),
                conn,
                params={"ticker": ticker}
            )
            
            if not samples.empty:
                print(f"\n  {ticker}:")
                print(samples.to_string(index=False))
    
    print("\n" + "="*80)

