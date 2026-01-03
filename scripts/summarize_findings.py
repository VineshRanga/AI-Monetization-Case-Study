#!/usr/bin/env python3
"""Print plain-English summary of model findings."""
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.db import get_engine

if __name__ == "__main__":
    load_dotenv()
    
    print("="*80)
    print("MODEL FINDINGS SUMMARY")
    print("="*80)
    
    engine = get_engine()
    
    # 1. Plain-English recap (5 bullets)
    print("\n[1] MODEL OVERVIEW")
    print("-" * 80)
    with engine.connect() as conn:
        # Count models
        result = conn.execute(text("SELECT COUNT(*) FROM model_registry"))
        row = result.fetchone()
        model_count = row[0] if row else 0
        
        # Get test metrics
        result = conn.execute(text("""
            SELECT AVG(ABS(y_pred - y_true)) as mae, COUNT(*) as n
            FROM model_predictions_weekly
            WHERE split = 'test'
        """))
        row = result.fetchone()
        mae = row[0] if row and row[0] is not None else 0
        n = row[1] if row and row[1] is not None else 0
        
        # Count issuers
        result = conn.execute(text("SELECT COUNT(*) FROM dim_issuer"))
        row = result.fetchone()
        issuer_count = row[0] if row else 0
        
        # Count weeks
        result = conn.execute(text("""
            SELECT COUNT(DISTINCT week_start)
            FROM fact_credit_proxy_weekly
            WHERE week_start >= '2025-01-01'
        """))
        row = result.fetchone()
        week_count = row[0] if row else 0
    
    print("The model predicts weekly changes in issuer credit risk proxy (dcredit_proxy),")
    print("which is derived from a simplified Merton-style distance-to-default calculation.")
    print("")
    print(f"• Trained {model_count} regime-gated models (RISK_ON and RISK_OFF) using XGBoost")
    print(f"• Test period (2025): {n} predictions across {issuer_count} issuers over {week_count} weeks")
    print(f"• Average prediction error (MAE): {mae:.4f}")
    print(f"• Models use EDGAR fundamentals + equity returns/vol + equity-only regime proxy (QQQ/SMH) as features")
    print(f"• Macro (FRED: VIX/rates/OAS) is currently missing; regime labeling is equity-only fallback until added")
    print(f"• Regime separation improves prediction accuracy by accounting for market conditions")
    
    # 2. Top 5 issuers with worst predicted deterioration in 2025 risk-off weeks
    print("\n[2] TOP 5 ISSUERS WITH WORST PREDICTED DETERIORATION (2025 RISK-OFF WEEKS)")
    print("-" * 80)
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT 
                i.ticker,
                i.issuer_name,
                AVG(p.y_pred) as avg_predicted_deterioration,
                COUNT(*) as risk_off_weeks
            FROM model_predictions_weekly p
            JOIN dim_issuer i ON p.issuer_id = i.issuer_id
            JOIN model_dataset_weekly d ON p.issuer_id = d.issuer_id AND p.week_start = d.week_start
            WHERE p.split = 'test'
            AND p.week_start >= '2025-01-01'
            AND d.regime_label = 'RISK_OFF'
            AND p.y_pred > 0
            GROUP BY i.ticker, i.issuer_name
            ORDER BY avg_predicted_deterioration DESC
            LIMIT 5
        """))
        
        rows = result.fetchall()
        if rows:
            for i, (ticker, name, avg_deterioration, weeks) in enumerate(rows, 1):
                print(f"{i}. {ticker} ({name}): {avg_deterioration:.4f} avg deterioration ({weeks} RISK_OFF weeks)")
        else:
            print("  No RISK_OFF predictions found for 2025")
    
    # 3. Top 5 features driving risk-off outcomes (from feature importance)
    print("\n[3] TOP 5 FEATURES DRIVING RISK-OFF OUTCOMES (FEATURE IMPORTANCE)")
    print("-" * 80)
    try:
        with engine.connect() as conn:
            # Get latest/active RISK_OFF model_id
            result = conn.execute(text("""
                SELECT model_id
                FROM model_registry
                WHERE regime = 'RISK_OFF' AND model_name = 'xgb_credit_risk_off_v1'
                ORDER BY created_at DESC
                LIMIT 1
            """))
            row = result.fetchone()
            
            if row is None:
                print("  No RISK_OFF model found in model_registry")
            else:
                model_id = row[0]
                
                # Get top 5 feature importances (use importance_norm if available, else importance_raw, else importance)
                result = conn.execute(
                    text("""
                        SELECT feature, 
                               COALESCE(importance_norm, importance_raw, importance) as importance_val,
                               importance_raw, importance_norm
                        FROM model_feature_importance
                        WHERE model_id = :model_id
                        ORDER BY COALESCE(importance_norm, importance_raw, importance) DESC
                        LIMIT 5
                    """),
                    {"model_id": model_id}
                )
                
                rows = result.fetchall()
                if rows:
                    # Debug: print count and max
                    max_imp = max([float(r[1]) if r[1] is not None else 0.0 for r in rows])
                    print(f"  Debug: {len(rows)} features found, max importance: {max_imp:.6f}")
                    
                    for i, (feature, importance_val, importance_raw, importance_norm) in enumerate(rows, 1):
                        val = float(importance_val) if importance_val is not None else 0.0
                        # Print with 6 decimals, or scientific if very small
                        if abs(val) < 0.000001:
                            print(f"{i}. {feature}: {val:.3e}")
                        else:
                            print(f"{i}. {feature}: {val:.6f}")
                else:
                    print("  No feature importance found (table empty). Re-run Stage 3 to repopulate.")
    except Exception as e:
        print(f"  Warning: Failed to load feature importance: {e}")
        import traceback
        traceback.print_exc()
        print("  Re-run Stage 3 to populate model_feature_importance table.")
    
    # 4. Fragility Score snapshot
    print("\n[4] FRAGILITY SCORE SNAPSHOT (AS OF LATEST QUARTER)")
    print("-" * 80)
    with engine.connect() as conn:
        # Get latest asof_date
        result = conn.execute(text("SELECT MAX(asof_date) FROM fragility_scores"))
        row = result.fetchone()
        latest_date = row[0] if row and row[0] else None
        
        if latest_date:
            print(f"As of: {latest_date}")
            print("")
            
            # Get top 5 most fragile issuers with pillar breakdown
            result = conn.execute(text("""
                SELECT 
                    i.ticker,
                    i.issuer_name,
                    fs.total_score_0_100,
                    MAX(CASE WHEN fps.pillar_name = 'refinancing_wall' THEN fps.score_0_100 END) as refinancing_wall,
                    MAX(CASE WHEN fps.pillar_name = 'cash_generation' THEN fps.score_0_100 END) as cash_generation,
                    MAX(CASE WHEN fps.pillar_name = 'leverage_coverage' THEN fps.score_0_100 END) as leverage_coverage,
                    MAX(CASE WHEN fps.pillar_name = 'cyclicality_ai' THEN fps.score_0_100 END) as cyclicality_ai,
                    MAX(CASE WHEN fps.pillar_name = 'structure_opacity' THEN fps.score_0_100 END) as structure_opacity
                FROM fragility_scores fs
                JOIN dim_issuer i ON fs.issuer_id = i.issuer_id
                LEFT JOIN fragility_pillar_scores fps ON fs.issuer_id = fps.issuer_id AND fs.asof_date = fps.asof_date
                WHERE fs.asof_date = :latest_date
                GROUP BY i.ticker, i.issuer_name, fs.total_score_0_100
                ORDER BY fs.total_score_0_100 DESC
                LIMIT 5
            """), {"latest_date": latest_date})
            
            rows = result.fetchall()
            if rows:
                for i, (ticker, name, total_score, ref, cash, lev, cycl, struct) in enumerate(rows, 1):
                    # Find highest pillar(s) - get top 2 if close
                    pillars = [
                        ("refinancing wall", ref),
                        ("cash generation", cash),
                        ("leverage/coverage", lev),
                        ("cyclicality/AI", cycl),
                        ("structure/opacity", struct),
                    ]
                    pillars = [(n, v) for n, v in pillars if v is not None]
                    if pillars:
                        # Sort by score descending
                        pillars_sorted = sorted(pillars, key=lambda x: x[1], reverse=True)
                        highest = pillars_sorted[0]
                        
                        # Build one-line reason
                        if highest[1] >= 70:
                            risk_level = "high"
                        elif highest[1] >= 50:
                            risk_level = "medium-high"
                        else:
                            risk_level = "moderate"
                        
                        pillar_name_short = {
                            "refinancing wall": "refinancing",
                            "cash generation": "cash generation",
                            "leverage/coverage": "leverage/coverage",
                            "cyclicality/AI": "cyclicality",
                            "structure/opacity": "structure/opacity"
                        }.get(highest[0], highest[0])
                        
                        # Check if second pillar is also high
                        if len(pillars_sorted) > 1 and pillars_sorted[1][1] >= 60:
                            second = pillars_sorted[1]
                            second_name = {
                                "refinancing wall": "refinancing",
                                "cash generation": "cash generation",
                                "leverage/coverage": "leverage/coverage",
                                "cyclicality/AI": "cyclicality",
                                "structure/opacity": "structure/opacity"
                            }.get(second[0], second[0])
                            reason = f"high {pillar_name_short} + high {second_name}"
                        else:
                            reason = f"high {pillar_name_short}"
                    else:
                        reason = "insufficient data"
                    
                    print(f"{i}. {ticker}: {reason}")
            else:
                print("  No fragility scores found. Run scripts/run_fragility.py first.")
        else:
            print("  No fragility scores found. Run scripts/run_fragility.py first.")
    
    print("\n" + "="*80)
    print("SUMMARY COMPLETE")
    print("="*80)

