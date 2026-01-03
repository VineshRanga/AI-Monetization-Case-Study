"""Build regime labels using rule-based approach."""
import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import Dict

from src.db.db import get_engine


def robust_zscore(series: pd.Series) -> pd.Series:
    """Compute robust z-scores using median and MAD."""
    median = series.median()
    mad = (series - median).abs().median()
    
    if mad == 0:
        return pd.Series(0, index=series.index)
    
    return (series - median) / (1.4826 * mad)  # MAD to SD conversion


def sigmoid(x: float) -> float:
    """Sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def build_regime_labels() -> Dict:
    """
    Build regime labels using rule-based approach.
    
    Primary method: Uses VIX, QQQ, and IG OAS if available.
    Fallback method: Uses equity-only (QQQ, SMH returns) if FRED data is missing.
    """
    engine = get_engine()
    
    # Get weekly market data (all available columns)
    # Filter to project window: 2020-01-01 to 2025-12-31
    with engine.connect() as conn:
        market_df = pd.read_sql(
            text("""
                SELECT week_start, vix_chg, qqq_ret, smh_ret, ig_oas_chg
                FROM fact_weekly_market
                WHERE week_start >= '2020-01-01' AND week_start <= '2025-12-31'
                ORDER BY week_start
            """),
            conn
        )
    
    if market_df.empty:
        print("No market data found in fact_weekly_market")
        return {"weeks_labeled": 0, "risk_off_count": 0, "risk_off_pct": 0.0}
    
    # Check if we have FRED data (VIX, OAS)
    has_vix = market_df["vix_chg"].notna().any()
    has_oas = market_df["ig_oas_chg"].notna().any()
    has_qqq = market_df["qqq_ret"].notna().any()
    has_smh = market_df["smh_ret"].notna().any()
    
    # Determine which method to use and set method name
    method_name = None
    use_equity_only = False
    
    if has_vix and has_oas and has_qqq:
        # Primary method: Use VIX, QQQ, OAS
        print("  Using primary method: VIX + QQQ + OAS")
        market_df_filtered = market_df[
            market_df["vix_chg"].notna() &
            market_df["qqq_ret"].notna() &
            market_df["ig_oas_chg"].notna()
        ].copy()
        
        if market_df_filtered.empty:
            print("  No complete FRED data rows. Falling back to equity-only method.")
            use_equity_only = True
            method_name = "EQUITY_FALLBACK_QQQ_SMH_V1"
        else:
            # Compute robust z-scores
            market_df_filtered["z_vix_chg"] = robust_zscore(market_df_filtered["vix_chg"])
            market_df_filtered["z_qqq_ret"] = robust_zscore(-market_df_filtered["qqq_ret"])  # Negative for risk-off
            market_df_filtered["z_ig_oas_chg"] = robust_zscore(market_df_filtered["ig_oas_chg"])
            
            # Rule-based probability
            # Weights: a1=0.4, a2=0.3, a3=0.3
            a1, a2, a3 = 0.4, 0.3, 0.3
            
            market_df_filtered["prob_risk_off"] = market_df_filtered.apply(
                lambda row: sigmoid(
                    a1 * row["z_vix_chg"] +
                    a2 * row["z_qqq_ret"] +
                    a3 * row["z_ig_oas_chg"]
                ),
                axis=1
            )
            method_name = "MACRO_VIX_RATES_OAS_V1"
            use_equity_only = False
    else:
        use_equity_only = True
        method_name = "EQUITY_FALLBACK_QQQ_SMH_V1"
    
    if use_equity_only:
        # Fallback method: Equity-only (QQQ, SMH)
        print("  Using equity-only fallback method: QQQ + SMH")
        market_df_filtered = market_df[
            (market_df["qqq_ret"].notna()) | (market_df["smh_ret"].notna())
        ].copy()
        
        if market_df_filtered.empty:
            print("  No equity return data available")
            return {"weeks_labeled": 0, "risk_off_count": 0, "risk_off_pct": 0.0}
        
        # Compute prob_risk_off from equity returns
        def compute_equity_prob(row):
            base = 0.0
            # QQQ contribution (70% weight)
            if pd.notna(row["qqq_ret"]):
                qqq_drawdown = -row["qqq_ret"]  # Negative return = drawdown
                qqq_contrib = np.clip(qqq_drawdown / 0.03, 0, 1) * 0.7  # 3% weekly drawdown ~ high risk-off
                base += qqq_contrib
            # SMH contribution (30% weight)
            if pd.notna(row["smh_ret"]):
                smh_drawdown = -row["smh_ret"]  # Negative return = drawdown
                smh_contrib = np.clip(smh_drawdown / 0.04, 0, 1) * 0.3  # 4% weekly drawdown ~ high risk-off
                base += smh_contrib
            return np.clip(base, 0, 1)
        
        market_df_filtered["prob_risk_off"] = market_df_filtered.apply(compute_equity_prob, axis=1)
    
    # Label
    market_df_filtered["regime_label"] = market_df_filtered["prob_risk_off"].apply(
        lambda p: "RISK_OFF" if p >= 0.6 else "RISK_ON"
    )
    
    # Add method column to dataframe
    market_df_filtered["method"] = method_name
    
    # Ensure week_start is date type and filter to project window (2020-01-01 to 2025-12-31)
    from datetime import date as date_type
    market_df_filtered["week_start"] = pd.to_datetime(market_df_filtered["week_start"]).dt.date
    min_date = date_type(2020, 1, 1)
    max_date = date_type(2025, 12, 31)
    market_df_filtered = market_df_filtered[
        (market_df_filtered["week_start"] >= min_date) &
        (market_df_filtered["week_start"] <= max_date)
    ].copy()
    
    # Upsert to database
    with engine.begin() as conn:
        for _, row in market_df_filtered.iterrows():
            conn.execute(
                text("""
                    INSERT INTO model_regime_weekly (
                        week_start, prob_risk_off, regime_label, method
                    )
                    VALUES (
                        :week_start, :prob_risk_off, :regime_label, :method
                    )
                    ON CONFLICT (week_start) DO UPDATE SET
                        prob_risk_off = EXCLUDED.prob_risk_off,
                        regime_label = EXCLUDED.regime_label,
                        method = EXCLUDED.method
                """),
                {
                    "week_start": row["week_start"],
                    "prob_risk_off": float(row["prob_risk_off"]),
                    "regime_label": str(row["regime_label"]),
                    "method": str(row["method"]),
                }
            )
    
    risk_off_count = (market_df_filtered["regime_label"] == "RISK_OFF").sum()
    risk_off_pct = (risk_off_count / len(market_df_filtered)) * 100 if len(market_df_filtered) > 0 else 0.0
    
    print(f"  Created {len(market_df_filtered)} regime labels")
    print(f"  Risk-OFF: {risk_off_count} ({risk_off_pct:.1f}%)")
    print(f"  Risk-ON: {len(market_df_filtered) - risk_off_count} ({100 - risk_off_pct:.1f}%)")
    
    return {
        "weeks_labeled": len(market_df_filtered),
        "risk_off_count": risk_off_count,
        "risk_off_pct": risk_off_pct,
    }

