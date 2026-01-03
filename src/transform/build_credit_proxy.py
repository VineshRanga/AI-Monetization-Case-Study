"""Build credit proxy (Merton-style distance-to-default) for weekly targets."""
import math
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import Dict

from src.db.db import get_engine
from src.transform.aggregate_weekly_features import get_week_start


def pd_from_dd(dd: float) -> float:
    """
    Numerically-stable logistic transform: pd_proxy = 1/(1+exp(dd)).
    
    This function is monotone decreasing in dd (higher dd = lower PD).
    Clamps dd to [-50, 50] to prevent overflow.
    
    Args:
        dd: Distance to default (can be any float)
    
    Returns:
        PD proxy in [0, 1] range
    """
    # Safety clamp and NaN check
    if dd is None or math.isnan(dd):
        return float("nan")
    
    # Clamp dd to prevent overflow
    dd = max(min(dd, 50.0), -50.0)
    
    # Numerically stable computation
    if dd >= 0:
        # For positive dd, use exp(-dd) which is safe (exp(-50) ≈ 1.9e-22)
        z = math.exp(-dd)
        return z / (1.0 + z)
    else:
        # For negative dd, use exp(dd) which is safe (dd < 0, so exp(dd) < 1)
        z = math.exp(dd)
        return 1.0 / (1.0 + z)


def build_credit_proxy() -> Dict:
    """
    Build credit proxy using simplified Merton-style distance-to-default.
    
    Method: MERTON_SIMPLIFIED_V1
    
    For each issuer-week:
    1. Equity market cap: E = shares_outstanding * equity_price
    2. Equity volatility: sigma_E = eq_vol_21d
    3. Debt proxy: D = total_debt (latest quarterly, lagged)
    4. Risk-free rate: r = DGS2 (weekly level)
    5. Asset value: V = E + D
    6. Asset volatility: sigma_V = sigma_E * (E / V)
    7. Default point: DP = 0.5 * D
    8. Distance to default: dd = (ln(V/DP) + (r - 0.5*sigma_V^2)*T) / (sigma_V * sqrt(T))
    9. PD proxy: pd_proxy = 1 / (1 + exp(dd))
    10. credit_proxy_level = pd_proxy
    11. dcredit_proxy = pd_proxy - lag(pd_proxy)
    """
    engine = get_engine()
    
    # Get all issuer-weeks with required data
    with engine.connect() as conn:
        issuer_weeks_df = pd.read_sql(
            text("""
                SELECT 
                    f.issuer_id,
                    f.week_start,
                    f.eq_vol_21d,
                    i.ticker,
                    f.net_debt
                FROM feat_issuer_weekly f
                JOIN dim_issuer i ON f.issuer_id = i.issuer_id
                WHERE f.eq_vol_21d IS NOT NULL
                ORDER BY f.issuer_id, f.week_start
            """),
            conn
        )
    
    if issuer_weeks_df.empty:
        print("No issuer-weeks with equity volatility found.")
        return {"rows_created": 0}
    
    # Convert numeric columns from Decimal to float
    numeric_cols = ["eq_vol_21d", "net_debt"]
    for col in numeric_cols:
        if col in issuer_weeks_df.columns:
            issuer_weeks_df[col] = pd.to_numeric(issuer_weeks_df[col], errors="coerce")
    
    # Get equity prices
    with engine.connect() as conn:
        tickers = issuer_weeks_df["ticker"].unique().tolist()
        ticker_str = "', '".join(tickers)
        
        equity_df = pd.read_sql(
            text(f"""
                SELECT symbol, date, adj_close
                FROM fact_equity_price_daily
                WHERE symbol IN ('{ticker_str}')
                ORDER BY symbol, date
            """),
            conn
        )
    
    equity_df["date"] = pd.to_datetime(equity_df["date"]).dt.date
    equity_df["week_start"] = equity_df["date"].apply(get_week_start)
    
    # Convert numeric columns from Decimal to float
    if "adj_close" in equity_df.columns:
        equity_df["adj_close"] = pd.to_numeric(equity_df["adj_close"], errors="coerce")
    
    # Get shares outstanding (latest per issuer)
    with engine.connect() as conn:
        shares_df = pd.read_sql(
            text("""
                SELECT issuer_id, shares_outstanding, period_end
                FROM fact_fundamentals_quarterly
                WHERE shares_outstanding IS NOT NULL
                ORDER BY issuer_id, period_end DESC
            """),
            conn
        )
    
    # Convert numeric columns from Decimal to float
    if "shares_outstanding" in shares_df.columns:
        shares_df["shares_outstanding"] = pd.to_numeric(shares_df["shares_outstanding"], errors="coerce")
    
    # Get latest shares per issuer
    latest_shares = shares_df.groupby("issuer_id").first().reset_index()
    
    # Get risk-free rate (DGS2) weekly levels
    with engine.connect() as conn:
        dgs2_df = pd.read_sql(
            text("""
                SELECT date, close as rate
                FROM fact_equity_price_daily
                WHERE symbol = 'DGS2'
                ORDER BY date
            """),
            conn
        )
    
    dgs2_df["date"] = pd.to_datetime(dgs2_df["date"]).dt.date
    dgs2_df["week_start"] = dgs2_df["date"].apply(get_week_start)
    
    # Convert numeric columns from Decimal to float
    if "rate" in dgs2_df.columns:
        dgs2_df["rate"] = pd.to_numeric(dgs2_df["rate"], errors="coerce")
    
    dgs2_weekly = dgs2_df.groupby("week_start")["rate"].last().reset_index()
    dgs2_weekly["rate"] = pd.to_numeric(dgs2_weekly["rate"], errors="coerce") / 100.0  # Convert to decimal
    
    # Get total_debt from fundamentals (latest per issuer-week with lag)
    with engine.connect() as conn:
        debt_df = pd.read_sql(
            text("""
                SELECT issuer_id, period_end, total_debt, debt_current, long_term_debt
                FROM fact_fundamentals_quarterly
                ORDER BY issuer_id, period_end
            """),
            conn
        )
    
    debt_df["period_end"] = pd.to_datetime(debt_df["period_end"]).dt.date
    
    # Convert numeric columns from Decimal to float
    debt_numeric_cols = ["total_debt", "debt_current", "long_term_debt"]
    for col in debt_numeric_cols:
        if col in debt_df.columns:
            debt_df[col] = pd.to_numeric(debt_df[col], errors="coerce")
    
    # Build credit proxy for each issuer-week
    rows_created = 0
    T = 1.0  # 1 year horizon
    
    for _, row in issuer_weeks_df.iterrows():
        issuer_id = row["issuer_id"]
        week_start = pd.to_datetime(row["week_start"]).date()
        ticker = row["ticker"]
        eq_vol = row["eq_vol_21d"]
        net_debt = row["net_debt"]
        
        # Get equity price for this week
        ticker_equity = equity_df[
            (equity_df["symbol"] == ticker) &
            (equity_df["week_start"] == week_start)
        ]
        
        if ticker_equity.empty:
            continue
        
        equity_price = float(ticker_equity["adj_close"].iloc[-1])  # Use last price in week
        if equity_price <= 0 or pd.isna(equity_price):
            continue
        
        # Get shares outstanding
        issuer_shares = latest_shares[latest_shares["issuer_id"] == issuer_id]
        if issuer_shares.empty:
            # Fallback: approximate using equity price only (note this in method)
            shares_outstanding = None
            E = equity_price  # Use price as proxy
        else:
            shares_outstanding_val = issuer_shares["shares_outstanding"].iloc[0]
            if pd.isna(shares_outstanding_val) or shares_outstanding_val <= 0:
                shares_outstanding = None
                E = equity_price
            else:
                shares_outstanding = float(shares_outstanding_val)
                E = shares_outstanding * equity_price
        
        # Get debt (latest with lag)
        lag_cutoff = week_start - timedelta(days=45)
        issuer_debt = debt_df[
            (debt_df["issuer_id"] == issuer_id) &
            (debt_df["period_end"] <= lag_cutoff)
        ].sort_values("period_end", ascending=False)
        
        if issuer_debt.empty:
            # Try using net_debt from features if available
            if net_debt is not None:
                D = max(net_debt, 1.0)  # Floor at 1
            else:
                continue
        else:
            latest_debt = issuer_debt.iloc[0]
            total_debt = latest_debt["total_debt"]
            if pd.isna(total_debt) or total_debt is None:
                debt_current = latest_debt["debt_current"]
                long_term_debt = latest_debt["long_term_debt"]
                if pd.notna(debt_current) and pd.notna(long_term_debt):
                    total_debt = float(debt_current) + float(long_term_debt)
                elif pd.notna(long_term_debt):
                    total_debt = float(long_term_debt)
                else:
                    if net_debt is not None:
                        D = max(float(net_debt), 1.0)
                    else:
                        continue
                    total_debt = D
            else:
                total_debt = float(total_debt)
            
            D = max(total_debt, 1.0)  # Floor at 1 to avoid division by zero
        
        # Get risk-free rate
        week_dgs2 = dgs2_weekly[dgs2_weekly["week_start"] == week_start]
        if week_dgs2.empty:
            # Use nearest available rate
            week_dgs2 = dgs2_weekly[
                dgs2_weekly["week_start"] <= week_start
            ].sort_values("week_start", ascending=False)
            if week_dgs2.empty:
                r = 0.02  # Default 2%
            else:
                r_val = week_dgs2["rate"].iloc[0]
                r = float(r_val) if pd.notna(r_val) else 0.02
        else:
            r_val = week_dgs2["rate"].iloc[0]
            r = float(r_val) if pd.notna(r_val) else 0.02
        
        # Convert equity vol to decimal (it's already annualized)
        sigma_E = eq_vol / 100.0 if eq_vol > 1.0 else eq_vol
        
        # Asset value
        V = E + D
        
        # Asset volatility
        if V > 0:
            sigma_V = sigma_E * (E / V)
        else:
            continue
        
        # Default point
        DP = 0.5 * D
        
        # Distance to default
        if V > 0 and DP > 0 and sigma_V > 0:
            ln_ratio = math.log(V / DP)
            drift = (r - 0.5 * sigma_V * sigma_V) * T
            denominator = sigma_V * math.sqrt(T)
            
            if denominator > 0:
                dd = (ln_ratio + drift) / denominator
            else:
                continue
        else:
            continue
        
        # PD proxy (logistic transform) - using numerically stable function
        pd_proxy = pd_from_dd(dd)
        
        # Get previous week's pd_proxy for dcredit_proxy
        with engine.connect() as conn:
            prev_result = conn.execute(
                text("""
                    SELECT credit_proxy_level
                    FROM fact_credit_proxy_weekly
                    WHERE issuer_id = :issuer_id
                    AND week_start < :week_start
                    ORDER BY week_start DESC
                    LIMIT 1
                """),
                {"issuer_id": issuer_id, "week_start": week_start}
            ).fetchone()
        
        # Compute dcredit_proxy (change from previous week)
        dcredit_proxy = None
        if prev_result and prev_result[0] is not None:
            prev_pd = prev_result[0]
            # Convert both to float to avoid Decimal vs float TypeError
            pd_proxy_f = float(pd_proxy) if pd_proxy is not None else None
            prev_pd_f = float(prev_pd) if prev_pd is not None else None
            
            if pd_proxy_f is not None and prev_pd_f is not None:
                dcredit_proxy = pd_proxy_f - prev_pd_f
            else:
                dcredit_proxy = None
        
        # Insert/update
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO fact_credit_proxy_weekly (
                        issuer_id, week_start, dd, pd_proxy, credit_proxy_level,
                        dcredit_proxy, method
                    )
                    VALUES (
                        :issuer_id, :week_start, :dd, :pd_proxy, :credit_proxy_level,
                        :dcredit_proxy, :method
                    )
                    ON CONFLICT (issuer_id, week_start) DO UPDATE SET
                        dd = EXCLUDED.dd,
                        pd_proxy = EXCLUDED.pd_proxy,
                        credit_proxy_level = EXCLUDED.credit_proxy_level,
                        dcredit_proxy = EXCLUDED.dcredit_proxy,
                        method = EXCLUDED.method
                """),
                {
                    "issuer_id": issuer_id,
                    "week_start": week_start,
                    "dd": float(dd),
                    "pd_proxy": float(pd_proxy),
                    "credit_proxy_level": float(pd_proxy),
                    "dcredit_proxy": float(dcredit_proxy) if dcredit_proxy is not None else None,
                    "method": "MERTON_SIMPLIFIED_V1",
                }
            )
            rows_created += 1
    
    return {"rows_created": rows_created}


if __name__ == "__main__":
    result = build_credit_proxy()
    print(f"Created {result['rows_created']} credit proxy records")

