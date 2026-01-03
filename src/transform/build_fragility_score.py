"""Build 5-pillar Fragility Score for issuers (EDGAR-first)."""
import json
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sqlalchemy import text
from typing import Dict, Optional

from src.db.db import get_engine


def winsorize_and_scale(values: pd.Series, higher_is_worse: bool = True) -> pd.Series:
    """
    Winsorize at 5th/95th percentiles and scale to 0-100.
    
    Args:
        values: Series of raw values
        higher_is_worse: If True, higher raw values -> higher scores (0-100)
                        If False, lower raw values -> higher scores (0-100)
    
    Returns:
        Series of scores (0-100)
    """
    if values.empty or values.isna().all():
        return pd.Series([0.0] * len(values), index=values.index)
    
    # Winsorize at 5th and 95th percentiles
    p5 = values.quantile(0.05)
    p95 = values.quantile(0.95)
    
    values_winsorized = values.clip(lower=p5, upper=p95)
    
    # Min-max scale to 0-100
    v_min = values_winsorized.min()
    v_max = values_winsorized.max()
    
    if v_max == v_min:
        # All values are the same
        return pd.Series([50.0] * len(values), index=values.index)
    
    scaled = ((values_winsorized - v_min) / (v_max - v_min)) * 100.0
    
    if not higher_is_worse:
        # Invert: lower raw values -> higher scores
        scaled = 100.0 - scaled
    
    return scaled


def compute_beta_and_drawdown(issuer_returns: pd.Series, market_returns: pd.Series) -> tuple:
    """
    Compute beta vs market and max drawdown.
    
    Returns:
        (beta, max_drawdown)
    """
    # Align dates
    aligned = pd.DataFrame({
        "issuer": issuer_returns,
        "market": market_returns
    }).dropna()
    
    if len(aligned) < 20:
        return (1.0, 0.0)  # Defaults
    
    # Beta: covariance / variance of market
    cov = aligned["issuer"].cov(aligned["market"])
    var_market = aligned["market"].var()
    beta = cov / var_market if var_market > 0 else 1.0
    
    # Max drawdown: peak-to-trough decline
    cumulative = (1 + aligned["issuer"]).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min()) if drawdown.min() < 0 else 0.0
    
    return (beta, max_drawdown)


def build_fragility_score(asof_date: Optional[date] = None) -> Dict:
    """
    Build 5-pillar Fragility Score for all issuers.
    
    Pillars:
    1. Refinancing wall / funding risk
    2. Cash generation vs committed capex
    3. Leverage / coverage
    4. Cyclicality & AI concentration
    5. Structure / opacity
    
    Args:
        asof_date: As-of date for scoring (default: latest quarter end in 2025)
    
    Returns:
        Dict with rows_created, rows_updated
    """
    engine = get_engine()
    
    if asof_date is None:
        # Use latest quarter end in 2025
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT MAX(period_end) as latest_date
                    FROM fact_fundamentals_quarterly
                    WHERE period_end <= '2025-12-31'
                """)
            )
            row = result.fetchone()
            if row and row[0]:
                asof_date = row[0] if isinstance(row[0], date) else row[0].date()
            else:
                asof_date = date(2025, 12, 31)
    
    print(f"\nBuilding Fragility Scores as of {asof_date}")
    
    # Get all issuers
    with engine.connect() as conn:
        issuers_df = pd.read_sql(
            text("SELECT issuer_id, ticker, bucket FROM dim_issuer ORDER BY ticker"),
            conn
        )
    
    if issuers_df.empty:
        print("No issuers found")
        return {"rows_created": 0, "rows_updated": 0}
    
    # Get latest fundamentals per issuer (most recent quarter <= asof_date)
    with engine.connect() as conn:
        fundamentals_df = pd.read_sql(
            text("""
                SELECT DISTINCT ON (issuer_id)
                    issuer_id,
                    period_end,
                    total_debt,
                    debt_current,
                    long_term_debt,
                    interest_expense,
                    cash,
                    cfo,
                    capex,
                    revenue,
                    op_income,
                    net_income,
                    fcf
                FROM fact_fundamentals_quarterly
                WHERE period_end <= :asof_date
                ORDER BY issuer_id, period_end DESC
            """),
            conn,
            params={"asof_date": asof_date}
        )
    
    # Compute EBITDA proxy (op_income if available, else use net_income as fallback)
    # Note: We don't have depreciation separately, so op_income is our best proxy
    fundamentals_df["ebitda"] = fundamentals_df["op_income"].fillna(fundamentals_df["net_income"]).fillna(0.0)
    
    # Get equity stats (2024-2025) - rolling 26w vol, max drawdown
    start_date = date(2024, 1, 1)
    with engine.connect() as conn:
        equity_stats_df = pd.read_sql(
            text("""
                SELECT 
                    f.issuer_id,
                    AVG(f.eq_vol_21d) as avg_eq_vol,
                    STDDEV(f.eq_ret) as std_eq_ret
                FROM feat_issuer_weekly f
                WHERE f.week_start >= :start_date
                    AND f.week_start <= :asof_date
                GROUP BY f.issuer_id
            """),
            conn,
            params={"start_date": start_date, "asof_date": asof_date}
        )
    
    # Get equity prices for beta and drawdown calculation
    tickers = issuers_df["ticker"].tolist()
    ticker_str = "', '".join(tickers)
    
    with engine.connect() as conn:
        equity_prices_df = pd.read_sql(
            text(f"""
                SELECT symbol, date, adj_close
                FROM fact_equity_price_daily
                WHERE symbol IN ('{ticker_str}')
                    AND date >= :start_date
                    AND date <= :asof_date
                ORDER BY symbol, date
            """),
            conn,
            params={"start_date": start_date, "asof_date": asof_date}
        )
    
    # Get QQQ prices for beta
    with engine.connect() as conn:
        qqq_df = pd.read_sql(
            text("""
                SELECT date, adj_close
                FROM fact_equity_price_daily
                WHERE symbol = 'QQQ'
                    AND date >= :start_date
                    AND date <= :asof_date
                ORDER BY date
            """),
            conn,
            params={"start_date": start_date, "asof_date": asof_date}
        )
    
    # Compute QQQ returns
    if not qqq_df.empty:
        qqq_df = qqq_df.sort_values("date")
        qqq_df["ret"] = qqq_df["adj_close"].pct_change()
        qqq_returns = qqq_df.set_index("date")["ret"]
    else:
        qqq_returns = pd.Series(dtype=float)
    
    # Merge data
    df = issuers_df.merge(fundamentals_df, on="issuer_id", how="left")
    df = df.merge(equity_stats_df, on="issuer_id", how="left")
    
    # Compute pillar scores
    pillar_rows = []
    
    for _, row in df.iterrows():
        issuer_id = row["issuer_id"]
        ticker = row["ticker"]
        bucket = row["bucket"]
        
        # Get equity returns for this issuer
        issuer_equity = equity_prices_df[equity_prices_df["symbol"] == ticker].copy()
        if not issuer_equity.empty:
            issuer_equity = issuer_equity.sort_values("date")
            issuer_equity["ret"] = issuer_equity["adj_close"].pct_change()
            issuer_returns = issuer_equity.set_index("date")["ret"]
            beta, max_drawdown = compute_beta_and_drawdown(issuer_returns, qqq_returns)
        else:
            beta = 1.0
            max_drawdown = 0.0
        
        # Convert to float
        total_debt = float(row["total_debt"]) if pd.notna(row["total_debt"]) else 0.0
        debt_current = float(row["debt_current"]) if pd.notna(row["debt_current"]) else 0.0
        interest_expense = float(row["interest_expense"]) if pd.notna(row["interest_expense"]) else 0.0
        cash = float(row["cash"]) if pd.notna(row["cash"]) else 0.0
        cfo = float(row["cfo"]) if pd.notna(row["cfo"]) else 0.0
        capex = float(row["capex"]) if pd.notna(row["capex"]) else 0.0
        revenue = float(row["revenue"]) if pd.notna(row["revenue"]) else 0.0
        ebitda = float(row["ebitda"]) if pd.notna(row["ebitda"]) else 0.0
        avg_eq_vol = float(row["avg_eq_vol"]) if pd.notna(row["avg_eq_vol"]) else 0.0
        
        # PILLAR 1: Refinancing wall / funding risk
        # Raw metrics: debt_to_cash, interest_burden
        raw_p1 = {}
        if total_debt > 0:
            debt_to_cash = total_debt / max(cash, 1.0)
            raw_p1["debt_to_cash"] = debt_to_cash
        else:
            debt_to_cash = 0.0
            raw_p1["debt_to_cash"] = 0.0
        
        if revenue > 0:
            interest_burden = (interest_expense * 4) / revenue  # Annualize quarterly
            raw_p1["interest_burden"] = interest_burden
        else:
            interest_burden = 0.0
            raw_p1["interest_burden"] = 0.0
        
        # Combined: higher debt_to_cash and interest_burden = higher risk
        pillar1_raw = debt_to_cash + (interest_burden * 10)
        
        # PILLAR 2: Cash generation vs committed capex
        # Raw metrics: fcf_margin, capex_intensity
        raw_p2 = {}
        if revenue > 0:
            if cfo > 0 and capex > 0:
                fcf = cfo - capex
                fcf_margin = fcf / revenue
                raw_p2["fcf_margin"] = fcf_margin
            else:
                fcf_margin = 0.0
                raw_p2["fcf_margin"] = 0.0
            
            capex_intensity = capex / revenue
            raw_p2["capex_intensity"] = capex_intensity
            
            # Risk: negative FCF margin OR high capex intensity
            pillar2_raw = -fcf_margin + (capex_intensity * 2)
        else:
            pillar2_raw = 0.0
            raw_p2["fcf_margin"] = 0.0
            raw_p2["capex_intensity"] = 0.0
        
        # PILLAR 3: Leverage / coverage
        # Raw metrics: net_debt_to_assets (proxy), inverse_coverage
        raw_p3 = {}
        if ebitda > 0:
            net_debt = max(total_debt - cash, 0.0)
            net_debt_ebitda = net_debt / ebitda
            raw_p3["net_debt_ebitda"] = net_debt_ebitda
            
            if interest_expense > 0:
                int_coverage = ebitda / (interest_expense * 4)  # Annualize
                inverse_coverage = 1.0 / max(int_coverage, 0.01)
                raw_p3["inverse_coverage"] = inverse_coverage
            else:
                inverse_coverage = 0.0
                raw_p3["inverse_coverage"] = 0.0
            
            # Use revenue as proxy for assets (rough approximation)
            if revenue > 0:
                net_debt_to_assets_proxy = net_debt / revenue
                raw_p3["net_debt_to_assets_proxy"] = net_debt_to_assets_proxy
            else:
                net_debt_to_assets_proxy = 0.0
                raw_p3["net_debt_to_assets_proxy"] = 0.0
            
            pillar3_raw = net_debt_ebitda + (inverse_coverage * 5) + (net_debt_to_assets_proxy * 0.1)
        else:
            pillar3_raw = 0.0
            raw_p3["net_debt_ebitda"] = 0.0
            raw_p3["inverse_coverage"] = 0.0
            raw_p3["net_debt_to_assets_proxy"] = 0.0
        
        # PILLAR 4: Cyclicality & AI concentration
        # Raw metrics: equity vol, drawdown, beta vs QQQ, AI bucket baseline
        raw_p4 = {}
        
        # AI exposure bucket baselines
        bucket_baselines = {
            "HYPERSCALER": 50,
            "SEMIS": 65,
            "DATACENTER": 55,
        }
        ai_baseline = bucket_baselines.get(bucket, 50)
        raw_p4["ai_baseline"] = ai_baseline
        
        vol_risk = avg_eq_vol * 100 if avg_eq_vol > 0 else 0.0
        raw_p4["eq_vol_26w"] = avg_eq_vol
        
        beta_risk = abs(beta - 1.0) * 20  # Deviation from market beta
        raw_p4["beta_vs_qqq"] = beta
        
        drawdown_risk = max_drawdown * 100
        raw_p4["max_drawdown_2024_2025"] = max_drawdown
        
        # Cyclicality score (0-100)
        cyc_score = winsorize_and_scale(
            pd.Series([vol_risk + beta_risk + drawdown_risk]),
            higher_is_worse=True
        ).iloc[0]
        
        # Combined: 60% cyclicality, 40% AI baseline
        pillar4_raw = (0.6 * cyc_score) + (0.4 * ai_baseline)
        
        # PILLAR 5: Structure / opacity
        # Since we don't have intangibles/goodwill, use debt complexity as proxy
        # Higher debt complexity (current vs long-term mix) = more opacity
        raw_p5 = {}
        if total_debt > 0:
            if debt_current > 0:
                st_debt_share = debt_current / total_debt
            else:
                st_debt_share = 0.0
            raw_p5["st_debt_share"] = st_debt_share
            
            # Opacity proxy: high ST debt share OR complex debt structure
            pillar5_raw = st_debt_share * 100
        else:
            pillar5_raw = 0.0
            raw_p5["st_debt_share"] = 0.0
        
        # Store raw values with JSON
        pillar_rows.append({
            "issuer_id": issuer_id,
            "pillar": "refinancing_wall",
            "raw_json": json.dumps(raw_p1),
            "raw_value": pillar1_raw,
        })
        pillar_rows.append({
            "issuer_id": issuer_id,
            "pillar": "cash_generation",
            "raw_json": json.dumps(raw_p2),
            "raw_value": pillar2_raw,
        })
        pillar_rows.append({
            "issuer_id": issuer_id,
            "pillar": "leverage_coverage",
            "raw_json": json.dumps(raw_p3),
            "raw_value": pillar3_raw,
        })
        pillar_rows.append({
            "issuer_id": issuer_id,
            "pillar": "cyclicality_ai",
            "raw_json": json.dumps(raw_p4),
            "raw_value": pillar4_raw,
        })
        pillar_rows.append({
            "issuer_id": issuer_id,
            "pillar": "structure_opacity",
            "raw_json": json.dumps(raw_p5),
            "raw_value": pillar5_raw,
        })
    
    # Convert to DataFrame for scaling
    pillar_df = pd.DataFrame(pillar_rows)
    
    # Scale each pillar to 0-100
    scored_rows = []
    for pillar_name in ["refinancing_wall", "cash_generation", "leverage_coverage", "cyclicality_ai", "structure_opacity"]:
        pillar_data = pillar_df[pillar_df["pillar"] == pillar_name].copy()
        if not pillar_data.empty:
            raw_values = pillar_data["raw_value"]
            scores = winsorize_and_scale(raw_values, higher_is_worse=True)
            
            for idx, (issuer_id, raw_val, raw_json) in enumerate(zip(
                pillar_data["issuer_id"], 
                raw_values, 
                pillar_data["raw_json"]
            )):
                scored_rows.append({
                    "issuer_id": issuer_id,
                    "pillar": pillar_name,
                    "raw_json": raw_json,
                    "raw_value": float(raw_val) if pd.notna(raw_val) else None,
                    "score_0_100": float(scores.iloc[idx]) if pd.notna(scores.iloc[idx]) else 0.0,
                })
    
    # Compute total scores (equal weights: 20% each)
    weights = {
        "refinancing_wall": 0.20,
        "cash_generation": 0.20,
        "leverage_coverage": 0.20,
        "cyclicality_ai": 0.20,
        "structure_opacity": 0.20,
    }
    
    total_scores = []
    for issuer_id in issuers_df["issuer_id"]:
        issuer_pillars = [r for r in scored_rows if r["issuer_id"] == issuer_id]
        if issuer_pillars:
            total_score = sum(r["score_0_100"] * weights[r["pillar"]] for r in issuer_pillars)
            total_scores.append({
                "issuer_id": issuer_id,
                "total_score_0_100": total_score,
            })
    
    # Upsert pillar scores
    rows_created = 0
    rows_updated = 0
    
    with engine.begin() as conn:
        for row in scored_rows:
            result = conn.execute(
                text("""
                    INSERT INTO fragility_pillar_scores (
                        issuer_id, asof_date, pillar_name, raw_json, score_0_100, method
                    )
                    VALUES (
                        :issuer_id, :asof_date, :pillar_name, :raw_json, :score_0_100, :method
                    )
                    ON CONFLICT (issuer_id, asof_date, pillar_name) DO UPDATE SET
                        raw_json = EXCLUDED.raw_json,
                        score_0_100 = EXCLUDED.score_0_100,
                        method = EXCLUDED.method,
                        created_at = NOW()
                    RETURNING (xmax = 0) as is_new
                """),
                {
                    "issuer_id": row["issuer_id"],
                    "asof_date": asof_date,
                    "pillar_name": row["pillar"],
                    "raw_json": row["raw_json"],
                    "score_0_100": row["score_0_100"],
                    "method": "EDGAR_FIRST_V1",
                }
            )
            if result.fetchone()[0]:
                rows_created += 1
            else:
                rows_updated += 1
    
    # Upsert total scores
    with engine.begin() as conn:
        for row in total_scores:
            conn.execute(
                text("""
                    INSERT INTO fragility_scores (
                        issuer_id, asof_date, total_score_0_100, weights_json, notes
                    )
                    VALUES (
                        :issuer_id, :asof_date, :total_score_0_100, :weights_json, :notes
                    )
                    ON CONFLICT (issuer_id, asof_date) DO UPDATE SET
                        total_score_0_100 = EXCLUDED.total_score_0_100,
                        weights_json = EXCLUDED.weights_json,
                        notes = EXCLUDED.notes,
                        created_at = NOW()
                """),
                {
                    "issuer_id": row["issuer_id"],
                    "asof_date": asof_date,
                    "total_score_0_100": row["total_score_0_100"],
                    "weights_json": json.dumps(weights),
                    "notes": "5-pillar fragility score (EDGAR-first). Equal weights. Macro FRED not loaded.",
                }
            )
    
    print(f"  Created {rows_created} pillar scores, updated {rows_updated}")
    print(f"  Created/updated {len(total_scores)} total fragility scores")
    
    return {
        "rows_created": rows_created,
        "rows_updated": rows_updated,
        "total_scores": len(total_scores),
        "asof_date": asof_date,
    }
