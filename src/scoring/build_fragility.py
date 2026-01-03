"""Build fragility score (0-100) with 5 pillars."""
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import Dict, Optional

from src.config.structure_opacity import STRUCTURE_OPACITY
from src.db.db import get_engine


def compute_percentile_score(value: float, series: pd.Series, higher_is_worse: bool = True) -> float:
    """Compute percentile score (0-1) where higher is worse."""
    if pd.isna(value) or len(series.dropna()) == 0:
        return 0.5  # Default to median
    
    if higher_is_worse:
        return (series < value).sum() / len(series)
    else:
        return (series > value).sum() / len(series)


def build_fragility_score(asof_date: Optional[date] = None) -> Dict:
    """Build fragility score for all issuers."""
    engine = get_engine()
    
    if asof_date is None:
        asof_date = date(2025, 12, 31)
    
    print(f"Building fragility scores as of {asof_date}")
    
    # Get latest fundamentals per issuer
    with engine.connect() as conn:
        fundamentals_df = pd.read_sql(
            text("""
                SELECT 
                    f.issuer_id,
                    i.ticker,
                    i.bucket,
                    f.period_end,
                    f.revenue,
                    f.ebitda,
                    f.op_income,
                    f.net_income,
                    f.total_debt,
                    f.cash,
                    f.net_debt,
                    f.interest_expense,
                    f.capex,
                    f.cfo,
                    f.fcf
                FROM fact_fundamentals_quarterly f
                JOIN dim_issuer i ON f.issuer_id = i.issuer_id
                WHERE f.period_end <= :asof_date
                ORDER BY f.issuer_id, f.period_end DESC
            """),
            conn,
            params={"asof_date": asof_date}
        )
    
    # Get latest per issuer
    if fundamentals_df.empty:
        print("No fundamentals data found")
        return {"rows_created": 0}
    
    latest_fundamentals = fundamentals_df.groupby("issuer_id").first().reset_index()
    
    # Get capital structure
    with engine.connect() as conn:
        cap_structure_df = pd.read_sql(
            text("""
                SELECT 
                    c.issuer_id,
                    c.maturity_24m_pct
                FROM fact_cap_structure c
                WHERE c.asof_date = :asof_date
            """),
            conn,
            params={"asof_date": asof_date}
        )
    
    # Get equity volatility (52-week rolling)
    with engine.connect() as conn:
        equity_vol_df = pd.read_sql(
            text("""
                SELECT 
                    f.issuer_id,
                    AVG(f.eq_vol_21d) as avg_eq_vol
                FROM feat_issuer_weekly f
                WHERE f.week_start >= :start_date
                    AND f.week_start <= :asof_date
                GROUP BY f.issuer_id
            """),
            conn,
            params={
                "start_date": date(asof_date.year - 1, asof_date.month, asof_date.day),
                "asof_date": asof_date
            }
        )
    
    # Merge data
    df = latest_fundamentals.merge(
        cap_structure_df,
        on="issuer_id",
        how="left"
    ).merge(
        equity_vol_df,
        on="issuer_id",
        how="left"
    )
    
    # Compute raw inputs
    inputs = []
    
    for _, row in df.iterrows():
        issuer_id = row["issuer_id"]
        ticker = row["ticker"]
        bucket = row["bucket"]
        
        # 1. Refi/Funding (25%)
        maturity_24m_pct = row.get("maturity_24m_pct")
        floating_pct = None  # Not available from current data
        
        # 2. Cash vs Capex (25%)
        fcf_after_capex = row.get("fcf")
        capex_intensity = None
        if row.get("revenue") and row.get("revenue") > 0 and row.get("capex"):
            capex_intensity = row["capex"] / row["revenue"]
        
        capex_commitment_proxy = None
        if bucket == "DATACENTER":
            # For DC: capex_intensity + leverage proxy
            if capex_intensity and row.get("net_debt") and row.get("revenue") and row["revenue"] > 0:
                leverage_proxy = row["net_debt"] / row["revenue"]
                capex_commitment_proxy = capex_intensity + (leverage_proxy * 0.1)
        else:
            # For others: use capex_intensity as proxy
            capex_commitment_proxy = capex_intensity
        
        # 3. Leverage/Coverage (20%)
        net_debt_ebitda = None
        if row.get("net_debt") and row.get("ebitda") and row["ebitda"] > 0:
            net_debt_ebitda = row["net_debt"] / row["ebitda"]
        elif row.get("net_debt") and row.get("revenue") and row["revenue"] > 0:
            # Proxy: net_debt / revenue
            net_debt_ebitda = row["net_debt"] / row["revenue"] * 10  # Scale factor
        
        int_coverage = None
        if row.get("interest_expense") and row["interest_expense"] > 0:
            if row.get("op_income"):
                int_coverage = row["op_income"] / row["interest_expense"]
            elif row.get("ebitda"):
                int_coverage = row["ebitda"] / row["interest_expense"]
        
        # 4. Cyclicality + AI concentration (20%)
        cyclicality_proxy = row.get("avg_eq_vol")
        if pd.isna(cyclicality_proxy):
            cyclicality_proxy = None
        
        # AI concentration (bucket-based priors)
        ai_concentration_map = {
            "SEMIS": 0.8,  # Highest
            "DATACENTER": 0.6,  # High
            "HYPERSCALER": 0.4,  # Medium
        }
        ai_concentration = ai_concentration_map.get(bucket, 0.5)
        
        # 5. Structure/Opacity (10%)
        structure_opacity_flag = STRUCTURE_OPACITY.get(ticker, 1)
        
        inputs.append({
            "issuer_id": issuer_id,
            "asof_date": asof_date,
            "maturity_24m_pct": maturity_24m_pct,
            "floating_pct": floating_pct,
            "fcf_after_capex": fcf_after_capex,
            "capex_intensity": capex_intensity,
            "capex_commitment_proxy": capex_commitment_proxy,
            "net_debt_ebitda": net_debt_ebitda,
            "int_coverage": int_coverage,
            "cyclicality_proxy": cyclicality_proxy,
            "ai_concentration": ai_concentration,
            "structure_opacity_flag": structure_opacity_flag,
        })
    
    inputs_df = pd.DataFrame(inputs)
    
    # Compute percentile scores within bucket
    scores = []
    
    for bucket in inputs_df["bucket"].unique():
        bucket_df = inputs_df[inputs_df["bucket"] == bucket].copy()
        
        # Impute missing values with bucket median
        for col in bucket_df.columns:
            if col not in ["issuer_id", "asof_date", "bucket"]:
                median_val = bucket_df[col].median()
                bucket_df[col] = bucket_df[col].fillna(median_val)
        
        # Pillar 1: Refi/Funding (25%)
        p_refi_components = []
        if not bucket_df["maturity_24m_pct"].isna().all():
            p_refi_components.append(
                bucket_df["maturity_24m_pct"].apply(
                    lambda x: compute_percentile_score(x, bucket_df["maturity_24m_pct"], higher_is_worse=True)
                )
            )
        if not bucket_df["floating_pct"].isna().all():
            p_refi_components.append(
                bucket_df["floating_pct"].apply(
                    lambda x: compute_percentile_score(x, bucket_df["floating_pct"], higher_is_worse=True)
                )
            )
        
        if p_refi_components:
            bucket_df["p_refi"] = pd.concat(p_refi_components, axis=1).mean(axis=1)
        else:
            bucket_df["p_refi"] = 0.5
        
        # Pillar 2: Cash/Capex (25%)
        p_cash_capex_components = []
        if not bucket_df["fcf_after_capex"].isna().all():
            p_cash_capex_components.append(
                bucket_df["fcf_after_capex"].apply(
                    lambda x: compute_percentile_score(x, bucket_df["fcf_after_capex"], higher_is_worse=False)  # Lower FCF is worse
                )
            )
        if not bucket_df["capex_intensity"].isna().all():
            p_cash_capex_components.append(
                bucket_df["capex_intensity"].apply(
                    lambda x: compute_percentile_score(x, bucket_df["capex_intensity"], higher_is_worse=True)
                )
            )
        if not bucket_df["capex_commitment_proxy"].isna().all():
            p_cash_capex_components.append(
                bucket_df["capex_commitment_proxy"].apply(
                    lambda x: compute_percentile_score(x, bucket_df["capex_commitment_proxy"], higher_is_worse=True)
                )
            )
        
        if p_cash_capex_components:
            bucket_df["p_cash_capex"] = pd.concat(p_cash_capex_components, axis=1).mean(axis=1)
        else:
            bucket_df["p_cash_capex"] = 0.5
        
        # Pillar 3: Leverage (20%)
        p_leverage_components = []
        if not bucket_df["net_debt_ebitda"].isna().all():
            p_leverage_components.append(
                bucket_df["net_debt_ebitda"].apply(
                    lambda x: compute_percentile_score(x, bucket_df["net_debt_ebitda"], higher_is_worse=True)
                )
            )
        if not bucket_df["int_coverage"].isna().all():
            p_leverage_components.append(
                bucket_df["int_coverage"].apply(
                    lambda x: compute_percentile_score(x, bucket_df["int_coverage"], higher_is_worse=False)  # Lower coverage is worse
                )
            )
        
        if p_leverage_components:
            bucket_df["p_leverage"] = pd.concat(p_leverage_components, axis=1).mean(axis=1)
        else:
            bucket_df["p_leverage"] = 0.5
        
        # Pillar 4: Cyclicality + AI (20%)
        p_cyc_ai_components = []
        if not bucket_df["cyclicality_proxy"].isna().all():
            p_cyc_ai_components.append(
                bucket_df["cyclicality_proxy"].apply(
                    lambda x: compute_percentile_score(x, bucket_df["cyclicality_proxy"], higher_is_worse=True)
                )
            )
        if not bucket_df["ai_concentration"].isna().all():
            p_cyc_ai_components.append(
                bucket_df["ai_concentration"].apply(
                    lambda x: compute_percentile_score(x, bucket_df["ai_concentration"], higher_is_worse=True)
                )
            )
        
        if p_cyc_ai_components:
            bucket_df["p_cyc_ai"] = pd.concat(p_cyc_ai_components, axis=1).mean(axis=1)
        else:
            bucket_df["p_cyc_ai"] = 0.5
        
        # Pillar 5: Structure (10%)
        bucket_df["p_structure"] = bucket_df["structure_opacity_flag"] / 2.0  # Normalize 0-2 to 0-1
        
        # Total score (weighted)
        bucket_df["total_score"] = (
            bucket_df["p_refi"] * 0.25 +
            bucket_df["p_cash_capex"] * 0.25 +
            bucket_df["p_leverage"] * 0.20 +
            bucket_df["p_cyc_ai"] * 0.20 +
            bucket_df["p_structure"] * 0.10
        ) * 100  # Scale to 0-100
        
        scores.append(bucket_df)
    
    scores_df = pd.concat(scores, ignore_index=True)
    
    # Save inputs
    with engine.begin() as conn:
        for _, row in inputs_df.iterrows():
            conn.execute(
                text("""
                    INSERT INTO fragility_inputs (
                        issuer_id, asof_date,
                        maturity_24m_pct, floating_pct,
                        fcf_after_capex, capex_intensity, capex_commitment_proxy,
                        net_debt_ebitda, int_coverage,
                        cyclicality_proxy, ai_concentration,
                        structure_opacity_flag
                    )
                    VALUES (
                        :issuer_id, :asof_date,
                        :maturity_24m_pct, :floating_pct,
                        :fcf_after_capex, :capex_intensity, :capex_commitment_proxy,
                        :net_debt_ebitda, :int_coverage,
                        :cyclicality_proxy, :ai_concentration,
                        :structure_opacity_flag
                    )
                    ON CONFLICT (issuer_id, asof_date) DO UPDATE SET
                        maturity_24m_pct = EXCLUDED.maturity_24m_pct,
                        floating_pct = EXCLUDED.floating_pct,
                        fcf_after_capex = EXCLUDED.fcf_after_capex,
                        capex_intensity = EXCLUDED.capex_intensity,
                        capex_commitment_proxy = EXCLUDED.capex_commitment_proxy,
                        net_debt_ebitda = EXCLUDED.net_debt_ebitda,
                        int_coverage = EXCLUDED.int_coverage,
                        cyclicality_proxy = EXCLUDED.cyclicality_proxy,
                        ai_concentration = EXCLUDED.ai_concentration,
                        structure_opacity_flag = EXCLUDED.structure_opacity_flag
                """),
                {
                    "issuer_id": int(row["issuer_id"]),
                    "asof_date": row["asof_date"],
                    "maturity_24m_pct": float(row["maturity_24m_pct"]) if pd.notna(row["maturity_24m_pct"]) else None,
                    "floating_pct": float(row["floating_pct"]) if pd.notna(row["floating_pct"]) else None,
                    "fcf_after_capex": float(row["fcf_after_capex"]) if pd.notna(row["fcf_after_capex"]) else None,
                    "capex_intensity": float(row["capex_intensity"]) if pd.notna(row["capex_intensity"]) else None,
                    "capex_commitment_proxy": float(row["capex_commitment_proxy"]) if pd.notna(row["capex_commitment_proxy"]) else None,
                    "net_debt_ebitda": float(row["net_debt_ebitda"]) if pd.notna(row["net_debt_ebitda"]) else None,
                    "int_coverage": float(row["int_coverage"]) if pd.notna(row["int_coverage"]) else None,
                    "cyclicality_proxy": float(row["cyclicality_proxy"]) if pd.notna(row["cyclicality_proxy"]) else None,
                    "ai_concentration": float(row["ai_concentration"]),
                    "structure_opacity_flag": int(row["structure_opacity_flag"]),
                }
            )
    
    # Save scores
    with engine.begin() as conn:
        for _, row in scores_df.iterrows():
            notes = f"Scored as of {asof_date}. Structure opacity is a proxy flag."
            
            conn.execute(
                text("""
                    INSERT INTO fragility_score (
                        issuer_id, asof_date,
                        total_score, p_refi, p_cash_capex, p_leverage, p_cyc_ai, p_structure, notes
                    )
                    VALUES (
                        :issuer_id, :asof_date,
                        :total_score, :p_refi, :p_cash_capex, :p_leverage, :p_cyc_ai, :p_structure, :notes
                    )
                    ON CONFLICT (issuer_id, asof_date) DO UPDATE SET
                        total_score = EXCLUDED.total_score,
                        p_refi = EXCLUDED.p_refi,
                        p_cash_capex = EXCLUDED.p_cash_capex,
                        p_leverage = EXCLUDED.p_leverage,
                        p_cyc_ai = EXCLUDED.p_cyc_ai,
                        p_structure = EXCLUDED.p_structure,
                        notes = EXCLUDED.notes
                """),
                {
                    "issuer_id": int(row["issuer_id"]),
                    "asof_date": row["asof_date"],
                    "total_score": float(row["total_score"]),
                    "p_refi": float(row["p_refi"] * 100),
                    "p_cash_capex": float(row["p_cash_capex"] * 100),
                    "p_leverage": float(row["p_leverage"] * 100),
                    "p_cyc_ai": float(row["p_cyc_ai"] * 100),
                    "p_structure": float(row["p_structure"] * 100),
                    "notes": notes,
                }
            )
    
    # Export CSV
    output_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scores_export = scores_df[["ticker", "bucket", "total_score", "p_refi", "p_cash_capex", "p_leverage", "p_cyc_ai", "p_structure"]].copy()
    scores_export.to_csv(output_dir / "fragility_scores.csv", index=False)
    
    return {
        "rows_created": len(scores_df),
        "scores_df": scores_df,
    }

