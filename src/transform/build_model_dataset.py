"""Build final modeling dataset from weekly features and credit proxy target."""
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import Dict

from src.db.db import get_engine


def build_model_dataset() -> Dict:
    """
    Build final modeling dataset by joining:
    - fact_credit_proxy_weekly (target: dcredit_proxy)
    - feat_issuer_weekly (issuer features)
    - fact_weekly_market (market features)
    - model_regime_weekly (regime)
    
    Persist to model_dataset_weekly.
    """
    engine = get_engine()
    
    # Join all tables
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT 
                    c.issuer_id,
                    c.week_start,
                    c.dcredit_proxy,
                    COALESCE(r.regime_label, 'RISK_ON') AS regime_label,
                    f.eq_ret,
                    f.eq_vol_21d,
                    f.bucket,
                    f.net_debt,
                    f.net_debt_ebitda,
                    f.int_coverage,
                    f.fcf_margin,
                    f.capex_intensity,
                    m.spx_ret,
                    m.qqq_ret,
                    m.vix_chg,
                    m.dgs2_chg,
                    m.dgs10_chg,
                    m.ig_oas_chg
                FROM fact_credit_proxy_weekly c
                JOIN feat_issuer_weekly f ON c.issuer_id = f.issuer_id AND c.week_start = f.week_start
                LEFT JOIN fact_weekly_market m ON c.week_start = m.week_start
                LEFT JOIN model_regime_weekly r ON c.week_start = r.week_start
                WHERE c.dcredit_proxy IS NOT NULL
                ORDER BY c.issuer_id, c.week_start
            """),
            conn
        )
    
    if df.empty:
        print("No data found. Run credit proxy and feature builders first.")
        return {"rows_created": 0}
    
    # Upsert into model_dataset_weekly
    rows_created = 0
    
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text("""
                    INSERT INTO model_dataset_weekly (
                        issuer_id, week_start, dcredit_proxy, regime_label,
                        eq_ret, eq_vol_21d, bucket,
                        net_debt, net_debt_ebitda, int_coverage,
                        fcf_margin, capex_intensity,
                        spx_ret, qqq_ret, vix_chg,
                        dgs2_chg, dgs10_chg, ig_oas_chg
                    )
                    VALUES (
                        :issuer_id, :week_start, :dcredit_proxy, :regime_label,
                        :eq_ret, :eq_vol_21d, :bucket,
                        :net_debt, :net_debt_ebitda, :int_coverage,
                        :fcf_margin, :capex_intensity,
                        :spx_ret, :qqq_ret, :vix_chg,
                        :dgs2_chg, :dgs10_chg, :ig_oas_chg
                    )
                    ON CONFLICT (issuer_id, week_start) DO UPDATE SET
                        dcredit_proxy = EXCLUDED.dcredit_proxy,
                        regime_label = EXCLUDED.regime_label,
                        eq_ret = EXCLUDED.eq_ret,
                        eq_vol_21d = EXCLUDED.eq_vol_21d,
                        bucket = EXCLUDED.bucket,
                        net_debt = EXCLUDED.net_debt,
                        net_debt_ebitda = EXCLUDED.net_debt_ebitda,
                        int_coverage = EXCLUDED.int_coverage,
                        fcf_margin = EXCLUDED.fcf_margin,
                        capex_intensity = EXCLUDED.capex_intensity,
                        spx_ret = EXCLUDED.spx_ret,
                        qqq_ret = EXCLUDED.qqq_ret,
                        vix_chg = EXCLUDED.vix_chg,
                        dgs2_chg = EXCLUDED.dgs2_chg,
                        dgs10_chg = EXCLUDED.dgs10_chg,
                        ig_oas_chg = EXCLUDED.ig_oas_chg
                """),
                {
                    "issuer_id": int(row["issuer_id"]),
                    "week_start": pd.to_datetime(row["week_start"]).date(),
                    "dcredit_proxy": float(row["dcredit_proxy"]) if pd.notna(row["dcredit_proxy"]) else None,
                    "regime_label": str(row["regime_label"]) if pd.notna(row["regime_label"]) else "RISK_ON",
                    "eq_ret": float(row["eq_ret"]) if pd.notna(row["eq_ret"]) else None,
                    "eq_vol_21d": float(row["eq_vol_21d"]) if pd.notna(row["eq_vol_21d"]) else None,
                    "bucket": row["bucket"] if pd.notna(row["bucket"]) else None,
                    "net_debt": float(row["net_debt"]) if pd.notna(row["net_debt"]) else None,
                    "net_debt_ebitda": float(row["net_debt_ebitda"]) if pd.notna(row["net_debt_ebitda"]) else None,
                    "int_coverage": float(row["int_coverage"]) if pd.notna(row["int_coverage"]) else None,
                    "fcf_margin": float(row["fcf_margin"]) if pd.notna(row["fcf_margin"]) else None,
                    "capex_intensity": float(row["capex_intensity"]) if pd.notna(row["capex_intensity"]) else None,
                    "spx_ret": float(row["spx_ret"]) if pd.notna(row["spx_ret"]) else None,
                    "qqq_ret": float(row["qqq_ret"]) if pd.notna(row["qqq_ret"]) else None,
                    "vix_chg": float(row["vix_chg"]) if pd.notna(row["vix_chg"]) else None,
                    "dgs2_chg": float(row["dgs2_chg"]) if pd.notna(row["dgs2_chg"]) else None,
                    "dgs10_chg": float(row["dgs10_chg"]) if pd.notna(row["dgs10_chg"]) else None,
                    "ig_oas_chg": float(row["ig_oas_chg"]) if pd.notna(row["ig_oas_chg"]) else None,
                }
            )
            rows_created += 1
    
    return {"rows_created": rows_created}


if __name__ == "__main__":
    result = build_model_dataset()
    print(f"Created {result['rows_created']} model dataset rows")
