"""Scenario engine: compute stress test scenarios for AI credit crisis."""
import json
import pandas as pd
from datetime import date, datetime
from sqlalchemy import text, bindparam
from sqlalchemy.engine import Engine
from typing import Dict, List, Optional
from pathlib import Path

from src.db.db import get_engine


# Bucket multipliers for AI shock scenarios
BUCKET_MULTIPLIERS = {
    "SEMIS": 1.3,
    "DATACENTER": 1.15,
    "HYPERSCALER": 1.0,
}

# Exposure groups (heuristic mapping for spillover)
EXPOSURE_GROUPS = {
    "Banks": {"to_datacenters": 0.6, "to_semis": 0.4},
    "AssetManagers": {"to_hyperscalers": 0.4, "to_semis": 0.3, "to_datacenters": 0.3},
    "TechSupplyChain": {"to_semis": 0.7, "to_datacenters": 0.3},
}


def get_baseline_predictions(engine: Engine, year: int = 2025) -> pd.DataFrame:
    """
    Load baseline predictions for specified year.
    
    Returns:
        DataFrame with columns: issuer_id, week_start, y_pred, regime_label, bucket
    """
    with engine.connect() as conn:
        # Get latest model_id for RISK_OFF and RISK_ON
        models_df = pd.read_sql(
            text("""
                SELECT model_id, regime
                FROM model_registry
                WHERE regime IN ('RISK_OFF', 'RISK_ON')
                AND model_name LIKE 'xgb_credit_%'
                ORDER BY created_at DESC
            """),
            conn
        )
        
        if models_df.empty:
            raise ValueError("No models found in model_registry")
        
        # Get most recent model per regime
        model_map = {}
        for regime in ["RISK_OFF", "RISK_ON"]:
            regime_models = models_df[models_df["regime"] == regime]
            if not regime_models.empty:
                # Convert to native Python int (not numpy.int64)
                model_map[regime] = int(regime_models.iloc[0]["model_id"])
        
        if not model_map:
            raise ValueError("No RISK_OFF or RISK_ON models found")
        
        # Convert model_ids to native Python ints
        model_ids = [int(x) for x in model_map.values()]
        print(f"    Using model_ids: {model_ids} (types: {[type(x).__name__ for x in model_ids]})")
        
        # Load predictions for 2025, matching model_id to regime_label
        # We need to join predictions with dataset to get regime_label, then filter by matching model_id
        # Use expanding bindparam for IN clause
        sql_query = text("""
            SELECT 
                p.issuer_id,
                p.week_start,
                p.y_pred as baseline_pred_dcredit_proxy,
                COALESCE(d.regime_label, 'RISK_ON') as regime_label,
                COALESCE(d.bucket, 'HYPERSCALER') as bucket,
                p.model_id
            FROM model_predictions_weekly p
            JOIN model_dataset_weekly d ON p.issuer_id = d.issuer_id AND p.week_start = d.week_start
            WHERE p.week_start >= :year_start
            AND p.week_start <= :year_end
            AND p.split = 'test'
            AND p.model_id IN :model_ids
            AND p.y_pred IS NOT NULL
            ORDER BY p.issuer_id, p.week_start
        """).bindparams(bindparam("model_ids", expanding=True))
        
        pred_df = pd.read_sql(
            sql_query,
            conn,
            params={
                "year_start": f"{year}-01-01",
                "year_end": f"{year}-12-31",
                "model_ids": model_ids,  # List of native Python ints
            }
        )
        
        if pred_df.empty:
            raise ValueError(f"No baseline predictions found for {year}")
        
        # Filter to keep only predictions where model_id matches regime_label
        def match_regime(row):
            regime = row["regime_label"]
            model_id = int(row["model_id"])  # Convert to native int for comparison
            return model_id == model_map.get(regime)
        
        pred_df = pred_df[pred_df.apply(match_regime, axis=1)].copy()
        pred_df = pred_df.drop(columns=["model_id"])
        
        if pred_df.empty:
            raise ValueError(f"No matching baseline predictions found for {year} (regime-model mismatch)")
        
        return pred_df


def get_fragility_scores(engine: Engine) -> pd.DataFrame:
    """
    Load latest fragility scores and pillar scores.
    
    Returns:
        DataFrame with columns: issuer_id, total_score_0_100, pillar_name, score_0_100
    """
    with engine.connect() as conn:
        # Get latest asof_date
        result = conn.execute(text("SELECT MAX(asof_date) FROM fragility_scores"))
        row = result.fetchone()
        if not row or row[0] is None:
            raise ValueError("No fragility scores found. Run run_fragility.py first.")
        
        latest_date = row[0]
        
        # Load total scores
        total_df = pd.read_sql(
            text("""
                SELECT issuer_id, total_score_0_100
                FROM fragility_scores
                WHERE asof_date = :latest_date
            """),
            conn,
            params={"latest_date": latest_date}
        )
        
        # Load pillar scores
        pillar_df = pd.read_sql(
            text("""
                SELECT issuer_id, pillar_name, score_0_100
                FROM fragility_pillar_scores
                WHERE asof_date = :latest_date
            """),
            conn,
            params={"latest_date": latest_date}
        )
        
        return total_df, pillar_df, latest_date


def compute_scenario_uplift(
    baseline_df: pd.DataFrame,
    fragility_total: float,
    fragility_p1: Optional[float],
    fragility_p3: Optional[float],
    bucket: str,
    regime_label: str,
    scenario_type: str,
    base_shock: float = 0.05,
) -> float:
    """
    Compute scenario uplift (additional ΔPD proxy) for an issuer-week.
    
    Args:
        baseline_df: Row from baseline predictions
        fragility_total: Total fragility score (0-100)
        fragility_p1: Pillar 1 (refinancing_wall) score, if available
        fragility_p3: Pillar 3 (leverage_coverage) score, if available
        bucket: Issuer bucket (SEMIS, DATACENTER, HYPERSCALER)
        regime_label: RISK_OFF or RISK_ON
        scenario_type: "base", "ai_shock", or "ai_shock_funding_freeze"
        base_shock: Base shock magnitude (default 0.05 = 5 bps ΔPD proxy)
    
    Returns:
        Uplift value (additional ΔPD proxy)
    """
    if scenario_type == "base":
        return 0.0
    
    # Base uplift formula: base_shock * (0.5 + fragility_total/100) * bucket_multiplier
    bucket_mult = BUCKET_MULTIPLIERS.get(bucket, 1.0)
    fragility_factor = 0.5 + (fragility_total / 100.0)
    uplift = base_shock * fragility_factor * bucket_mult
    
    if scenario_type == "ai_shock_funding_freeze":
        # Add refinancing + leverage emphasis
        if fragility_p1 is not None and fragility_p3 is not None:
            funding_factor = 1.0 + 0.5 * ((fragility_p1 + fragility_p3) / 200.0)
            uplift *= funding_factor
        
        # Regime amplification
        if regime_label == "RISK_OFF":
            uplift *= 1.5
    
    return uplift


def run_scenario(
    engine: Engine,
    scenario_name: str,
    scenario_desc: str,
    scenario_type: str,
    base_shock: float = 0.05,
    year: int = 2025,
) -> Dict:
    """
    Run a single scenario and store results.
    
    Returns:
        Dict with scenario_id and row counts
    """
    print(f"\n  Running scenario: {scenario_name}")
    
    # Load baseline predictions
    baseline_df = get_baseline_predictions(engine, year=year)
    print(f"    Loaded {len(baseline_df)} baseline predictions")
    
    # Load fragility scores
    total_df, pillar_df, fragility_date = get_fragility_scores(engine)
    print(f"    Loaded fragility scores as of {fragility_date}")
    
    # Merge fragility data
    baseline_df = baseline_df.merge(
        total_df[["issuer_id", "total_score_0_100"]],
        on="issuer_id",
        how="left"
    )
    baseline_df["total_score_0_100"] = baseline_df["total_score_0_100"].fillna(50.0)
    
    # Merge pillar scores (P1 and P3)
    pillar_pivot = pillar_df.pivot(index="issuer_id", columns="pillar_name", values="score_0_100")
    if "refinancing_wall" in pillar_pivot.columns:
        baseline_df = baseline_df.merge(
            pillar_pivot[["refinancing_wall"]].reset_index(),
            on="issuer_id",
            how="left"
        )
        baseline_df["fragility_p1"] = baseline_df["refinancing_wall"]
    else:
        baseline_df["fragility_p1"] = None
    
    if "leverage_coverage" in pillar_pivot.columns:
        baseline_df = baseline_df.merge(
            pillar_pivot[["leverage_coverage"]].reset_index(),
            on="issuer_id",
            how="left"
        )
        baseline_df["fragility_p3"] = baseline_df["leverage_coverage"]
    else:
        baseline_df["fragility_p3"] = None
    
    # Compute uplift for each row
    baseline_df["uplift"] = baseline_df.apply(
        lambda row: compute_scenario_uplift(
            row,
            row["total_score_0_100"],
            row.get("fragility_p1"),
            row.get("fragility_p3"),
            row["bucket"],
            row["regime_label"],
            scenario_type,
            base_shock=base_shock,
        ),
        axis=1
    )
    
    # Compute scenario prediction
    baseline_df["scenario_pred_dcredit_proxy"] = (
        baseline_df["baseline_pred_dcredit_proxy"] + baseline_df["uplift"]
    )
    
    # Build drivers_json
    baseline_df["drivers_json"] = baseline_df.apply(
        lambda row: json.dumps({
            "fragility_total": float(row["total_score_0_100"]),
            "fragility_p1": float(row.get("fragility_p1", 0.0)) if pd.notna(row.get("fragility_p1")) else None,
            "fragility_p3": float(row.get("fragility_p3", 0.0)) if pd.notna(row.get("fragility_p3")) else None,
            "bucket": str(row["bucket"]),
            "bucket_multiplier": BUCKET_MULTIPLIERS.get(row["bucket"], 1.0),
            "regime_label": str(row["regime_label"]),
            "regime_multiplier": 1.5 if row["regime_label"] == "RISK_OFF" and scenario_type == "ai_shock_funding_freeze" else 1.0,
        }),
        axis=1
    )
    
    # Store scenario definition
    parameters = {
        "scenario_type": scenario_type,
        "base_shock": base_shock,
        "bucket_multipliers": BUCKET_MULTIPLIERS,
        "year": year,
        "fragility_asof_date": fragility_date.isoformat() if isinstance(fragility_date, date) else str(fragility_date),
    }
    
    with engine.begin() as conn:
        # Upsert scenario definition
        result = conn.execute(
            text("""
                INSERT INTO scenario_definition (
                    scenario_name, scenario_desc, asof_date, parameters_json
                )
                VALUES (
                    :scenario_name, :scenario_desc, :asof_date, :parameters_json
                )
                ON CONFLICT (scenario_name) DO UPDATE SET
                    scenario_desc = EXCLUDED.scenario_desc,
                    asof_date = EXCLUDED.asof_date,
                    parameters_json = EXCLUDED.parameters_json,
                    created_at = NOW()
                RETURNING scenario_id
            """),
            {
                "scenario_name": scenario_name,
                "scenario_desc": scenario_desc,
                "asof_date": date.today(),
                "parameters_json": json.dumps(parameters),
            }
        )
        scenario_id = result.fetchone()[0]
    
    # Store shock paths (weekly)
    with engine.begin() as conn:
        # Delete existing paths for this scenario
        conn.execute(
            text("DELETE FROM scenario_shock_path_weekly WHERE scenario_id = :scenario_id"),
            {"scenario_id": scenario_id}
        )
        
        # Group by week to compute average shock
        weekly_shocks = baseline_df.groupby("week_start").agg({
            "uplift": "mean",
            "regime_label": lambda x: "RISK_OFF" if (x == "RISK_OFF").any() else "RISK_ON",
        }).reset_index()
        
        weekly_shocks["shock_base"] = weekly_shocks["uplift"]
        weekly_shocks["shock_regime_mult"] = weekly_shocks["regime_label"].apply(
            lambda x: 1.5 if x == "RISK_OFF" and scenario_type == "ai_shock_funding_freeze" else 1.0
        )
        
        for _, row in weekly_shocks.iterrows():
            conn.execute(
                text("""
                    INSERT INTO scenario_shock_path_weekly (
                        scenario_id, week_start, shock_regime_mult, shock_base
                    )
                    VALUES (
                        :scenario_id, :week_start, :shock_regime_mult, :shock_base
                    )
                    ON CONFLICT (scenario_id, week_start) DO UPDATE SET
                        shock_regime_mult = EXCLUDED.shock_regime_mult,
                        shock_base = EXCLUDED.shock_base
                """),
                {
                    "scenario_id": scenario_id,
                    "week_start": row["week_start"],
                    "shock_regime_mult": float(row["shock_regime_mult"]),
                    "shock_base": float(row["shock_base"]),
                }
            )
    
    # Store issuer-week results
    rows_created = 0
    with engine.begin() as conn:
        # Delete existing results for this scenario
        conn.execute(
            text("DELETE FROM scenario_results_issuer_weekly WHERE scenario_id = :scenario_id"),
            {"scenario_id": scenario_id}
        )
        
        for _, row in baseline_df.iterrows():
            conn.execute(
                text("""
                    INSERT INTO scenario_results_issuer_weekly (
                        scenario_id, issuer_id, week_start,
                        baseline_pred_dcredit_proxy, scenario_pred_dcredit_proxy,
                        uplift, drivers_json
                    )
                    VALUES (
                        :scenario_id, :issuer_id, :week_start,
                        :baseline_pred_dcredit_proxy, :scenario_pred_dcredit_proxy,
                        :uplift, :drivers_json
                    )
                    ON CONFLICT (scenario_id, issuer_id, week_start) DO UPDATE SET
                        baseline_pred_dcredit_proxy = EXCLUDED.baseline_pred_dcredit_proxy,
                        scenario_pred_dcredit_proxy = EXCLUDED.scenario_pred_dcredit_proxy,
                        uplift = EXCLUDED.uplift,
                        drivers_json = EXCLUDED.drivers_json,
                        created_at = NOW()
                """),
                {
                    "scenario_id": scenario_id,
                    "issuer_id": int(row["issuer_id"]),
                    "week_start": row["week_start"],
                    "baseline_pred_dcredit_proxy": float(row["baseline_pred_dcredit_proxy"]),
                    "scenario_pred_dcredit_proxy": float(row["scenario_pred_dcredit_proxy"]),
                    "uplift": float(row["uplift"]),
                    "drivers_json": str(row["drivers_json"]),
                }
            )
            rows_created += 1
    
    print(f"    Stored {rows_created} issuer-week results")
    
    return {
        "scenario_id": scenario_id,
        "rows_created": rows_created,
    }


def compute_spillover(engine: Engine, scenario_id: int) -> Dict:
    """
    Compute spillover indices for exposure groups.
    
    Returns:
        Dict with group_name -> spillover_index
    """
    with engine.connect() as conn:
        # Get bucket-level average uplift
        bucket_uplift = pd.read_sql(
            text("""
                SELECT 
                    d.bucket,
                    AVG(sr.uplift) as avg_uplift
                FROM scenario_results_issuer_weekly sr
                JOIN model_dataset_weekly d ON sr.issuer_id = d.issuer_id AND sr.week_start = d.week_start
                WHERE sr.scenario_id = :scenario_id
                GROUP BY d.bucket
            """),
            conn,
            params={"scenario_id": scenario_id}
        )
        
        if bucket_uplift.empty:
            return {}
        
        bucket_map = bucket_uplift.set_index("bucket")["avg_uplift"].to_dict()
        
        # Compute spillover for each group
        spillover_results = {}
        
        with engine.begin() as conn:
            # Delete existing spillover for this scenario
            conn.execute(
                text("DELETE FROM scenario_spillover_groups WHERE scenario_id = :scenario_id"),
                {"scenario_id": scenario_id}
            )
            
            for group_name, exposures in EXPOSURE_GROUPS.items():
                spillover_index = 0.0
                drivers = {}
                
                for bucket_key, weight in exposures.items():
                    # Map bucket_key to actual bucket name
                    bucket_name = bucket_key.replace("to_", "").upper()
                    if bucket_name == "HYPERS" or bucket_name == "HYPERSCALERS":
                        bucket_name = "HYPERSCALER"
                    elif bucket_name == "DATACENTERS":
                        bucket_name = "DATACENTER"
                    elif bucket_name == "SEMIS":
                        bucket_name = "SEMIS"
                    
                    if bucket_name in bucket_map:
                        contribution = bucket_map[bucket_name] * weight
                        spillover_index += contribution
                        drivers[bucket_key] = {
                            "weight": weight,
                            "bucket_uplift": float(bucket_map[bucket_name]),
                            "contribution": float(contribution),
                        }
                
                drivers["total_spillover_index"] = float(spillover_index)
                
                # Store
                conn.execute(
                    text("""
                        INSERT INTO scenario_spillover_groups (
                            scenario_id, group_name, spillover_index, drivers_json
                        )
                        VALUES (
                            :scenario_id, :group_name, :spillover_index, :drivers_json
                        )
                        ON CONFLICT (scenario_id, group_name) DO UPDATE SET
                            spillover_index = EXCLUDED.spillover_index,
                            drivers_json = EXCLUDED.drivers_json,
                            created_at = NOW()
                    """),
                    {
                        "scenario_id": scenario_id,
                        "group_name": group_name,
                        "spillover_index": float(spillover_index),
                        "drivers_json": json.dumps(drivers),
                    }
                )
                
                spillover_results[group_name] = spillover_index
        
        return spillover_results


def run_all_scenarios(engine: Engine, year: int = 2025) -> Dict:
    """
    Run all 3 scenarios and compute spillover.
    
    Returns:
        Dict with scenario results
    """
    scenarios = [
        {
            "name": "Base Risk-Off",
            "desc": "Baseline model predictions (no additional shock)",
            "type": "base",
            "base_shock": 0.0,
        },
        {
            "name": "AI Monetization Shock",
            "desc": "AI capex reversal + utilization disappointment. Bucket-sensitive (semis/datacenters most affected).",
            "type": "ai_shock",
            "base_shock": 0.05,
        },
        {
            "name": "AI Shock + Funding Freeze",
            "desc": "AI shock + 2008-style funding freeze. Amplified by refinancing wall + leverage + risk-off regime.",
            "type": "ai_shock_funding_freeze",
            "base_shock": 0.05,
        },
    ]
    
    results = {}
    
    for scenario in scenarios:
        scenario_result = run_scenario(
            engine,
            scenario["name"],
            scenario["desc"],
            scenario["type"],
            base_shock=scenario["base_shock"],
            year=year,
        )
        
        # Compute spillover
        spillover = compute_spillover(engine, scenario_result["scenario_id"])
        scenario_result["spillover"] = spillover
        
        results[scenario["name"]] = scenario_result
    
    return results

