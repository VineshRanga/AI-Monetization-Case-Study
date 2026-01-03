"""Output CSV manifests for bond universe."""
import pandas as pd
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.engine import Engine


def write_manifests(engine: Engine, output_dir: Path):
    """
    Write CSV manifests to output directory.
    
    Args:
        engine: SQLAlchemy engine
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Bond manifest
    with engine.connect() as conn:
        bonds_df = pd.read_sql(
            text("""
                SELECT 
                    i.ticker,
                    b.issuer_name,
                    b.cusip,
                    b.coupon,
                    b.maturity_date,
                    b.issue_date,
                    b.security_type,
                    b.first_seen_trade_date,
                    b.last_seen_trade_date,
                    b.is_active
                FROM dim_bond b
                JOIN dim_issuer i ON b.issuer_id = i.issuer_id
                ORDER BY i.ticker, b.last_seen_trade_date DESC
            """),
            conn
        )
    
    manifest_path = output_dir / "bond_manifest.csv"
    bonds_df.to_csv(manifest_path, index=False)
    print(f"Bond manifest written: {manifest_path} ({len(bonds_df)} bonds)")
    
    # Bond counts by issuer
    with engine.connect() as conn:
        counts_df = pd.read_sql(
            text("""
                SELECT 
                    i.ticker,
                    i.issuer_name,
                    COUNT(b.cusip) as num_cusips
                FROM dim_issuer i
                LEFT JOIN dim_bond b ON i.issuer_id = b.issuer_id
                GROUP BY i.ticker, i.issuer_name
                ORDER BY num_cusips DESC, i.ticker
            """),
            conn
        )
    
    counts_path = output_dir / "bond_counts_by_issuer.csv"
    counts_df.to_csv(counts_path, index=False)
    print(f"Bond counts written: {counts_path}")

