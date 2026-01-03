"""Select representative bonds per issuer using objective liquidity rules."""
import pandas as pd
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import List, Dict


def select_bonds_initial(engine: Engine, target_maturity_start: int = 2028, target_maturity_end: int = 2036) -> pd.DataFrame:
    """
    Select bonds using initial rules with coverage-based fallback.
    
    Strategy:
    1. Primary: Prefer bonds with maturity_date >= 2025-12-31 (if maturity exists)
    2. Fallback: If no active maturity bonds, use coverage-based ranking
       (last_seen_trade_date DESC, first_seen_trade_date ASC)
    3. If maturity_date is null, include in selection (use coverage ranking)
    
    Returns DataFrame with selected bonds.
    """
    with engine.connect() as conn:
        # Get all bonds with issuer info
        query = text("""
            SELECT 
                b.cusip,
                b.issuer_id,
                i.ticker,
                i.issuer_name,
                b.maturity_date,
                b.first_seen_trade_date,
                b.last_seen_trade_date,
                b.coupon,
                b.security_type
            FROM dim_bond b
            JOIN dim_issuer i ON b.issuer_id = i.issuer_id
            ORDER BY i.ticker, b.last_seen_trade_date DESC NULLS LAST
        """)
        
        df = pd.read_sql(query, conn)
    
    if df.empty:
        return pd.DataFrame()
    
    # Compute coverage days (handle nulls)
    df['coverage_days'] = (
        pd.to_datetime(df['last_seen_trade_date'], errors='coerce') - 
        pd.to_datetime(df['first_seen_trade_date'], errors='coerce')
    ).dt.days
    
    # Extract maturity year (handle nulls)
    df['maturity_date_parsed'] = pd.to_datetime(df['maturity_date'], errors='coerce')
    df['maturity_year'] = df['maturity_date_parsed'].dt.year
    
    # Score bonds
    selected_bonds = []
    
    for ticker in df['ticker'].unique():
        issuer_bonds = df[df['ticker'] == ticker].copy()
        
        # Primary filter: bonds with maturity >= 2025-12-31 (if maturity exists)
        has_maturity = issuer_bonds['maturity_date_parsed'].notna()
        active_maturity = issuer_bonds[has_maturity & (issuer_bonds['maturity_date_parsed'] >= pd.Timestamp('2025-12-31'))]
        
        if len(active_maturity) > 0:
            # Use active maturity bonds
            candidate_bonds = active_maturity.copy()
            selection_method = 'MATURITY_ACTIVE'
        else:
            # Fallback: use coverage-based ranking (all bonds, including null maturity)
            candidate_bonds = issuer_bonds.copy()
            selection_method = 'COVERAGE_FALLBACK'
        
        # Score bonds
        candidate_bonds = candidate_bonds.copy()
        candidate_bonds['score'] = 0.0
        
        # Coverage score (normalized) - prefer longer coverage
        if candidate_bonds['coverage_days'].notna().any():
            max_coverage = candidate_bonds['coverage_days'].max()
            if max_coverage > 0:
                candidate_bonds['score'] += (candidate_bonds['coverage_days'] / max_coverage) * 100
        
        # Prefer later last_seen_trade_date (more recent activity)
        if candidate_bonds['last_seen_trade_date'].notna().any():
            max_last_seen = pd.to_datetime(candidate_bonds['last_seen_trade_date']).max()
            last_seen_days = (pd.to_datetime(candidate_bonds['last_seen_trade_date']) - max_last_seen).dt.days.abs()
            if last_seen_days.max() > 0:
                candidate_bonds['score'] += (1 - last_seen_days / last_seen_days.max()) * 50
        
        # Maturity preference: target range gets bonus (if maturity exists)
        if selection_method == 'MATURITY_ACTIVE' and candidate_bonds['maturity_year'].notna().any():
            target_range = (
                (candidate_bonds['maturity_year'] >= target_maturity_start) &
                (candidate_bonds['maturity_year'] <= target_maturity_end)
            )
            candidate_bonds.loc[target_range, 'score'] += 50
        
        # Sort by score, then by last_seen desc, then first_seen asc
        candidate_bonds = candidate_bonds.sort_values(
            ['score', 'last_seen_trade_date', 'first_seen_trade_date'],
            ascending=[False, False, True],
            na_position='last'
        )
        
        # Select top 2
        top_bonds = candidate_bonds.head(2)
        
        for idx, (_, bond) in enumerate(top_bonds.iterrows(), 1):
            maturity_str = str(bond['maturity_date'])[:10] if pd.notna(bond['maturity_date']) else None
            coverage_days = bond['coverage_days'] if pd.notna(bond['coverage_days']) else 0
            
            if selection_method == 'MATURITY_ACTIVE':
                reason = f"MATURITY_ACTIVE: maturity={maturity_str}, coverage={coverage_days:.0f} days"
            else:
                reason = f"COVERAGE_FALLBACK: coverage={coverage_days:.0f} days, last_seen={str(bond['last_seen_trade_date'])[:10] if pd.notna(bond['last_seen_trade_date']) else 'N/A'}"
            
            selected_bonds.append({
                'cusip': bond['cusip'],
                'issuer_id': bond['issuer_id'],
                'ticker': bond['ticker'],
                'issuer_name': bond['issuer_name'],
                'selected_rank': idx,
                'selected_reason': reason,
                'maturity_date': bond['maturity_date'],
                'first_seen_trade_date': bond['first_seen_trade_date'],
                'last_seen_trade_date': bond['last_seen_trade_date'],
                'coverage_days': coverage_days,
            })
    
    return pd.DataFrame(selected_bonds)


def select_bonds_from_activity(engine: Engine) -> pd.DataFrame:
    """
    Re-select bonds based on actual activity in fact_bond_daily.
    
    Strategy:
    - Total trades or total volume between 2020-2025
    - Select TOP1 or TOP2 per issuer
    """
    with engine.connect() as conn:
        # Get activity stats
        query = text("""
            SELECT 
                b.cusip,
                b.issuer_id,
                i.ticker,
                i.issuer_name,
                COUNT(f.trade_date) as trade_days,
                SUM(COALESCE(f.trades, 0)) as total_trades,
                SUM(COALESCE(f.volume, 0)) as total_volume
            FROM dim_bond b
            JOIN dim_issuer i ON b.issuer_id = i.issuer_id
            LEFT JOIN fact_bond_daily f ON b.cusip = f.cusip
            WHERE b.is_selected = TRUE
            GROUP BY b.cusip, b.issuer_id, i.ticker, i.issuer_name
            ORDER BY i.ticker, total_volume DESC NULLS LAST, total_trades DESC NULLS LAST
        """)
        
        df = pd.read_sql(query, conn)
    
    if df.empty:
        return pd.DataFrame()
    
    selected_bonds = []
    
    for ticker in df['ticker'].unique():
        issuer_bonds = df[df['ticker'] == ticker].copy()
        issuer_bonds = issuer_bonds.sort_values(['total_volume', 'total_trades'], ascending=False)
        
        # Select top 2
        top_bonds = issuer_bonds.head(2)
        
        for idx, (_, bond) in enumerate(top_bonds.iterrows(), 1):
            selected_bonds.append({
                'cusip': bond['cusip'],
                'issuer_id': bond['issuer_id'],
                'ticker': bond['ticker'],
                'issuer_name': bond['issuer_name'],
                'selected_rank': idx,
                'selected_reason': f"volume={bond['total_volume']:.0f}, trades={bond['total_trades']:.0f}",
                'total_volume': bond['total_volume'],
                'total_trades': bond['total_trades'],
            })
    
    return pd.DataFrame(selected_bonds)


def update_bond_selection(engine: Engine, selected_df: pd.DataFrame):
    """Update dim_bond with selection flags."""
    if selected_df.empty:
        return
    
    # First, clear all selections
    with engine.begin() as conn:
        conn.execute(text("UPDATE dim_bond SET is_selected = FALSE, selected_rank = NULL, selected_reason = NULL"))
    
    # Update selected bonds
    with engine.begin() as conn:
        for _, row in selected_df.iterrows():
            conn.execute(
                text("""
                    UPDATE dim_bond
                    SET is_selected = TRUE,
                        selected_rank = :rank,
                        selected_reason = :reason
                    WHERE cusip = :cusip
                """),
                {
                    "cusip": row['cusip'],
                    "rank": int(row['selected_rank']),
                    "reason": str(row['selected_reason']),
                }
            )


def save_selection_csv(selected_df: pd.DataFrame, output_dir: Path):
    """Save selected bonds to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "selected_bonds.csv"
    selected_df.to_csv(output_path, index=False)
    print(f"Selected bonds saved to: {output_path}")

