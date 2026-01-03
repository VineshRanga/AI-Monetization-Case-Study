#!/usr/bin/env python3
"""Select representative bonds per issuer."""
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.db import get_engine
from src.transform.select_bonds import (
    select_bonds_initial,
    save_selection_csv,
    update_bond_selection,
)

if __name__ == "__main__":
    load_dotenv()
    
    print("="*60)
    print("Selecting Representative Bonds")
    print("="*60)
    
    engine = get_engine()
    
    # Select bonds using initial rules
    print("\nSelecting bonds using coverage and maturity rules...")
    selected_df = select_bonds_initial(engine)
    
    if selected_df.empty:
        print("No bonds found to select")
        sys.exit(1)
    
    # Update database
    update_bond_selection(engine, selected_df)
    print(f"Selected {len(selected_df)} bonds across {selected_df['ticker'].nunique()} issuers")
    
    # Save CSV
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    save_selection_csv(selected_df, data_dir)
    
    # Print summary
    print("\nSelected bonds by issuer:")
    print("-" * 60)
    for ticker in sorted(selected_df['ticker'].unique()):
        issuer_bonds = selected_df[selected_df['ticker'] == ticker]
        print(f"\n{ticker}:")
        for _, bond in issuer_bonds.iterrows():
            print(f"  Rank {bond['selected_rank']}: {bond['cusip']} - {bond['selected_reason']}")

