"""Structure opacity flags for fragility scoring."""
# Manual mapping: 0=low, 1=moderate, 2=higher complexity
# This is a placeholder proxy and must be clearly noted as such
STRUCTURE_OPACITY = {
    "AAPL": 0,  # Low complexity
    "MSFT": 0,  # Low complexity
    "AMZN": 1,  # Moderate
    "GOOGL": 1,  # Moderate
    "META": 1,  # Moderate
    "ORCL": 2,  # Higher (structured financing)
    "NVDA": 1,  # Moderate
    "INTC": 1,  # Moderate
    "AVGO": 1,  # Moderate
    "AMD": 1,  # Moderate
    "MU": 1,  # Moderate
    "TSM": 1,  # Moderate
    "EQIX": 2,  # Higher (REIT leverage)
    "DLR": 2,  # Higher (REIT leverage)
    "IRM": 2,  # Higher (project-style exposure)
}

