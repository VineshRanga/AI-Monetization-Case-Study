"""Issuer universe configuration."""
ISSUER_UNIVERSE = [
    # Tech (Hyperscalers)
    {"ticker": "AAPL", "issuer_name": "Apple Inc.", "bucket": "HYPERSCALER"},
    {"ticker": "MSFT", "issuer_name": "Microsoft Corporation", "bucket": "HYPERSCALER"},
    {"ticker": "AMZN", "issuer_name": "Amazon.com, Inc.", "bucket": "HYPERSCALER"},
    {"ticker": "GOOGL", "issuer_name": "Alphabet Inc.", "bucket": "HYPERSCALER"},
    {"ticker": "META", "issuer_name": "Meta Platforms, Inc.", "bucket": "HYPERSCALER"},
    {"ticker": "ORCL", "issuer_name": "Oracle Corporation", "bucket": "HYPERSCALER"},
    # Semiconductors
    {"ticker": "NVDA", "issuer_name": "NVIDIA Corporation", "bucket": "SEMIS"},
    {"ticker": "INTC", "issuer_name": "Intel Corporation", "bucket": "SEMIS"},
    {"ticker": "AVGO", "issuer_name": "Broadcom Inc.", "bucket": "SEMIS"},
    {"ticker": "AMD", "issuer_name": "Advanced Micro Devices, Inc.", "bucket": "SEMIS"},
    {"ticker": "MU", "issuer_name": "Micron Technology, Inc.", "bucket": "SEMIS"},
    {"ticker": "TSM", "issuer_name": "Taiwan Semiconductor Manufacturing Company Ltd.", "bucket": "SEMIS"},
    # Data Centers
    {"ticker": "EQIX", "issuer_name": "Equinix, Inc.", "bucket": "DATACENTER"},
    {"ticker": "DLR", "issuer_name": "Digital Realty Trust, Inc.", "bucket": "DATACENTER"},
    {"ticker": "IRM", "issuer_name": "Iron Mountain Incorporated", "bucket": "DATACENTER"},
]

