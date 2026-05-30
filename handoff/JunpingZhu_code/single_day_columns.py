"""Shared column definitions for the trade-driven single-day pipeline.

Paper variable families from Section 2.3:
- Breadth
- Immediacy
- VolumeAll / VolumeAvg / VolumeMax
- Lambda
- LobImbalance
- TxnImbalance
- PastReturn
- Turnover
- AutoCov
- QuotedSpread
- EffectiveSpread
"""

HORIZONS = [5, 10, 30, 60]

TRADE_SPANS = [
    (0, 5),
    (5, 10),
    (10, 20),
    (20, 40),
    (40, 80),
    (80, 120),
    (120, 180),
    (180, 240),
    (240, 300),
]

BASE_FEATURE_COLS = [
    "mid",
    "rel_spread",
    "bid_sz",
    "ask_sz",
    "quote_sz",
    "lob_imb",
    "strike",
    "dte",
    "cp_flag",
]

SPAN_FEATURE_NAMES = [
    "breadth",
    "immediacy",
    "volume_all",
    "volume_avg",
    "volume_max",
    "lambda_",
    "lob_imbalance",
    "txn_imbalance",
    "past_return",
    "turnover",
    "autocov",
    "quoted_spread",
    "effective_spread",
    "no_quote",
    "no_trade",
]


def span_label(left: int, right: int) -> str:
    return f"{left}_{right}s"


FEATURE_COLS = BASE_FEATURE_COLS + [
    f"{name}__{span_label(left, right)}"
    for left, right in TRADE_SPANS
    for name in SPAN_FEATURE_NAMES
]

LEGACY_BASE_MAP = {
    "current_mid": "mid",
    "current_relative_spread": "rel_spread",
    "current_bid_size": "bid_sz",
    "current_ask_size": "ask_sz",
    "current_quote_size_sum": "quote_sz",
    "current_lob_imbalance": "lob_imb",
    "DTE": "dte",
    "option_type_flag": "cp_flag",
}

LEGACY_SPAN_MAP = {
    "breadth": "breadth",
    "immediacy": "immediacy",
    "volume_all": "volume_all",
    "volume_avg": "volume_avg",
    "volume_max": "volume_max",
    "mid_lambda": "lambda_",
    "lob_imbalance": "lob_imbalance",
    "txn_imbalance": "txn_imbalance",
    "past_return": "past_return",
    "turnover": "turnover",
    "mid_autocov": "autocov",
    "quoted_spread_mean": "quoted_spread",
    "effective_spread": "effective_spread",
    "no_quote_flag": "no_quote",
    "no_trade_flag": "no_trade",
}
