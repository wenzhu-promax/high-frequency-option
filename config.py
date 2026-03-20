# -*- coding: utf-8 -*-
"""项目配置：路径、成分股、时间窗口等"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "output"

# SP100 成分股（用于 quote 过滤）
SP100 = [
    "AAPL", "ABT", "ACN", "ADBE", "ADP", "AMGN", "AMT", "AMZN", "AXP", "BA",
    "BAC", "BDX", "BMY", "BSX", "C", "CAT", "CB", "CI", "CMCSA", "CME", "COP",
    "COST", "CRM", "CSCO", "CVS", "CVX", "D", "DHR", "DUK", "FIS", "FISV",
    "GE", "GILD", "GOOG", "GS", "HD", "HON", "IBM", "INTC", "INTU", "ISRG",
    "JNJ", "JPM", "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDT", "MMM", "MO",
    "MRK", "MS", "MSFT", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG",
    "PNC", "QCOM", "SBUX", "SO", "SYK", "T", "TGT", "TJX", "TXN", "UNH", "UNP",
    "UPS", "USB", "VZ", "WFC", "WMT",
]

# 特征工程时间窗口 (秒)，格式 (left, right) 表示 lookback 区间
SPANS = [
    (0, 5), (5, 10), (10, 20), (20, 40), (40, 80),
    (80, 120), (120, 180), (180, 240), (240, 300),
]

# 期权筛选
DTE_MIN, DTE_MAX = 0, 14
MONEYNESS_THRESHOLD = 0.05

# 滚动训练
TRAIN_MINUTES = 30
REFIT_EVERY_MINUTES = 5

# 特征分组列
DEFAULT_GROUP_COLS = ["date", "underlying", "expiry", "cp_flag", "strike"]

# 默认模型
DEFAULT_MODEL = "Lasso"
