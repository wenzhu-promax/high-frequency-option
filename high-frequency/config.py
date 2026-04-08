# -*- coding: utf-8 -*-
"""项目配置：路径、成分股、时间窗口等"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
# 与 run_sp100_batch.py 一致：父目录为 KaidiZhang，其下为 trade/quote parquet 与现货 CSV
_KAIDIZHANG_ROOT = _PROJECT_ROOT.parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "output"
# run_data_pipeline：--no-exchange-filter 且未指定 --output-dir 时默认写入此目录，避免与 output/ 混用
DEFAULT_OUTPUT_DIR_NO_EXCHANGE = _PROJECT_ROOT / "output_no_exchange"
DEFAULT_TRADE_PARQUET = _KAIDIZHANG_ROOT / "tradeSP100.parquet"
DEFAULT_QUOTE_PARQUET = _KAIDIZHANG_ROOT / "sp100quotes" / "quotesSP100.parquet"
DEFAULT_SPOT_CSV = _KAIDIZHANG_ROOT / "sp100_spot_prices_20251201.csv"
# 仓库内副本（父目录无现货文件时回退）
LOCAL_SPOT_CSV = _PROJECT_ROOT / "sp100_spot_prices_20251201.csv"

# SP100 成分股（用于 quote 过滤）
SP100 = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMAT", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C",
    "CAT", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX",
    "DE", "DHR", "DIS", "DUK", "EMR", "FDX", "GD", "GE", "GEV", "GILD",
    "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU", "ISRG",
    "JNJ", "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "LRCX", "MA", "MCD",
    "MDLZ", "MDT", "META", "MMM", "MO", "MRK", "MS", "MSFT", "MU", "NEE",
    "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP", "PFE", "PG", "PLTR", "PM",
    "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TMO", "TMUS", "TSLA",
    "TXN", "UBER", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM",
]

# 特征工程时间窗口 (秒)，格式 (left, right) 表示 lookback 区间
SPANS = [
    (0, 5), (5, 10), (10, 20), (20, 40), (40, 80),
    (80, 120), (120, 180), (180, 240), (240, 300),
]

# 扩展 Calendar Clock（更长窗口）
SPANS_CALENDAR_EXT = [
    (0, 5), (5, 10), (10, 20), (20, 40), (40, 80),
    (80, 120), (120, 180), (180, 300), (300, 600), (600, 1800),
]

# Transaction Clock：按交易笔数（期权适配，阈值放大）
SPANS_TRANSACTION = [
    (0, 10), (10, 20), (20, 50), (50, 100), (100, 200),
]

# Volume Clock：按累积成交量（期权适配，阈值放大，单位：份）
SPANS_VOLUME = [
    (0, 1000), (1000, 5000), (5000, 10000), (10000, 20000), (20000, 50000),
]

# 默认时钟类型：calendar | transaction | volume
DEFAULT_CLOCK_TYPE = "calendar"

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

# run_data_pipeline 正式实验：SP100 × 下列网格 × get_model_pipelines() 全模型；CLI 可覆盖子集/网格
FILTER_SAME_EXCHANGE = True
MONEYNESS_LIST = [0.08, 0.10, 0.12]
TOP_N_CONTRACTS = 30  # 放宽：从 50 降到 30，纳入更多合约
HORIZON_LIST = [5, 10, 30, 60, 90, 120]
MIN_VALID_Y = 10  # 放宽：从 20 降到 10
MIN_TRAIN_ROWS = 100  # 放宽：从 200 降到 100
MIN_SAMPLES_UNDERLYING = 50  # 放宽：从 100 降到 50
# 默认单进程；多进程请用 run_data_pipeline.py --workers N（注意内存随进程数放大）
DEFAULT_EXPERIMENT_WORKERS = 1
# 试跑示例（单标的 + 简化滚动，见 CLI）：python run_data_pipeline.py --tickers ORCL --horizons 5 --moneyness 0.10 --train-minutes 15 --refit-every-minutes 10 --min-train-rows 100
