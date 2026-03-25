# OPRA Options Experiment

期权 trade/quote：**CSV→parquet（按需）→ 全量实验**（多标的 × 交易所过滤 × 多档 moneyness × TopN × 多 horizon × 多模型，单组训练超参、无网格）。

## 依赖

```bash
pip install -r requirements.txt
```

Python 3.10+。

## 现货价格 CSV

与本仓库主入口同级目录下的 **`sp100_spot_prices_20251201.csv`**（列 `Ticker`、`Close_Price_2025-12-01` 等）为 **Massive 官方 API** 拉取的 SP100 成分股在指定日期的收盘价快照，用于计算 moneyness；若未传 `--spot-csv` 且该文件存在，主流程会自动加载。缺失标的时仍会按近月成交量加权 strike 回退估算。

## 跑流程

数据放 `dataset/`，默认 `trades_2025-12-01.csv.gz` 与 `quotes_2025-12-01.csv.gz`。首次运行会生成 `output/tradeSP100.parquet`、`output/quotesSP100.parquet`，再跑全量实验：

```bash
python run_data_pipeline.py
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--data-dir` | 默认 `dataset/` |
| `--output-dir` | 默认 `output/` |
| `--spot-csv` | 可选；列 `Ticker` + 价格列（默认列名见 `--spot-price-col`）。省略时使用仓库内 `sp100_spot_prices_20251201.csv`（Massive API） |
| `--tickers` | 逗号分隔，默认 SP100 全部 |
| `--trade-parquet` / `--quote-parquet` | 指定则跳过从 CSV 准备 |
| `--no-prepare` | 不调用 CSV→parquet，使用已有 parquet（默认路径见上） |
| `--no-models` | 只统计特征行，不训练 |

汇总：`output/summary_experiment.csv`。过滤与 horizon、TopN 等在 **`config.py`**（`FILTER_SAME_EXCHANGE`、`MONEYNESS_LIST`、`TOP_N_CONTRACTS`、`HORIZON_LIST` 等）。

## 结构

- **`config.py`**：路径、`SP100`、`SPANS`、DTE、**全量实验开关**（`FILTER_SAME_EXCHANGE`、`MONEYNESS_LIST`、`TOP_N_CONTRACTS`、`HORIZON_LIST`、`MIN_VALID_Y`、`MIN_TRAIN_ROWS`）、`TRAIN_MINUTES` 等  
- **`data_processing.py`**：`load_trade`、`filter_quotes_sp100`、`merge_trades_quotes`、`fix_dt_et_from_trade_ts`、`filter_rth`、`clean_merged_option_data`、`deduplicate_by_ticker_trade_ts`、`add_y_5s_return`  
- **`feature_engineering.py`**：`add_trade_direction_proxy`、`build_calendar_predictors`、`get_predictor_cols`  
- **`model_training.py`**：`get_model_pipelines`、`rolling_train_predict`、`evaluate_predictions`  
- **`run_data_pipeline.py`**：唯一入口 — `run_experiment`（全量循环 + 汇总 CSV）  

- **`model_plots.py`**：可视化（独立使用）
