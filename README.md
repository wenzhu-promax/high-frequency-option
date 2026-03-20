# OPRA Options Experiment

期权 trade/quote 数据预处理，特征构造，滚动 多模型预测 5 秒收益。

## 依赖

```bash
pip install -r requirements.txt
```

Python 3.10+。

## 跑流程

数据放 `dataset/`，默认找 `trades_2025-12-01.csv.gz` 和 `quotes_2025-12-01.csv.gz`：

```bash
python run_data_pipeline.py
```

可选参数：`--data-dir`、`--output-dir`、`--no-models`（跳过模型）。

输出在 `output/`：`tradeSP100.parquet`、`quotesSP100.parquet`、`AAPL.parquet`、`AAPL_features.parquet`。

## 结构

- **`config.py`**：项目配置  
  - 路径：`DEFAULT_DATA_DIR`、`DEFAULT_OUTPUT_DIR`  
  - 成分股：`SP100`（quote 过滤）  
  - 特征窗口：`SPANS`（9 个 lookback 区间，秒）  
  - 期权筛选：`DTE_MIN/MAX`、`MONEYNESS_THRESHOLD`  
  - 训练：`TRAIN_MINUTES`、`REFIT_EVERY_MINUTES`、`DEFAULT_MODEL`  

- **`data_processing.py`**：数据加载、合并、清洗、标签  
  - `load_trade`：加载 trade CSV，解析 ticker  
  - `filter_quotes_sp100`：DuckDB 过滤 quote 为 SP100 成分股  
  - `merge_trades_quotes`：merge_asof 对齐 trade 与 quote  
  - `filter_rth`：保留 RTH 时段（09:30–16:00 ET）  
  - `clean_merged_option_data`：异常过滤（7 类规则）  
  - `deduplicate_by_ticker_trade_ts`：去重聚合  
  - `add_y_5s_return`：构造 5 秒收益率标签  

- **`feature_engineering.py`**：特征工程  
  - `add_trade_direction_proxy`：Lee-Ready 买卖方向  
  - `build_calendar_predictors`：13 类特征（breadth、immediacy、volume、lambda、lob_imbalance、txn_imbalance、past_return、turnover、autocov、quoted_spread、effective_spread 等），多时间窗口  
  - `get_predictor_cols`：提取 predictor 列名  
  - `check_span_occupancy`：统计 span 窗口非空比例  

- **`model_training.py`**：模型训练与评估  
  - `get_model_pipelines`：Lasso、Ridge、ElasticNet、RandomForest、XGBoost  
  - `rolling_train_predict`：滚动训练预测  
  - `evaluate_predictions`：RMSE、R²、方向准确率  

- **`model_plots.py`**：可视化  
  - `plot_pred_vs_actual`、`plot_residuals`、`plot_model_comparison`  
  - `plot_rolling_r2`、`plot_hourly_direction_accuracy`  
  - `plot_coef_heatmap`、`plot_multi_model_rolling_r2`  

- **`run_data_pipeline.py`**：主流程  
  - `_stage_load_trade_quote`：加载 trade/quote  
  - `_stage_merge_and_clean`：合并、清洗、去重  
  - `_stage_select_options`：DTE/moneyness 筛选  
  - `_stage_build_features`：标签与特征  
  - `_stage_run_models`：滚动训练与评估  
  - `main`：串联上述阶段
