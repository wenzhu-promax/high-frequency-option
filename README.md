# Push Mainline

该目录为准备推送的**高频期权实验**主线代码快照：含 `README.md` 与 `high-frequency/` 下 9 个顶层 `.py` + `requirements.txt`，以及子目录 `high-frequency/carryforward/`（carry-forward 入口、共用模块与 `backup/` 旧版备份）；不包含实验结果目录、日志或其它子项目。

## 文件清单（与仓库一致）

| 路径 | 说明 |
|------|------|
| `high-frequency/config.py` | 路径、SP100、窗口、实验网格等常量 |
| `high-frequency/data_processing.py` | Trade/quote 加载、合并、清洗、标签 `y_5s_ret` |
| `high-frequency/feature_engineering.py` | 方向代理、多时钟预测特征 |
| `high-frequency/model_training.py` | 多模型 Pipeline、滚动训练、评估 |
| `high-frequency/model_explain.py` | 末窗系数 / importance / SHAP |
| `high-frequency/model_plots.py` | 预测诊断图等；**summary 分面箱线图请以 `analyze_summary_eqweight.py` 为准**（`plot_summary_*_facets` 已弃用） |
| `high-frequency/run_data_pipeline.py` | **正式实验主入口**：数据 → 特征 → 滚动训练 → `output/` |
| `high-frequency/carryforward/run_carryforward_experiment.py` | **唯一 carry-forward 入口**：调用 `carryforward_experiment_common`；默认输出 `carryforward/output_carryforward/` |
| `high-frequency/carryforward/carryforward_experiment_common.py` | `load_d0_for_cf`、`iter_mo_horizon_features`、`rolling_train_predict_carryforward`、`write_by_ticker_csv` |
| `high-frequency/carryforward/backup/run_carryforward_experiment_monolithic_legacy.py` | 旧版单体脚本备份（内联 `rolling_train_predict_carryforward`），不参与日常运行 |
| `high-frequency/analyze_summary_eqweight.py` | 读 `summary_experiment.csv`，`--mode eqweight/rigorous/all` |
| `high-frequency/requirements.txt` | Python 依赖 |

## 入口与说明

- 正式实验：`python high-frequency/run_data_pipeline.py`（工作目录建议为 `high-frequency/` 或按脚本内路径配置数据）。
- Carry-forward：工作目录为 `high-frequency/`，执行 `python carryforward/run_carryforward_experiment.py`（共用 `carryforward/carryforward_experiment_common.py`）；未指定 `--output-dir` 时默认写入 `carryforward/output_carryforward/`。
- 汇总分析：`python high-frequency/analyze_summary_eqweight.py`。
- `--explain` / SHAP：`model_explain` 依赖 `shap`；环境无 `shap` 时 `shap_mean` 等为空。

## 主流程调用链（便于对照索引）

1. **正式实验**：`run_data_pipeline.main` → `run_experiment` → `_process_one_underlying` → `model_training` 滚动训练 / 评估；数据侧为 `data_processing`、`feature_engineering`。
2. **Carry-forward**：`carryforward.run_carryforward_experiment.main` → `run_carryforward_experiment` → `_process_one_underlying_cf` → `carryforward.carryforward_experiment_common`（`load_d0_for_cf` / `iter_mo_horizon_features`）→ `rolling_train_predict_carryforward`。
3. **汇总**：`analyze_summary_eqweight.main` → `run_eqweight_analysis` / `run_rigorous_analysis`。

## 函数定义索引（`def` 行号）

路径相对于 `push-mainline/`。行号为 **`def` 所在行**，改代码后可能变化，需同步更新。

### 入口

| 函数 | 文件 | 行号 |
|------|------|------|
| `main` | `high-frequency/run_data_pipeline.py` | 452 |
| `run_experiment` | `high-frequency/run_data_pipeline.py` | 378 |
| `main` | `high-frequency/carryforward/run_carryforward_experiment.py` | 324 |
| `run_carryforward_experiment` | `high-frequency/carryforward/run_carryforward_experiment.py` | 190 |
| `main` | `high-frequency/analyze_summary_eqweight.py` | 680 |

### `run_data_pipeline.py`

| 函数 | 行号 |
|------|------|
| `_ensure_parquets` | 61 |
| `_clean` | 83 |
| `_spot` | 99 |
| `_read_trade_for_tickers` | 113 |
| `_pool_worker_init` | 167 |
| `_write_explain_csv` | 189 |
| `_collect_results` | 202 |
| `_process_one_underlying` | 222 |
| `_safe_process_one` | 357 |
| `_write_ticker_csv` | 368 |
| `run_experiment` | 378 |
| `main` | 452 |

### `carryforward/run_carryforward_experiment.py`

| 函数 | 行号 |
|------|------|
| `_pool_worker_init_cf` | 78 |
| `_process_one_underlying_cf` | 115 |
| `_safe_process_one_cf` | 183 |
| `run_carryforward_experiment` | 190 |
| `run_synthetic_selftest` | 273 |
| `main` | 324 |

### `model_training.py`

| 函数 | 行号 |
|------|------|
| `_linear_pipeline` | 30 |
| `get_model_pipelines` | 35 |
| `_rolling_train_predict_core` | 69 |
| `rolling_train_predict` | 126 |
| `rolling_train_predict_with_explain` | 152 |
| `evaluate_predictions` | 180 |

### `model_explain.py`

| 函数 | 行号 |
|------|------|
| `_X_transformed` | 13 |
| `explain_snapshot` | 23 |

### `data_processing.py`

| 函数 | 行号 |
|------|------|
| `parse_option_ticker` | 15 |
| `load_trade` | 34 |
| `add_trade_dt_cols` | 56 |
| `fix_dt_et_from_trade_ts` | 70 |
| `load_quote_csv` | 78 |
| `filter_quotes_sp100` | 85 |
| `merge_trades_quotes` | 112 |
| `filter_rth` | 140 |
| `clean_merged_option_data` | 155 |
| `deduplicate_by_ticker_trade_ts` | 229 |
| `add_y_5s_return` | 244 |

### `feature_engineering.py`

| 函数 | 行号 |
|------|------|
| `get_predictor_cols` | 13 |
| `_span_label` | 24 |
| `_safe_nanmean` | 33 |
| `_safe_nanmax` | 40 |
| `_safe_nansum` | 47 |
| `add_trade_direction_proxy` | 54 |
| `build_calendar_predictors` | 110 |
| `check_span_occupancy` | 329 |

### `model_plots.py`

| 函数 | 行号 |
|------|------|
| `_has_y_pred_cols` | 14 |
| `plot_pred_vs_actual` | 22 |
| `plot_residuals` | 45 |
| `plot_model_comparison` | 72 |
| `plot_rolling_r2` | 103 |
| `plot_hourly_direction_accuracy` | 136 |
| `plot_coef_heatmap` | 162 |
| `plot_multi_model_rolling_r2` | 195 |
| `plot_summary_metrics_boxplot` | 233 |
| `plot_summary_boxplot_facets` | 266（弃用桩，`NotImplementedError`） |
| `plot_summary_metric_boxplots` | 283（弃用桩） |
| `plot_boxplot_facets` / `plot_metric_boxplots` | 298–299（别名） |

### `analyze_summary_eqweight.py`

| 函数 | 行号 |
|------|------|
| `parse_args` | 30 |
| `model_order_present` | 73 |
| `ensure_numeric` | 81 |
| `complete_cells` | 93 |
| `equal_weight_over_underlyings` | 102 |
| `overall_equal_weight_summary` | 118 |
| `equal_weight_win_rate` | 137 |
| `horizon_decay_equal_weight` | 155 |
| `moneyness_equal_weight` | 164 |
| `weighted_summary` | 173 |
| `weighted_mean` | 196 |
| `sensitivity_analysis` | 208 |
| `boxplot_facets` | 224 |
| `lineplot_metrics` | 277 |
| `compare_equal_vs_weighted` | 305 |
| `plot_metric_facets_rigorous` | 334 |
| `plot_n_distribution` | 392 |
| `plot_weighted_summary_by_horizon` | 416 |
| `write_eqweight_readme` | 459 |
| `write_rigorous_readme` | 486 |
| `run_eqweight_analysis` | 521 |
| `run_rigorous_analysis` | 584 |
| `main` | 611 |

### `carryforward/carryforward_experiment_common.py`

| 函数 | 行号 |
|------|------|
| `rolling_train_predict_carryforward` | 39 |
| `load_d0_for_cf` | 138 |
| `iter_mo_horizon_features` | 176 |
| `write_by_ticker_csv` | 223 |

说明：`config.py` 仅常量，无顶层 `def`。各文件内嵌套的 `def`（如 `add_y_5s_return` 中的 `_one_group`）未单独列出，可在文件内搜索 `def ` 定位。
