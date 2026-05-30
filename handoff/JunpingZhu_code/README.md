# Handoff Experiment Code

## Purpose
This folder keeps the minimal delivery version of the experiment code.
It only retains the core path:
- data loading
- feature construction
- model input assembly
- training / evaluation

## Included files
- `extract_aapl_from_sp100_parquet.py`
  - Extract AAPL parquet from SP100 parquet with DuckDB.
- `run_aapl_single_day.py`
  - User-facing single-day entrypoint.
- `run_aapl_multi_day.py`
  - User-facing multi-day entrypoint.
- `multi_day_runner.py`
  - Multi-day orchestration and expanding-window evaluation.
- `single_day_runner.py`
  - Thin single-day orchestration layer.
- `single_day_columns.py`
  - Shared single-day feature names and span definitions.
- `single_day_data.py`
  - Quote/trade loading and cleaning.
- `single_day_features.py`
  - Trade/quote alignment and feature construction.
- `single_day_modeling.py`
  - Single-day evaluation and metric utilities.
- `run_single_day_cross_section_chunked.py`
  - Backward-compatible wrapper around the refactored single-day modules.
- `run_aapl_cross_day.py`
  - Backward-compatible wrapper around the refactored multi-day modules.
- `VARIABLE_DEFINITIONS.md`
  - Variable definitions and GitHub-alignment notes.

## Recommended reading order
1. `VARIABLE_DEFINITIONS.md`
2. `run_aapl_single_day.py`
3. `single_day_runner.py`
4. `single_day_columns.py`
5. `single_day_data.py`
6. `single_day_features.py`
7. `single_day_modeling.py`
8. `run_aapl_multi_day.py`
9. `multi_day_runner.py`
10. `run_aapl_cross_day.py`
11. `run_single_day_cross_section_chunked.py`
12. `extract_aapl_from_sp100_parquet.py`

## Entrypoints
- Single-day experiments: `run_aapl_single_day.py`
- Multi-day experiments: `run_aapl_multi_day.py`

## Where Functions Live

### Single-day entry and orchestration
- [run_aapl_single_day.py](C:/Users/a9249/Desktop/hkust(GZ)-RA/har-transformer/handoff_from_server/JunpingZhu_code/run_aapl_single_day.py)
  - User-facing single-day CLI entrypoint
- [single_day_runner.py](C:/Users/a9249/Desktop/hkust(GZ)-RA/har-transformer/handoff_from_server/JunpingZhu_code/single_day_runner.py)
  - `parse_args`
  - `resolve_universe_file`
  - `run`
  - `main`

### Single-day shared definitions
- [single_day_columns.py](C:/Users/a9249/Desktop/hkust(GZ)-RA/har-transformer/handoff_from_server/JunpingZhu_code/single_day_columns.py)
  - `HORIZONS`
  - `TRADE_SPANS`
  - `BASE_FEATURE_COLS`
  - `FEATURE_COLS`
  - `span_label`
  - legacy column-name mapping tables

### Quote/trade loading and cleaning
- [single_day_data.py](C:/Users/a9249/Desktop/hkust(GZ)-RA/har-transformer/handoff_from_server/JunpingZhu_code/single_day_data.py)
  - `load_sp100_symbols`
  - `detect_columns`
  - `detect_trade_columns`
  - `parse_timestamp_series`
  - `parse_option_ticker`
  - `clean_quotes_chunk`
  - `load_trade_events`
  - `chunk_list`
  - `get_unique_underlyings`
  - `write_cleaned_chunk_parquets`

### Trade alignment and feature construction
- [single_day_features.py](C:/Users/a9249/Desktop/hkust(GZ)-RA/har-transformer/handoff_from_server/JunpingZhu_code/single_day_features.py)
  - `_window_stats`
  - `_window_sum`
  - `_safe_divide`
  - `ensure_aligned_feature_columns`
  - `merge_trades_with_quotes_asof`
  - `add_trade_direction_proxy`
  - `build_trade_aligned_features_labels_for_horizon`
  - `build_features_labels_for_horizon`
  - `build_horizon_chunk`
  - `build_feature_chunk_files`

### Single-day modeling and metrics
- [single_day_modeling.py](C:/Users/a9249/Desktop/hkust(GZ)-RA/har-transformer/handoff_from_server/JunpingZhu_code/single_day_modeling.py)
  - `build_rolling_windows`
  - `build_time_split_window`
  - `balanced_sample_train`
  - `init_stats`
  - `update_stats`
  - `finalize_stats`
  - `add_dte_bucket`
  - `query_window_df`
  - `prepare_xy`
  - `run_horizon_model_from_chunks`

### Multi-day entry and orchestration
- [run_aapl_multi_day.py](C:/Users/a9249/Desktop/hkust(GZ)-RA/har-transformer/handoff_from_server/JunpingZhu_code/run_aapl_multi_day.py)
  - User-facing multi-day CLI entrypoint
- [multi_day_runner.py](C:/Users/a9249/Desktop/hkust(GZ)-RA/har-transformer/handoff_from_server/JunpingZhu_code/multi_day_runner.py)
  - `Roll`
  - `parse_args`
  - `load_dates`
  - `build_expanding_rolls`
  - `write_roll_plan`
  - `append_log`
  - `parse_horizons`
  - `get_prep_dir`
  - `get_horizon_chunk_paths`
  - `ensure_prepared_day`
  - `read_feature_files`
  - `prepare_xy_for_model`
  - `build_model`
  - `build_pred_df`
  - `summarize_pred_df`
  - `configure_model_globals`
  - `log_run_header`
  - `run_one_roll_horizon`
  - `run`
  - `main`

### Compatibility wrappers
- [run_single_day_cross_section_chunked.py](C:/Users/a9249/Desktop/hkust(GZ)-RA/har-transformer/handoff_from_server/JunpingZhu_code/run_single_day_cross_section_chunked.py)
  - Backward-compatible re-export layer for the refactored single-day modules
- [run_aapl_cross_day.py](C:/Users/a9249/Desktop/hkust(GZ)-RA/har-transformer/handoff_from_server/JunpingZhu_code/run_aapl_cross_day.py)
  - Backward-compatible re-export layer for the refactored multi-day modules

### Utility script
- [extract_aapl_from_sp100_parquet.py](C:/Users/a9249/Desktop/hkust(GZ)-RA/har-transformer/handoff_from_server/JunpingZhu_code/extract_aapl_from_sp100_parquet.py)
  - `parse_args`
  - `main`

## Quick Navigation
- If you want to change input cleaning: go to `single_day_data.py`
- If you want to change trade/quote alignment or feature formulas: go to `single_day_features.py`
- If you want to change single-day train/test windows or metrics: go to `single_day_modeling.py`
- If you want to change single-day CLI behavior: go to `single_day_runner.py`
- If you want to change multi-day roll logic: go to `multi_day_runner.py`
- If you want to keep old imports working: check the compatibility wrappers

## Note
This is intentionally a simplified handoff package rather than the full working repo.
