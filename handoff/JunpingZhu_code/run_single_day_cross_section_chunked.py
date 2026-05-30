"""Compatibility wrapper for the refactored single-day pipeline.

New code should import from:
- `single_day_columns`
- `single_day_data`
- `single_day_features`
- `single_day_modeling`
- `single_day_runner`
"""

from single_day_columns import FEATURE_COLS, HORIZONS, TRADE_SPANS
from single_day_data import (
    EASTERN_TZ,
    TICKER_PATTERN,
    clean_quotes_chunk,
    detect_columns,
    detect_trade_columns,
    get_unique_underlyings,
    load_sp100_symbols,
    load_trade_events,
    parse_option_ticker,
    parse_timestamp_series,
    write_cleaned_chunk_parquets,
)
from single_day_features import (
    add_trade_direction_proxy,
    build_feature_chunk_files,
    build_features_labels_for_horizon,
    build_horizon_chunk,
    build_trade_aligned_features_labels_for_horizon,
    ensure_aligned_feature_columns,
    merge_trades_with_quotes_asof,
)
from single_day_modeling import (
    add_dte_bucket,
    balanced_sample_train,
    build_rolling_windows,
    build_time_split_window,
    finalize_stats,
    init_stats,
    prepare_xy,
    query_window_df,
    run_horizon_model_from_chunks,
    update_stats,
)
from single_day_runner import main, parse_args, resolve_universe_file, run

__all__ = [
    "EASTERN_TZ",
    "FEATURE_COLS",
    "HORIZONS",
    "TRADE_SPANS",
    "TICKER_PATTERN",
    "add_dte_bucket",
    "add_trade_direction_proxy",
    "balanced_sample_train",
    "build_feature_chunk_files",
    "build_features_labels_for_horizon",
    "build_horizon_chunk",
    "build_rolling_windows",
    "build_time_split_window",
    "build_trade_aligned_features_labels_for_horizon",
    "clean_quotes_chunk",
    "detect_columns",
    "detect_trade_columns",
    "ensure_aligned_feature_columns",
    "finalize_stats",
    "get_unique_underlyings",
    "init_stats",
    "load_sp100_symbols",
    "load_trade_events",
    "main",
    "merge_trades_with_quotes_asof",
    "parse_args",
    "parse_option_ticker",
    "parse_timestamp_series",
    "prepare_xy",
    "query_window_df",
    "resolve_universe_file",
    "run",
    "run_horizon_model_from_chunks",
    "update_stats",
    "write_cleaned_chunk_parquets",
]


if __name__ == "__main__":
    main()
