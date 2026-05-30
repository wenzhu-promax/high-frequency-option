"""Compatibility wrapper for the refactored multi-day pipeline.

New code should import from `multi_day_runner`.
"""

from multi_day_runner import (
    HORIZONS,
    FEATURE_COLS,
    Roll,
    append_log,
    build_expanding_rolls,
    ensure_prepared_day,
    get_horizon_chunk_paths,
    get_prep_dir,
    load_dates,
    main,
    parse_args,
    parse_horizons,
    prepare_xy_for_model,
    read_feature_files,
    run,
    run_one_roll_horizon,
    write_roll_plan,
)

__all__ = [
    "FEATURE_COLS",
    "HORIZONS",
    "Roll",
    "append_log",
    "build_expanding_rolls",
    "ensure_prepared_day",
    "get_horizon_chunk_paths",
    "get_prep_dir",
    "load_dates",
    "main",
    "parse_args",
    "parse_horizons",
    "prepare_xy_for_model",
    "read_feature_files",
    "run",
    "run_one_roll_horizon",
    "write_roll_plan",
]


if __name__ == "__main__":
    main()
