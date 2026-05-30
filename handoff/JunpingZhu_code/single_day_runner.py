"""User-facing single-day runner with a smaller orchestration surface."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from single_day_columns import HORIZONS
from single_day_data import (
    load_sp100_symbols,
    load_trade_events,
    write_cleaned_chunk_parquets,
)
from single_day_features import (
    build_feature_chunk_files,
)
from single_day_modeling import run_horizon_model_from_chunks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run single-day trade-aligned option experiments.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--raw_quote_path", required=True)
    parser.add_argument("--raw_trade_path", default=None)
    parser.add_argument("--sp100_file", default=None)
    parser.add_argument("--universe_file", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--eval_mode", choices=["rolling", "time_split"], default="rolling")
    parser.add_argument("--train_cutoff", default="13:30:00")
    parser.add_argument("--max_train_per_underlying", type=int, default=50000)
    parser.add_argument("--train_minutes", type=int, default=30)
    parser.add_argument("--refit_every_minutes", type=int, default=5)
    parser.add_argument("--underlying_batch_size", type=int, default=5)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--scan_batch_size", type=int, default=2_000_000)
    parser.add_argument("--min_dte", type=int, default=0)
    parser.add_argument("--max_dte", type=int, default=14)
    parser.add_argument("--min_quotes_per_contract", type=int, default=100)
    parser.add_argument("--top_n_contracts", type=int, default=50)
    parser.add_argument("--label_mode", choices=["mid_avg", "trade_avg"], default="mid_avg")
    args = parser.parse_args(argv)
    if not args.sp100_file and not args.universe_file:
        parser.error("One of --sp100_file or --universe_file is required.")
    return args


def resolve_universe_file(args) -> str:
    return args.universe_file or args.sp100_file


def run(args) -> Path:
    trade_date = pd.Timestamp(args.date).date()
    out_dir = Path(args.out_dir) / args.date
    out_dir.mkdir(parents=True, exist_ok=True)
    universe_file = resolve_universe_file(args)

    print("Running pooled cross-sectional single-day experiment", flush=True)
    print(f"date={args.date}", flush=True)
    print(f"raw_quote_path={args.raw_quote_path}", flush=True)
    print(f"raw_trade_path={args.raw_trade_path}", flush=True)
    print(f"universe_file={universe_file}", flush=True)
    print(f"eval_mode={args.eval_mode}", flush=True)
    print(f"underlying_batch_size={args.underlying_batch_size}", flush=True)
    print(f"n_workers={args.n_workers}", flush=True)
    print(f"label_mode={args.label_mode}", flush=True)

    symbols = load_sp100_symbols(universe_file)
    clean_chunk_paths = write_cleaned_chunk_parquets(
        args.raw_quote_path,
        trade_date,
        symbols,
        args.underlying_batch_size,
        args.scan_batch_size,
        out_dir,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
    )
    print(f"Clean chunk files ready: {len(clean_chunk_paths)}", flush=True)

    trade_parquet_path = None
    if args.raw_trade_path:
        trades_df = load_trade_events(
            args.raw_trade_path,
            trade_date,
            symbols,
            scan_batch_size=args.scan_batch_size,
        )
        trade_parquet_path = out_dir / "filtered_trades.parquet"
        trades_df.to_parquet(trade_parquet_path, index=False)
        print(f"Filtered trades ready: {len(trades_df):,} -> {trade_parquet_path}", flush=True)

    horizon_files = build_feature_chunk_files(
        clean_chunk_paths,
        trade_date,
        out_dir,
        args.n_workers,
        selected_horizons=HORIZONS,
        min_quotes_per_contract=args.min_quotes_per_contract,
        top_n_contracts=args.top_n_contracts,
        label_mode=args.label_mode,
        trade_parquet_path=trade_parquet_path,
    )

    summary_rows = []
    underlying_rows = []
    bucket_rows = []
    for horizon in HORIZONS:
        result = run_horizon_model_from_chunks(
            horizon,
            horizon_files[horizon],
            args.date,
            out_dir,
            args.eval_mode,
            args.train_cutoff,
            args.max_train_per_underlying,
            args.train_minutes,
            args.refit_every_minutes,
        )
        if result is None:
            continue
        summary, underlying_df, bucket_df = result
        summary_rows.append(summary)
        underlying_rows.append(underlying_df)
        bucket_rows.append(bucket_df)

    summary_df = pd.DataFrame(summary_rows)
    underlying_df = pd.concat(underlying_rows, ignore_index=True) if underlying_rows else pd.DataFrame()
    bucket_df = pd.concat(bucket_rows, ignore_index=True) if bucket_rows else pd.DataFrame()

    summary_path = out_dir / "metrics_summary.csv"
    underlying_path = out_dir / "metrics_by_underlying.csv"
    bucket_path = out_dir / "metrics_by_bucket.csv"
    summary_df.to_csv(summary_path, index=False)
    underlying_df.to_csv(underlying_path, index=False)
    bucket_df.to_csv(bucket_path, index=False)

    print(f"Saved summary metrics to {summary_path}", flush=True)
    print(f"Saved underlying metrics to {underlying_path}", flush=True)
    print(f"Saved bucket metrics to {bucket_path}", flush=True)
    return out_dir


def main(argv=None):
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
