"""User-facing multi-day runner and supporting helpers."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

from single_day_columns import FEATURE_COLS, HORIZONS
from single_day_data import load_sp100_symbols, load_trade_events, write_cleaned_chunk_parquets
from single_day_features import build_feature_chunk_files
from single_day_modeling import add_dte_bucket, balanced_sample_train, finalize_stats, init_stats, update_stats


@dataclass
class Roll:
    roll_id: int
    train_start_date: str
    train_end_date: str
    test_date: str
    n_train_days: int


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run AAPL expanding-window cross-day experiments")
    parser.add_argument("--dates-file", required=True, help="One YYYY-MM-DD per line.")
    parser.add_argument("--raw-quote-path-template", required=True, help="Template with {date}.")
    parser.add_argument("--universe-file", required=True, help="Usually config/aapl_symbols.txt")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--raw-trade-path-template", default=None, help="Template with {date} for trade files.")
    parser.add_argument("--min-train-days", type=int, default=5)
    parser.add_argument("--max-train-per-underlying", type=int, default=5_000_000)
    parser.add_argument("--underlying-batch-size", type=int, default=1)
    parser.add_argument("--n-workers", type=int, default=2)
    parser.add_argument("--scan-batch-size", type=int, default=2_000_000)
    parser.add_argument("--min-dte", type=int, default=0)
    parser.add_argument("--max-dte", type=int, default=14)
    parser.add_argument("--min-quotes-per-contract", type=int, default=100)
    parser.add_argument("--top-n-contracts", type=int, default=50)
    parser.add_argument("--model", default="ridge", choices=["ridge", "lasso", "rf"])
    parser.add_argument("--label-mode", default="mid_avg", choices=["mid_avg", "trade_avg"])
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--rf-n-estimators", type=int, default=100)
    parser.add_argument("--rf-max-depth", type=int, default=8)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=20)
    parser.add_argument("--rf-n-jobs", type=int, default=-1)
    parser.add_argument("--horizons", default="5,10,30,60")
    parser.add_argument("--skip-existing-prep", action="store_true")
    return parser.parse_args(argv)


def load_dates(path: Path) -> list[str]:
    dates = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            d = line.strip()
            if not d:
                continue
            datetime.strptime(d, "%Y-%m-%d")
            dates.append(d)
    dates = sorted(set(dates))
    if len(dates) < 2:
        raise ValueError("Need at least 2 dates.")
    return dates


def build_expanding_rolls(dates: list[str], min_train_days: int) -> list[Roll]:
    if min_train_days < 1:
        raise ValueError("min_train_days must be >= 1.")
    if len(dates) <= min_train_days:
        raise ValueError("Not enough dates for at least one roll.")
    rolls = []
    roll_id = 1
    for test_idx in range(min_train_days, len(dates)):
        rolls.append(Roll(roll_id=roll_id, train_start_date=dates[0], train_end_date=dates[test_idx - 1], test_date=dates[test_idx], n_train_days=test_idx))
        roll_id += 1
    return rolls


def write_roll_plan(rolls: list[Roll], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["roll_id", "train_start_date", "train_end_date", "test_date", "n_train_days"])
        for r in rolls:
            writer.writerow([r.roll_id, r.train_start_date, r.train_end_date, r.test_date, r.n_train_days])


def append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def parse_horizons(horizon_str: str) -> list[int]:
    vals = sorted(set(int(token.strip()) for token in horizon_str.split(",") if token.strip()))
    invalid = [h for h in vals if h not in HORIZONS]
    if invalid:
        raise ValueError(f"Unsupported horizons {invalid}. Supported={HORIZONS}")
    if not vals:
        raise ValueError("No horizons requested.")
    return vals


def get_prep_dir(out_root: Path, date_str: str) -> Path:
    return out_root / "prepared_by_day" / date_str


def get_horizon_chunk_paths(prep_dir: Path, horizon: int) -> list[Path]:
    return sorted((prep_dir / "horizon_chunks").glob(f"h{horizon}_chunk_*.parquet"))


def ensure_prepared_day(args, date_str: str, raw_quote_path: Path, universe_symbols: set[str], out_root: Path, selected_horizons: list[int]) -> dict[int, list[Path]]:
    prep_dir = get_prep_dir(out_root, date_str)
    horizon_paths = {h: get_horizon_chunk_paths(prep_dir, h) for h in selected_horizons}
    if args.skip_existing_prep and all(horizon_paths[h] for h in selected_horizons):
        print(f"Using cached prepared chunks for {date_str}", flush=True)
        return horizon_paths
    if all(horizon_paths[h] for h in selected_horizons):
        print(f"Prepared chunks already exist for {date_str}", flush=True)
        return horizon_paths

    print(f"Preparing date={date_str} raw_quote_path={raw_quote_path}", flush=True)
    clean_chunk_paths = write_cleaned_chunk_parquets(
        str(raw_quote_path),
        pd.Timestamp(date_str).date(),
        universe_symbols,
        args.underlying_batch_size,
        args.scan_batch_size,
        prep_dir,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
    )
    trade_parquet_path = None
    if args.raw_trade_path_template:
        raw_trade_path = Path(args.raw_trade_path_template.format(date=date_str))
        print(f"Preparing trade events from raw_trade_path={raw_trade_path}", flush=True)
        trades_df = load_trade_events(raw_trade_path, pd.Timestamp(date_str).date(), universe_symbols, scan_batch_size=args.scan_batch_size)
        trade_parquet_path = prep_dir / "filtered_trades.parquet"
        trades_df.to_parquet(trade_parquet_path, index=False)
        print(f"Prepared trade events rows={len(trades_df):,} file={trade_parquet_path}", flush=True)
    build_feature_chunk_files(
        clean_chunk_paths,
        pd.Timestamp(date_str).date(),
        prep_dir,
        args.n_workers,
        selected_horizons=selected_horizons,
        min_quotes_per_contract=args.min_quotes_per_contract,
        top_n_contracts=args.top_n_contracts,
        label_mode=args.label_mode,
        trade_parquet_path=trade_parquet_path,
    )
    out = {h: get_horizon_chunk_paths(prep_dir, h) for h in selected_horizons}
    for h in selected_horizons:
        print(f"Prepared date={date_str} horizon={h}s files={len(out[h])}", flush=True)
    return out


TRAIN_REQUIRED_COLS = [
    "date",
    "timestamp",
    "timestamp_ns",
    "underlying",
    "option_ticker",
    "horizon",
    "option_type",
    "dte",
    "current_relative_spread",
    "n_future_quotes",
    "return_mid_avg",
] + FEATURE_COLS


def _dedupe_keep_order(cols: list[str]) -> list[str]:
    seen = set()
    out = []
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def read_feature_files(file_paths: list[Path], columns: list[str]) -> pd.DataFrame:
    if not file_paths:
        return pd.DataFrame()
    file_list = ", ".join("'" + str(Path(p).as_posix()).replace("'", "''") + "'" for p in file_paths)
    select_cols = ", ".join(columns)
    sql = f"SELECT {select_cols} FROM read_parquet([{file_list}])"
    con = duckdb.connect(database=":memory:")
    try:
        return con.execute(sql).fetch_df()
    finally:
        con.close()


def prepare_xy_for_model(train_df, test_df, model_name: str):
    train_df = train_df.dropna(subset=["return_mid_avg"]).copy()
    test_df = test_df.dropna(subset=["return_mid_avg"]).copy()
    if train_df.empty or test_df.empty:
        return None
    X_train = train_df[FEATURE_COLS].copy()
    X_test = test_df[FEATURE_COLS].copy()
    y_train = train_df["return_mid_avg"].to_numpy(dtype=float)
    y_test = test_df["return_mid_avg"].to_numpy(dtype=float)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    if model_name in {"ridge", "lasso"}:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return train_df, test_df, X_train, X_test, y_train, y_test


def build_model():
    if MODEL_NAME == "ridge":
        return Ridge(alpha=MODEL_ALPHA)
    if MODEL_NAME == "lasso":
        return Lasso(alpha=MODEL_ALPHA, max_iter=2000)
    if MODEL_NAME == "rf":
        return RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=42,
            n_jobs=RF_N_JOBS,
        )
    raise ValueError(f"Unsupported model {MODEL_NAME}")


def build_pred_df(test_df, y_test, y_pred, roll: Roll):
    pred_df = test_df[["date", "timestamp", "underlying", "option_ticker", "horizon", "option_type", "dte", "current_relative_spread", "n_future_quotes"]].copy()
    pred_df = pred_df.rename(columns={"current_relative_spread": "relative_spread"})
    pred_df["y_true"] = y_test
    pred_df["y_pred"] = y_pred
    pred_df["direction_true"] = (y_test > 0).astype(int)
    pred_df["direction_pred"] = (y_pred > 0).astype(int)
    pred_df["roll_id"] = roll.roll_id
    pred_df["train_start_date"] = roll.train_start_date
    pred_df["train_end_date"] = roll.train_end_date
    pred_df["test_date"] = roll.test_date
    return pred_df


def summarize_pred_df(pred_df, roll: Roll, horizon: int, n_train: int):
    pooled = init_stats()
    update_stats(pooled, pred_df["y_true"], pred_df["y_pred"])
    pooled_final = finalize_stats(pooled)

    per_underlying_rows = []
    for underlying, group in pred_df.groupby("underlying"):
        stats = init_stats()
        update_stats(stats, group["y_true"], group["y_pred"])
        final = finalize_stats(stats)
        per_underlying_rows.append(
            {
                "roll_id": roll.roll_id,
                "train_start_date": roll.train_start_date,
                "train_end_date": roll.train_end_date,
                "test_date": roll.test_date,
                "horizon": f"{horizon}s",
                "underlying": underlying,
                "n_samples": final["n_samples"],
                "r2": final["r2"],
                "direction_accuracy": final["direction_accuracy"],
            }
        )

    bucket_rows = []
    for option_type, group in pred_df.groupby("option_type"):
        stats = init_stats()
        update_stats(stats, group["y_true"], group["y_pred"])
        final = finalize_stats(stats)
        bucket_rows.append(
            {
                "roll_id": roll.roll_id,
                "train_start_date": roll.train_start_date,
                "train_end_date": roll.train_end_date,
                "test_date": roll.test_date,
                "horizon": f"{horizon}s",
                "bucket_type": "option_type",
                "bucket_name": option_type,
                "n_samples": final["n_samples"],
                "r2": final["r2"],
                "direction_accuracy": final["direction_accuracy"],
            }
        )
    with_bucket = add_dte_bucket(pred_df)
    for bucket_name, group in with_bucket.groupby("DTE_bucket"):
        stats = init_stats()
        update_stats(stats, group["y_true"], group["y_pred"])
        final = finalize_stats(stats)
        bucket_rows.append(
            {
                "roll_id": roll.roll_id,
                "train_start_date": roll.train_start_date,
                "train_end_date": roll.train_end_date,
                "test_date": roll.test_date,
                "horizon": f"{horizon}s",
                "bucket_type": "DTE_bucket",
                "bucket_name": bucket_name,
                "n_samples": final["n_samples"],
                "r2": final["r2"],
                "direction_accuracy": final["direction_accuracy"],
            }
        )

    model_label = {"ridge": "Ridge", "lasso": "Lasso", "rf": "RF"}[MODEL_NAME]
    summary = {
        "roll_id": roll.roll_id,
        "train_start_date": roll.train_start_date,
        "train_end_date": roll.train_end_date,
        "test_date": roll.test_date,
        "n_train_days": roll.n_train_days,
        "horizon": f"{horizon}s",
        "model": model_label,
        "model_scope": "single_underlying_cross_day",
        "underlying_scope": "AAPL",
        "n_train": int(n_train),
        "n_test": int(len(pred_df)),
        "pooled_r2": pooled_final["r2"],
        "pooled_mse": pooled_final["mse"],
        "pooled_rmse": pooled_final["rmse"],
        "pooled_mae": pooled_final["mae"],
        "pooled_direction_accuracy": pooled_final["direction_accuracy"],
    }
    return summary, pd.DataFrame(per_underlying_rows), pd.DataFrame(bucket_rows)


def configure_model_globals(args):
    global MODEL_NAME, MODEL_ALPHA, RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF, RF_N_JOBS
    MODEL_NAME = args.model.lower()
    MODEL_ALPHA = args.alpha
    RF_N_ESTIMATORS = args.rf_n_estimators
    RF_MAX_DEPTH = args.rf_max_depth
    RF_MIN_SAMPLES_LEAF = args.rf_min_samples_leaf
    RF_N_JOBS = args.rf_n_jobs


def log_run_header(args, dates, selected_horizons, universe_symbols, log_path: Path, out_root: Path):
    print(f"Loaded dates: {dates}", flush=True)
    print(f"Universe symbols: {sorted(universe_symbols)}", flush=True)
    print(f"Selected horizons: {selected_horizons}", flush=True)
    print(f"Model: {MODEL_NAME} alpha={MODEL_ALPHA}", flush=True)
    if MODEL_NAME == "rf":
        print(f"RF params: n_estimators={RF_N_ESTIMATORS} max_depth={RF_MAX_DEPTH} min_samples_leaf={RF_MIN_SAMPLES_LEAF} n_jobs={RF_N_JOBS}", flush=True)
    print(f"Label mode: {args.label_mode}", flush=True)
    print(f"Filters: min_dte={args.min_dte} max_dte={args.max_dte} min_quotes_per_contract={args.min_quotes_per_contract} top_n_contracts={args.top_n_contracts}", flush=True)

    append_log(log_path, f"START total_dates={len(dates)} out_root={out_root}")
    append_log(log_path, f"HORIZONS {selected_horizons}")
    append_log(log_path, f"MODEL {MODEL_NAME} alpha={MODEL_ALPHA}")
    append_log(log_path, f"LABEL_MODE {args.label_mode}")


def run_one_roll_horizon(roll: Roll, horizon: int, train_files: list[Path], test_files: list[Path], out_dir: Path, max_train_per_underlying: int):
    print(f"roll={roll.roll_id} horizon={horizon}s loading train_files={len(train_files)} test_files={len(test_files)}", flush=True)
    required_cols = _dedupe_keep_order(TRAIN_REQUIRED_COLS)
    train_df = read_feature_files(train_files, required_cols)
    test_df = read_feature_files(test_files, required_cols)
    print(f"roll={roll.roll_id} horizon={horizon}s train_rows={len(train_df):,} test_rows={len(test_df):,}", flush=True)
    if train_df.empty or test_df.empty:
        return None

    train_df = balanced_sample_train(train_df, max_train_per_underlying)
    prepared = prepare_xy_for_model(train_df, test_df, MODEL_NAME)
    if prepared is None:
        return None
    train_df, test_df, X_train, X_test, y_train, y_test = prepared

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    pred_df = build_pred_df(test_df, y_test, y_pred, roll)
    pred_df_out = pred_df.copy()
    pred_df_out["timestamp"] = pred_df_out["timestamp"].astype(str)
    pred_df_out.to_csv(out_dir / f"predictions_roll_{roll.roll_id:02d}_horizon_{horizon}s.csv", index=False)

    summary, per_underlying_df, bucket_df = summarize_pred_df(pred_df, roll, horizon, len(train_df))
    print(f"roll={roll.roll_id} horizon={horizon}s pooled_r2={summary['pooled_r2']:.6f} direction_acc={summary['pooled_direction_accuracy']:.6f}", flush=True)
    return summary, per_underlying_df, bucket_df


def run(args) -> Path:
    configure_model_globals(args)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run.log"
    selected_horizons = parse_horizons(args.horizons)
    dates = load_dates(Path(args.dates_file))
    universe_symbols = load_sp100_symbols(args.universe_file)
    if universe_symbols != {"AAPL"}:
        raise ValueError(f"Expected universe-file with only AAPL, got: {sorted(universe_symbols)}")
    log_run_header(args, dates, selected_horizons, universe_symbols, log_path, out_root)

    per_day_horizon_files = {}
    for idx, date_str in enumerate(dates, start=1):
        raw_quote_path = Path(args.raw_quote_path_template.format(date=date_str))
        if not raw_quote_path.exists():
            append_log(log_path, f"SKIP_PREP missing_raw_quote_path={raw_quote_path}")
            continue
        print(f"[prep {idx}/{len(dates)}] date={date_str}", flush=True)
        per_day_horizon_files[date_str] = ensure_prepared_day(args, date_str, raw_quote_path, universe_symbols, out_root, selected_horizons)

    rolls = build_expanding_rolls(dates, args.min_train_days)
    write_roll_plan(rolls, out_root / "roll_plan.csv")
    print(f"Built {len(rolls)} expanding-window rolls", flush=True)

    summary_rows = []
    underlying_frames = []
    bucket_frames = []
    by_roll_dir = out_root / "by_roll"
    by_roll_dir.mkdir(parents=True, exist_ok=True)

    for roll_idx, roll in enumerate(rolls, start=1):
        print(f"[roll {roll_idx}/{len(rolls)}] roll_id={roll.roll_id} train={roll.train_start_date}..{roll.train_end_date} test={roll.test_date}", flush=True)
        append_log(log_path, f"ROLL {roll.roll_id} train={roll.train_start_date}..{roll.train_end_date} test={roll.test_date}")
        if roll.test_date not in per_day_horizon_files:
            append_log(log_path, f"SKIP_ROLL missing_test_features_for={roll.test_date}")
            continue
        train_dates = [d for d in dates if roll.train_start_date <= d <= roll.train_end_date and d in per_day_horizon_files]
        if not train_dates:
            append_log(log_path, f"SKIP_ROLL no_train_dates_for_roll={roll.roll_id}")
            continue
        roll_dir = by_roll_dir / f"roll_{roll.roll_id:02d}_{roll.test_date}"
        roll_dir.mkdir(parents=True, exist_ok=True)
        for horizon in selected_horizons:
            train_files = []
            for d in train_dates:
                train_files.extend(per_day_horizon_files[d].get(horizon, []))
            test_files = per_day_horizon_files[roll.test_date].get(horizon, [])
            result = run_one_roll_horizon(roll, horizon, train_files, test_files, roll_dir, args.max_train_per_underlying)
            if result is None:
                append_log(log_path, f"SKIP_HORIZON roll={roll.roll_id} horizon={horizon}s")
                continue
            summary, underlying_df, bucket_df = result
            summary_rows.append(summary)
            underlying_frames.append(underlying_df)
            bucket_frames.append(bucket_df)

    summary_df = pd.DataFrame(summary_rows)
    underlying_df = pd.concat(underlying_frames, ignore_index=True) if underlying_frames else pd.DataFrame()
    bucket_df = pd.concat(bucket_frames, ignore_index=True) if bucket_frames else pd.DataFrame()
    summary_df.to_csv(out_root / "metrics_summary_all_rolls.csv", index=False)
    underlying_df.to_csv(out_root / "metrics_by_underlying_all_rolls.csv", index=False)
    bucket_df.to_csv(out_root / "metrics_by_bucket_all_rolls.csv", index=False)
    print(f"Wrote roll summary: {out_root / 'metrics_summary_all_rolls.csv'}", flush=True)
    print(f"Wrote underlying metrics: {out_root / 'metrics_by_underlying_all_rolls.csv'}", flush=True)
    print(f"Wrote bucket metrics: {out_root / 'metrics_by_bucket_all_rolls.csv'}", flush=True)
    return out_root


def main(argv=None):
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
