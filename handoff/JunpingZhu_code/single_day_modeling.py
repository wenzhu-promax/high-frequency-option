"""Modeling utilities for the single-day and multi-day pipelines."""

from __future__ import annotations

import math

import duckdb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from single_day_columns import FEATURE_COLS
from single_day_data import EASTERN_TZ


def build_rolling_windows(date_str, train_minutes, refit_every_minutes):
    market_open = pd.Timestamp(f"{date_str} 09:30:00", tz=EASTERN_TZ)
    market_close = pd.Timestamp(f"{date_str} 16:00:00", tz=EASTERN_TZ)
    train_delta = pd.Timedelta(minutes=train_minutes)
    step_delta = pd.Timedelta(minutes=refit_every_minutes)
    split_id = 0
    windows = []
    test_start = market_open + train_delta
    while test_start < market_close:
        train_start = test_start - train_delta
        test_end = min(test_start + step_delta, market_close)
        windows.append(
            {
                "split_id": split_id,
                "train_start": train_start,
                "train_end": test_start,
                "test_start": test_start,
                "test_end": test_end,
                "train_start_ns": train_start.tz_convert("UTC").value,
                "train_end_ns": test_start.tz_convert("UTC").value,
                "test_start_ns": test_start.tz_convert("UTC").value,
                "test_end_ns": test_end.tz_convert("UTC").value,
            }
        )
        split_id += 1
        test_start = test_end
    return windows


def build_time_split_window(date_str, train_cutoff):
    market_open = pd.Timestamp(f"{date_str} 09:30:00", tz=EASTERN_TZ)
    market_close = pd.Timestamp(f"{date_str} 16:00:00", tz=EASTERN_TZ)
    cutoff = pd.Timestamp(f"{date_str} {train_cutoff}", tz=EASTERN_TZ)
    return [
        {
            "split_id": 0,
            "train_start": market_open,
            "train_end": cutoff,
            "test_start": cutoff,
            "test_end": market_close,
            "train_start_ns": market_open.tz_convert("UTC").value,
            "train_end_ns": cutoff.tz_convert("UTC").value,
            "test_start_ns": cutoff.tz_convert("UTC").value,
            "test_end_ns": market_close.tz_convert("UTC").value,
        }
    ]


def balanced_sample_train(train_df, max_per_underlying):
    sampled = (
        train_df.sort_values(["underlying", "timestamp_ns"])
        .groupby("underlying", group_keys=False)
        .head(max_per_underlying)
        .reset_index(drop=True)
    )
    print(
        f"Balanced sampling: train before={len(train_df):,}, after={len(sampled):,}, underlyings={sampled['underlying'].nunique():,}",
        flush=True,
    )
    return sampled


def init_stats():
    return {"n": 0, "sum_y": 0.0, "sum_y2": 0.0, "sse": 0.0, "sae": 0.0, "correct": 0}


def update_stats(stats, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    stats["n"] += len(y_true)
    stats["sum_y"] += float(y_true.sum())
    stats["sum_y2"] += float((y_true ** 2).sum())
    stats["sse"] += float(((y_true - y_pred) ** 2).sum())
    stats["sae"] += float(np.abs(y_true - y_pred).sum())
    stats["correct"] += int(((y_true > 0) == (y_pred > 0)).sum())


def finalize_stats(stats):
    n = stats["n"]
    if n == 0:
        return {"n_samples": 0, "r2": np.nan, "mse": np.nan, "rmse": np.nan, "mae": np.nan, "direction_accuracy": np.nan}
    mean_y = stats["sum_y"] / n
    denom = stats["sum_y2"] - n * mean_y * mean_y
    r2 = np.nan if denom <= 0 else 1.0 - stats["sse"] / denom
    mse = stats["sse"] / n
    mae = stats["sae"] / n
    acc = stats["correct"] / n
    return {"n_samples": n, "r2": r2, "mse": mse, "rmse": math.sqrt(mse), "mae": mae, "direction_accuracy": acc}


def add_dte_bucket(df):
    dte_col = "dte" if "dte" in df.columns else "DTE"
    bucket = np.where(
        df[dte_col] == 0,
        "0DTE",
        np.where(
            (df[dte_col] >= 1) & (df[dte_col] <= 7),
            "1-7DTE",
            np.where((df[dte_col] >= 8) & (df[dte_col] <= 30), "8-30DTE", "30DTE+"),
        ),
    )
    out = df.copy()
    out["DTE_bucket"] = bucket
    return out


def query_window_df(con, file_paths, start_ns, end_ns):
    if not file_paths:
        return pd.DataFrame()
    file_list = ", ".join("'" + str(p).replace("\\", "/").replace("'", "''") + "'" for p in file_paths)
    sql = f"""
        SELECT *
        FROM read_parquet([{file_list}])
        WHERE timestamp_ns >= {int(start_ns)} AND timestamp_ns < {int(end_ns)}
    """
    return con.execute(sql).fetch_df()


def prepare_xy(train_df, test_df):
    train_df = train_df.dropna(subset=["return_mid_avg"]).copy()
    test_df = test_df.dropna(subset=["return_mid_avg"]).copy()
    if train_df.empty or test_df.empty:
        return None
    X_train = train_df[FEATURE_COLS].copy()
    X_test = test_df[FEATURE_COLS].copy()
    y_train = train_df["return_mid_avg"].to_numpy(dtype=float)
    y_test = test_df["return_mid_avg"].to_numpy(dtype=float)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test = scaler.transform(imputer.transform(X_test))
    return train_df, test_df, X_train, X_test, y_train, y_test


def run_horizon_model_from_chunks(horizon, file_paths, date_str, out_dir, eval_mode, train_cutoff, max_train_per_underlying, train_minutes, refit_every_minutes):
    if not file_paths:
        return None
    con = duckdb.connect(database=":memory:")
    windows = build_time_split_window(date_str, train_cutoff) if eval_mode == "time_split" else build_rolling_windows(date_str, train_minutes, refit_every_minutes)
    print(f"horizon={horizon}s windows prepared: {len(windows)}", flush=True)

    pred_out = out_dir / f"predictions_horizon_{horizon}s.csv"
    header_written = False
    pooled = init_stats()
    per_underlying = {}
    per_bucket = {}
    split_train_sizes = []
    n_refits = 0

    for window in windows:
        train_df = query_window_df(con, file_paths, window["train_start_ns"], window["train_end_ns"])
        test_df = query_window_df(con, file_paths, window["test_start_ns"], window["test_end_ns"])
        print(f"horizon={horizon}s split={window['split_id']} train_rows={len(train_df):,} test_rows={len(test_df):,}", flush=True)
        if train_df.empty or test_df.empty:
            continue
        train_df = balanced_sample_train(train_df, max_train_per_underlying)
        prepared = prepare_xy(train_df, test_df)
        if prepared is None:
            continue
        train_df, test_df, X_train, X_test, y_train, y_test = prepared
        model = Ridge(alpha=10.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        split_train_sizes.append(len(train_df))
        n_refits += 1

        pred_df = test_df[["date", "timestamp", "underlying", "option_ticker", "horizon", "option_type", "dte", "current_relative_spread", "n_future_quotes"]].copy()
        pred_df = pred_df.rename(columns={"current_relative_spread": "relative_spread"})
        pred_df["y_true"] = y_test
        pred_df["y_pred"] = y_pred
        pred_df["direction_true"] = (y_test > 0).astype(int)
        pred_df["direction_pred"] = (y_pred > 0).astype(int)
        pred_df["split_id"] = window["split_id"]
        pred_df["train_start"] = window["train_start"]
        pred_df["train_end"] = window["train_end"]
        pred_df["test_start"] = window["test_start"]
        pred_df["test_end"] = window["test_end"]

        pred_df_out = pred_df.copy()
        pred_df_out["timestamp"] = pred_df_out["timestamp"].astype(str)
        for col in ["train_start", "train_end", "test_start", "test_end"]:
            pred_df_out[col] = pred_df_out[col].astype(str)
        pred_df_out.to_csv(pred_out, index=False, mode="a", header=not header_written)
        header_written = True

        update_stats(pooled, y_test, y_pred)
        for underlying, group in pred_df.groupby("underlying"):
            stats = per_underlying.setdefault(underlying, init_stats())
            update_stats(stats, group["y_true"], group["y_pred"])
        for option_type, group in pred_df.groupby("option_type"):
            stats = per_bucket.setdefault(("option_type", option_type), init_stats())
            update_stats(stats, group["y_true"], group["y_pred"])
        with_bucket = add_dte_bucket(pred_df)
        for bucket_name, group in with_bucket.groupby("DTE_bucket"):
            stats = per_bucket.setdefault(("DTE_bucket", bucket_name), init_stats())
            update_stats(stats, group["y_true"], group["y_pred"])

    con.close()
    if pooled["n"] == 0:
        return None

    pooled_final = finalize_stats(pooled)
    underlying_rows = []
    eligible_r2 = []
    eligible_acc = []
    for underlying, stats in sorted(per_underlying.items()):
        final = finalize_stats(stats)
        include = int(final["n_samples"] >= 1000)
        if include:
            eligible_r2.append(final["r2"])
            eligible_acc.append(final["direction_accuracy"])
        underlying_rows.append(
            {
                "date": date_str,
                "horizon": f"{horizon}s",
                "model": "Ridge",
                "model_scope": "pooled_cross_sectional",
                "eval_mode": eval_mode,
                "underlying": underlying,
                "n_samples": final["n_samples"],
                "r2": final["r2"],
                "direction_accuracy": final["direction_accuracy"],
                "include_in_equal_weight": include,
            }
        )

    bucket_rows = []
    for (bucket_type, bucket_name), stats in sorted(per_bucket.items()):
        final = finalize_stats(stats)
        bucket_rows.append(
            {
                "date": date_str,
                "horizon": f"{horizon}s",
                "model": "Ridge",
                "model_scope": "pooled_cross_sectional",
                "eval_mode": eval_mode,
                "bucket_type": bucket_type,
                "bucket_name": bucket_name,
                "n_samples": final["n_samples"],
                "r2": final["r2"],
                "direction_accuracy": final["direction_accuracy"],
            }
        )

    summary = {
        "date": date_str,
        "horizon": f"{horizon}s",
        "model": "Ridge",
        "model_scope": "pooled_cross_sectional",
        "eval_mode": eval_mode,
        "n_train": int(np.sum(split_train_sizes)),
        "n_test": pooled_final["n_samples"],
        "avg_train_per_refit": float(np.mean(split_train_sizes)) if split_train_sizes else np.nan,
        "n_refits": n_refits,
        "pooled_r2": pooled_final["r2"],
        "pooled_mse": pooled_final["mse"],
        "pooled_rmse": pooled_final["rmse"],
        "pooled_mae": pooled_final["mae"],
        "pooled_direction_accuracy": pooled_final["direction_accuracy"],
        "eq_avg_r2": float(np.nanmean(eligible_r2)) if eligible_r2 else np.nan,
        "eq_avg_direction_accuracy": float(np.nanmean(eligible_acc)) if eligible_acc else np.nan,
        "n_underlyings": len(eligible_r2),
    }
    print(f"horizon={horizon}s pooled_r2={summary['pooled_r2']:.6f} direction_acc={summary['pooled_direction_accuracy']:.6f}", flush=True)
    return summary, pd.DataFrame(underlying_rows), pd.DataFrame(bucket_rows)
