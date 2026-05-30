"""Feature construction for trade-aligned single-day experiments."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from single_day_columns import BASE_FEATURE_COLS, FEATURE_COLS, LEGACY_BASE_MAP, LEGACY_SPAN_MAP, TRADE_SPANS, span_label
from single_day_data import EASTERN_TZ


def _window_stats(prefix_sum, prefix_sq, left_idx, right_idx):
    count = right_idx - left_idx
    mean = np.full(len(count), np.nan, dtype=float)
    std = np.full(len(count), np.nan, dtype=float)
    valid = count > 0
    if valid.any():
        sums = prefix_sum[right_idx[valid]] - prefix_sum[left_idx[valid]]
        sqs = prefix_sq[right_idx[valid]] - prefix_sq[left_idx[valid]]
        mean_valid = sums / count[valid]
        var_valid = np.maximum(sqs / count[valid] - mean_valid**2, 0.0)
        std_valid = np.sqrt(var_valid)
        mean[valid] = mean_valid
        std[valid] = std_valid
    return count, mean, std


def _window_sum(prefix_sum, left_idx, right_idx):
    return prefix_sum[right_idx] - prefix_sum[left_idx]


def _safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast(numerator, denominator).shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def ensure_aligned_feature_columns(df):
    out = df.copy()
    for legacy_col, aligned_col in LEGACY_BASE_MAP.items():
        if aligned_col not in out.columns and legacy_col in out.columns:
            out[aligned_col] = out[legacy_col]

    for left, right in TRADE_SPANS:
        label = span_label(left, right)
        for legacy_suffix, aligned_name in LEGACY_SPAN_MAP.items():
            legacy_col = f"window_{left}_{right}s_{legacy_suffix}"
            aligned_col = f"{aligned_name}__{label}"
            if aligned_col not in out.columns and legacy_col in out.columns:
                out[aligned_col] = out[legacy_col]

    for col in BASE_FEATURE_COLS:
        if col not in out.columns:
            out[col] = np.nan
    for col in FEATURE_COLS:
        if col not in out.columns:
            out[col] = np.nan
    if "dte" in out.columns and "DTE" in out.columns:
        out = out.drop(columns=["DTE"])
    return out


def merge_trades_with_quotes_asof(cleaned_df, trades_df, trade_date):
    if cleaned_df.empty or trades_df is None or trades_df.empty:
        return pd.DataFrame()

    quote_cols = [
        "option_ticker",
        "timestamp",
        "timestamp_ns",
        "date",
        "underlying",
        "option_type",
        "option_type_flag",
        "strike",
        "DTE",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "mid",
        "relative_spread",
        "lob_imbalance",
    ]
    quotes = (
        cleaned_df[quote_cols]
        .rename(columns={"timestamp": "quote_timestamp", "timestamp_ns": "quote_timestamp_ns"})
        .sort_values(["quote_timestamp_ns", "option_ticker"])
        .reset_index(drop=True)
    )
    trades = (
        trades_df[["option_ticker", "timestamp_ns", "trade_price", "trade_size"]]
        .rename(columns={"timestamp_ns": "trade_timestamp_ns"})
        .sort_values(["trade_timestamp_ns", "option_ticker"])
        .reset_index(drop=True)
    )
    merged = pd.merge_asof(
        trades,
        quotes,
        left_on="trade_timestamp_ns",
        right_on="quote_timestamp_ns",
        by="option_ticker",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.dropna(subset=["quote_timestamp_ns", "mid", "trade_price", "trade_size"]).copy()
    if merged.empty:
        return merged
    merged["timestamp_ns"] = merged["trade_timestamp_ns"].astype("int64")
    merged["timestamp"] = pd.to_datetime(merged["trade_timestamp_ns"], unit="ns", utc=True).dt.tz_convert(EASTERN_TZ)
    merged["date"] = str(trade_date)
    merged["current_mid"] = pd.to_numeric(merged["mid"], errors="coerce")
    merged["current_relative_spread"] = pd.to_numeric(merged["relative_spread"], errors="coerce")
    merged["current_bid_size"] = pd.to_numeric(merged["bid_size"], errors="coerce")
    merged["current_ask_size"] = pd.to_numeric(merged["ask_size"], errors="coerce")
    merged["current_quote_size_sum"] = merged["current_bid_size"] + merged["current_ask_size"]
    merged["current_lob_imbalance"] = pd.to_numeric(merged["lob_imbalance"], errors="coerce")
    merged["spread"] = pd.to_numeric(merged["ask"], errors="coerce") - pd.to_numeric(merged["bid"], errors="coerce")
    merged = merged[
        merged["timestamp"].notna()
        & np.isfinite(merged["current_mid"])
        & (merged["current_mid"] > 0)
        & np.isfinite(merged["trade_price"])
        & (merged["trade_price"] > 0)
        & np.isfinite(merged["trade_size"])
        & (merged["trade_size"] > 0)
        & np.isfinite(merged["spread"])
        & (merged["spread"] >= 0)
    ].copy()
    if merged.empty:
        return merged

    day_median_spread = merged.groupby("option_ticker")["spread"].transform("median")
    cond_small_both_sizes = (merged["current_bid_size"] <= 1) & (merged["current_ask_size"] <= 1)
    cond_spread_too_large = merged["spread"] > 50 * day_median_spread
    cond_ratio_too_large = (merged["bid"] > 0) & ((merged["ask"] / merged["bid"]) > 5)
    cond_trade_price_outside = (
        (merged["trade_price"] < (merged["bid"] - merged["spread"]))
        | (merged["trade_price"] > (merged["ask"] + merged["spread"]))
    )
    merged = merged[~(cond_small_both_sizes | cond_spread_too_large | cond_ratio_too_large | cond_trade_price_outside)].copy()
    if merged.empty:
        return merged

    merged = (
        merged.sort_values(["option_ticker", "trade_timestamp_ns", "quote_timestamp_ns"])
        .groupby(["option_ticker", "trade_timestamp_ns"], as_index=False)
        .agg(
            {
                "trade_price": "mean",
                "trade_size": "sum",
                "quote_timestamp_ns": "last",
                "quote_timestamp": "last",
                "date": "last",
                "underlying": "last",
                "option_type": "last",
                "option_type_flag": "last",
                "strike": "last",
                "DTE": "last",
                "bid": "last",
                "ask": "last",
                "bid_size": "last",
                "ask_size": "last",
                "mid": "last",
                "relative_spread": "last",
                "lob_imbalance": "last",
                "timestamp": "last",
                "current_mid": "last",
                "current_relative_spread": "last",
                "current_bid_size": "last",
                "current_ask_size": "last",
                "current_quote_size_sum": "last",
                "current_lob_imbalance": "last",
                "spread": "last",
            }
        )
    )
    merged["timestamp_ns"] = merged["trade_timestamp_ns"].astype("int64")
    merged["timestamp"] = pd.to_datetime(merged["trade_timestamp_ns"], unit="ns", utc=True).dt.tz_convert(EASTERN_TZ)
    return merged.sort_values(["option_ticker", "timestamp_ns"]).reset_index(drop=True)


def add_trade_direction_proxy(df):
    if df.empty:
        out = df.copy()
        out["trade_direction"] = np.array([], dtype=int)
        return out

    out = df.sort_values(["option_ticker", "timestamp_ns"]).copy()

    def _one_group(g):
        g = g.copy()
        px = pd.to_numeric(g["trade_price"], errors="coerce").to_numpy(dtype=float)
        mid = pd.to_numeric(g["current_mid"], errors="coerce").to_numpy(dtype=float)
        direction = np.zeros(len(g), dtype=float)
        gt = np.isfinite(px) & np.isfinite(mid) & (px > mid)
        lt = np.isfinite(px) & np.isfinite(mid) & (px < mid)
        eq = np.isfinite(px) & np.isfinite(mid) & (px == mid)
        direction[gt] = 1.0
        direction[lt] = -1.0

        price_diff = np.diff(px, prepend=np.nan)
        tick_sign = np.sign(price_diff)
        last_nonzero = 0.0
        for i in range(len(tick_sign)):
            if np.isfinite(tick_sign[i]) and tick_sign[i] != 0:
                last_nonzero = tick_sign[i]
            elif np.isfinite(tick_sign[i]) and tick_sign[i] == 0:
                tick_sign[i] = last_nonzero

        use_tick = eq | (~gt & ~lt & np.isfinite(px))
        direction[use_tick] = tick_sign[use_tick]
        direction[~np.isfinite(direction)] = 0.0
        direction[direction > 0] = 1.0
        direction[direction < 0] = -1.0
        g["trade_direction"] = direction.astype(int)
        return g

    return out.groupby("option_ticker", group_keys=False, sort=False).apply(_one_group).reset_index(drop=True)


def build_trade_aligned_features_labels_for_horizon(aligned_df, horizon_seconds, trade_date, min_quotes_per_contract=100, top_n_contracts=50, label_mode="trade_avg"):
    if aligned_df.empty:
        print(f"Horizon {horizon_seconds}s valid label count: 0", flush=True)
        return pd.DataFrame()

    contract_counts = aligned_df.groupby("option_ticker").size().rename("trade_count")
    eligible = contract_counts[contract_counts >= min_quotes_per_contract]
    if top_n_contracts and top_n_contracts > 0:
        eligible = eligible.sort_values(ascending=False).head(top_n_contracts)
    selected_contracts = set(eligible.index.tolist())
    if not selected_contracts:
        print(
            f"Horizon {horizon_seconds}s valid label count: 0 | no contracts passed trade-anchor filters min_quotes_per_contract={min_quotes_per_contract} top_n_contracts={top_n_contracts}",
            flush=True,
        )
        return pd.DataFrame()

    aligned_df = aligned_df[aligned_df["option_ticker"].isin(selected_contracts)].copy()
    aligned_df = add_trade_direction_proxy(aligned_df)
    session_close = pd.Timestamp(f"{trade_date} 16:00:00", tz=EASTERN_TZ).tz_convert("UTC").value
    horizon_ns = int(horizon_seconds * 1_000_000_000)
    results = []
    total_valid = 0

    for option_ticker, group in aligned_df.groupby("option_ticker", sort=False):
        g = group.sort_values(["timestamp_ns"]).reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue

        ts = g["timestamp_ns"].to_numpy(dtype=np.int64)
        px = g["trade_price"].to_numpy(dtype=float)
        sz = g["trade_size"].to_numpy(dtype=float)
        mid = g["current_mid"].to_numpy(dtype=float)
        rel_spread = g["current_relative_spread"].to_numpy(dtype=float)
        lob = g["current_lob_imbalance"].to_numpy(dtype=float)
        turnover_base = g["current_quote_size_sum"].to_numpy(dtype=float)
        direction = g["trade_direction"].to_numpy(dtype=float)

        next_idx = np.searchsorted(ts, ts, side="right")
        future_right = np.searchsorted(ts, ts + horizon_ns, side="right")
        future_count = future_right - next_idx
        prefix_px = np.concatenate(([0.0], np.cumsum(px)))
        future_sum = prefix_px[future_right] - prefix_px[next_idx]
        future_avg = np.full(n, np.nan, dtype=float)
        valid_future = (future_count > 0) & ((ts + horizon_ns) <= session_close) & np.isfinite(mid) & (mid > 0)
        future_avg[valid_future] = future_sum[valid_future] / future_count[valid_future]
        return_mid_avg = future_avg / mid - 1.0
        direction_true = (return_mid_avg > 0).astype(float)

        log_px = np.where(px > 0, np.log(px), np.nan)
        trade_ret_1 = np.diff(log_px, prepend=np.nan)
        trade_ret_2 = np.roll(trade_ret_1, 1)
        trade_ret_2[0] = np.nan
        autocov_term = trade_ret_1 * trade_ret_2

        data = {
            "date": g["date"].to_numpy(),
            "timestamp": g["timestamp"].to_numpy(),
            "timestamp_ns": ts,
            "underlying": g["underlying"].to_numpy(),
            "option_ticker": option_ticker,
            "horizon": f"{horizon_seconds}s",
            "return_mid_avg": return_mid_avg,
            "direction": direction_true,
            "n_future_quotes": future_count,
            "n_future_trades": future_count,
            "label_mode": np.full(n, label_mode, dtype=object),
            "current_mid": mid,
            "current_relative_spread": rel_spread,
            "current_bid_size": g["current_bid_size"].to_numpy(dtype=float),
            "current_ask_size": g["current_ask_size"].to_numpy(dtype=float),
            "current_quote_size_sum": turnover_base,
            "current_lob_imbalance": lob,
            "mid": mid,
            "rel_spread": rel_spread,
            "bid_sz": g["current_bid_size"].to_numpy(dtype=float),
            "ask_sz": g["current_ask_size"].to_numpy(dtype=float),
            "quote_sz": turnover_base,
            "lob_imb": lob,
            "strike": g["strike"].to_numpy(dtype=float),
            "dte": pd.to_numeric(g["DTE"], errors="coerce").to_numpy(dtype=float),
            "option_type": g["option_type"].to_numpy(),
            "option_type_flag": g["option_type_flag"].to_numpy(dtype=float),
            "cp_flag": g["option_type_flag"].to_numpy(dtype=float),
        }

        for left_sec, right_sec in TRADE_SPANS:
            label = span_label(left_sec, right_sec)
            breadth = np.zeros(n, dtype=float)
            immediacy = np.full(n, np.nan, dtype=float)
            volume_all = np.full(n, np.nan, dtype=float)
            volume_avg = np.full(n, np.nan, dtype=float)
            volume_max = np.full(n, np.nan, dtype=float)
            mid_lambda = np.full(n, np.nan, dtype=float)
            lob_imbalance = np.full(n, np.nan, dtype=float)
            txn_imbalance = np.full(n, np.nan, dtype=float)
            past_return = np.full(n, np.nan, dtype=float)
            turnover = np.full(n, np.nan, dtype=float)
            mid_autocov = np.full(n, np.nan, dtype=float)
            quoted_spread = np.full(n, np.nan, dtype=float)
            effective_spread = np.full(n, np.nan, dtype=float)
            no_trade_flag = np.ones(n, dtype=int)
            no_quote_flag = np.ones(n, dtype=int)

            left_ns = int(left_sec * 1_000_000_000)
            right_ns = int(right_sec * 1_000_000_000)
            start_idx = np.searchsorted(ts, ts - right_ns, side="right")
            end_idx = np.searchsorted(ts, ts - left_ns, side="right")

            for i in range(n):
                a = start_idx[i]
                b = end_idx[i]
                if b <= a:
                    continue
                w = slice(a, b)
                count = b - a

                # Paper:
                # Breadth(T, Δ1, Δ2, M) = number of transactions in the lookback interval.
                breadth[i] = count
                no_trade_flag[i] = 0
                no_quote_flag[i] = 0

                # Paper:
                # Immediacy(T, Δ1, Δ2, M) = (Δ2 - Δ1) / Breadth(T, Δ1, Δ2, M).
                immediacy[i] = (right_sec - left_sec) / count

                # Paper:
                # VolumeAll = sum of transaction sizes in the interval.
                vol = np.nansum(sz[w])
                volume_all[i] = vol

                # Paper:
                # VolumeAvg = VolumeAll / Breadth.
                volume_avg[i] = vol / count if count > 0 else np.nan

                # Paper:
                # VolumeMax = maximum transaction size in the interval.
                volume_max[i] = np.nanmax(sz[w]) if np.isfinite(sz[w]).any() else np.nan

                first_idx = a
                last_idx = b - 1

                # Paper:
                # Lambda = (Pmax(I) - Pmin(I)) / VolumeAll.
                # Current implementation still uses aligned mid snapshots rather than
                # the paper's transaction-price max/min definition.
                if vol > 0 and np.isfinite(mid[first_idx]) and np.isfinite(mid[last_idx]):
                    mid_lambda[i] = (mid[last_idx] - mid[first_idx]) / vol

                # Paper:
                # LobImbalance = Average[(Sa_t - Sb_t) / (Sa_t + Sb_t)] over the interval.
                # Current implementation averages quote imbalance on trade-aligned snapshots.
                lob_imbalance[i] = np.nanmean(lob[w]) if np.isfinite(lob[w]).any() else np.nan

                # Paper:
                # TxnImbalance = sum(V_t * Dir_t^LR) / VolumeAll.
                # Current implementation uses the trade-direction proxy built upstream.
                if vol > 0:
                    txn_imbalance[i] = np.nansum(sz[w] * direction[w]) / vol

                # Paper:
                # PastReturn = 1 - Average[P_t^txn : t in I] / Pmax(I).
                # Current implementation still uses mid_last in the denominator,
                # so this is intentionally marked here as not yet paper-exact.
                if np.isfinite(mid[last_idx]) and mid[last_idx] > 0:
                    avg_trade_price = np.nanmean(px[w]) if np.isfinite(px[w]).any() else np.nan
                    if np.isfinite(avg_trade_price):
                        past_return[i] = 1.0 - avg_trade_price / mid[last_idx]

                # Paper:
                # Turnover = VolumeAll / S, where S is shares outstanding.
                # Current implementation uses displayed quote size as a placeholder base,
                # so this is not yet paper-exact.
                if np.isfinite(turnover_base[i]) and turnover_base[i] > 0:
                    turnover[i] = vol / turnover_base[i]

                # Paper:
                # AutoCov = Average[log(P_t^txn / P_Lt^txn) * log(P_Lt^txn / P_L(Lt)^txn)].
                mid_autocov[i] = np.nanmean(autocov_term[w]) if np.isfinite(autocov_term[w]).any() else np.nan

                # Paper:
                # QuotedSpread = Average[(P^a_t - P^b_t) / P_t] over the interval.
                quoted_spread[i] = np.nanmean(rel_spread[w]) if np.isfinite(rel_spread[w]).any() else np.nan

                dollar = sz[w] * px[w]
                denom = np.nansum(dollar)
                if denom > 0:
                    # Paper:
                    # EffectiveSpread =
                    # sum(log(P_t^txn / P_t) * Dir_t^LR * V_t * P_t^txn) /
                    # sum(V_t * P_t^txn).
                    eff_num = np.nansum(np.log(np.where(mid[w] > 0, px[w] / mid[w], np.nan)) * direction[w] * dollar)
                    effective_spread[i] = eff_num / denom

            data[f"breadth__{label}"] = breadth
            data[f"immediacy__{label}"] = immediacy
            data[f"volume_all__{label}"] = volume_all
            data[f"volume_avg__{label}"] = volume_avg
            data[f"volume_max__{label}"] = volume_max
            data[f"lambda__{label}"] = mid_lambda
            data[f"lob_imbalance__{label}"] = lob_imbalance
            data[f"txn_imbalance__{label}"] = txn_imbalance
            data[f"past_return__{label}"] = past_return
            data[f"turnover__{label}"] = turnover
            data[f"autocov__{label}"] = mid_autocov
            data[f"quoted_spread__{label}"] = quoted_spread
            data[f"effective_spread__{label}"] = effective_spread
            data[f"no_quote__{label}"] = no_quote_flag
            data[f"no_trade__{label}"] = no_trade_flag

            data[f"window_{label}_breadth"] = breadth
            data[f"window_{label}_immediacy"] = immediacy
            data[f"window_{label}_volume_all"] = volume_all
            data[f"window_{label}_volume_avg"] = volume_avg
            data[f"window_{label}_volume_max"] = volume_max
            data[f"window_{label}_mid_lambda"] = mid_lambda
            data[f"window_{label}_lob_imbalance"] = lob_imbalance
            data[f"window_{label}_txn_imbalance"] = txn_imbalance
            data[f"window_{label}_past_return"] = past_return
            data[f"window_{label}_turnover"] = turnover
            data[f"window_{label}_mid_autocov"] = mid_autocov
            data[f"window_{label}_quoted_spread_mean"] = quoted_spread
            data[f"window_{label}_effective_spread"] = effective_spread
            data[f"window_{label}_no_quote_flag"] = no_quote_flag
            data[f"window_{label}_no_trade_flag"] = no_trade_flag

        frame = pd.DataFrame(data)
        frame = frame.loc[valid_future].copy()
        total_valid += len(frame)
        if not frame.empty:
            results.append(frame)

    out = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    out = ensure_aligned_feature_columns(out) if not out.empty else out
    print(f"Horizon {horizon_seconds}s valid label count: {total_valid:,}", flush=True)
    return out


def build_features_labels_for_horizon(cleaned_df, horizon_seconds, trade_date, min_quotes_per_contract=100, top_n_contracts=50, label_mode="mid_avg", trades_df=None):
    if cleaned_df.empty:
        print(f"Horizon {horizon_seconds}s valid label count: 0", flush=True)
        return pd.DataFrame()
    if trades_df is None or trades_df.empty:
        print(f"Horizon {horizon_seconds}s valid label count: 0 | trade file required in refactored pipeline", flush=True)
        return pd.DataFrame()
    aligned = merge_trades_with_quotes_asof(cleaned_df, trades_df, trade_date)
    if aligned.empty:
        print(f"Horizon {horizon_seconds}s valid label count: 0 | no trade-anchor matches", flush=True)
        return pd.DataFrame()
    return build_trade_aligned_features_labels_for_horizon(
        aligned,
        horizon_seconds,
        trade_date,
        min_quotes_per_contract=min_quotes_per_contract,
        top_n_contracts=top_n_contracts,
        label_mode=label_mode,
    )


def build_horizon_chunk(clean_chunk_path, horizon_seconds, trade_date, out_path, min_quotes_per_contract=100, top_n_contracts=50, label_mode="mid_avg", trade_parquet_path=None):
    print(f"Chunk horizon build start: {clean_chunk_path.name} horizon={horizon_seconds}s", flush=True)
    df = pd.read_parquet(clean_chunk_path)
    df = df.sort_values(["option_ticker", "timestamp_ns"]).reset_index(drop=True)
    trades_df = pd.read_parquet(trade_parquet_path) if trade_parquet_path else None
    out = build_features_labels_for_horizon(
        df,
        horizon_seconds,
        trade_date,
        min_quotes_per_contract=min_quotes_per_contract,
        top_n_contracts=top_n_contracts,
        label_mode=label_mode,
        trades_df=trades_df,
    )
    if out.empty:
        return None, 0
    out = ensure_aligned_feature_columns(out)
    out.to_parquet(out_path, index=False)
    print(f"Chunk horizon build done: {out_path.name} rows={len(out):,}", flush=True)
    return str(out_path), len(out)


def build_feature_chunk_files(clean_chunk_paths, trade_date, out_dir, n_workers, selected_horizons=None, min_quotes_per_contract=100, top_n_contracts=50, label_mode="mid_avg", trade_parquet_path=None):
    horizon_dir = out_dir / "horizon_chunks"
    horizon_dir.mkdir(parents=True, exist_ok=True)
    horizons_to_build = selected_horizons or [5, 10, 30, 60]
    horizon_files = {h: [] for h in horizons_to_build}
    max_workers = max(1, min(n_workers, len(horizons_to_build)))

    for idx, clean_path in enumerate(clean_chunk_paths):
        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for horizon in horizons_to_build:
                out_path = horizon_dir / f"h{horizon}_chunk_{idx:03d}.parquet"
                futures[
                    ex.submit(
                        build_horizon_chunk,
                        clean_path,
                        horizon,
                        trade_date,
                        out_path,
                        min_quotes_per_contract,
                        top_n_contracts,
                        label_mode,
                        trade_parquet_path,
                    )
                ] = (horizon, out_path)
            for future in as_completed(futures):
                horizon, _ = futures[future]
                path_str, rows = future.result()
                if path_str and rows > 0:
                    horizon_files[horizon].append(path_str)
    for horizon in horizons_to_build:
        print(f"Horizon {horizon}s chunk files: {len(horizon_files[horizon])}", flush=True)
    return horizon_files
