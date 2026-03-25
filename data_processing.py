# -*- coding: utf-8 -*-
"""数据加载与处理：trade/quote 读取、合并、清洗、标签构造"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import duckdb
import numpy as np
import pandas as pd

from config import SP100


def parse_option_ticker(sym: str) -> pd.Series:
    """
    解析期权 ticker。
    格式: O:AAPL251219C00150000 或 AAPL251219C00150000
    正则: ([A-Z]+) 标的, (\\d{6}) 到期YYMMDD, ([CP]) 认购/认沽, (\\d{8}) 行权价×1000
    返回: [underlying, expiry, cp_flag, strike]，strike 已除以 1000
    """
    s = sym
    if s.startswith("O:"):
        s = s[2:]
    m = re.match(r"^([A-Z]+)(\d{6})([CP])(\d{8})$", s)
    if m is None:
        return pd.Series([None, None, None, None])
    underlying, yymmdd, cp, strike_raw = m.groups()
    strike = int(strike_raw) / 1000
    expiry = pd.to_datetime(yymmdd, format="%y%m%d")
    return pd.Series([underlying, expiry, cp, strike])


def load_trade(
    path: Union[str, Path],
    underlying_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    加载 trade CSV。
    构造变量:
      dt: sip_timestamp 转 datetime（纳秒）
      date: dt 的日期部分
      underlying, expiry, cp_flag, strike: 由 parse_option_ticker 解析
      dt_utc, dt_et, dt_ct: 由 add_trade_dt_cols 添加
    """
    trade = pd.read_csv(path)
    trade["dt"] = pd.to_datetime(trade["sip_timestamp"])  # 纳秒时间戳
    trade["date"] = trade["dt"].dt.date
    trade[["underlying", "expiry", "cp_flag", "strike"]] = trade["ticker"].apply(parse_option_ticker)
    trade = add_trade_dt_cols(trade)
    if underlying_filter is not None:
        trade = trade[trade.underlying.isin(underlying_filter)].copy()
    return trade


def add_trade_dt_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 sip_timestamp（纳秒）构造时区列:
      dt_utc: UTC 时间
      dt_et: 美东 (America/New_York)
      dt_ct: 美中 (America/Chicago)，期权交易时段
    """
    out = df.copy()
    out["dt_utc"] = pd.to_datetime(out["sip_timestamp"], unit="ns", utc=True)
    out["dt_et"] = out["dt_utc"].dt.tz_convert("America/New_York")
    out["dt_ct"] = out["dt_utc"].dt.tz_convert("America/Chicago")
    return out


def fix_dt_et_from_trade_ts(df: pd.DataFrame) -> pd.DataFrame:
    """merge 后用 trade_ts 重建 dt_et（纽约），避免 parquet 时区错误导致 RTH 为空。"""
    out = df.copy()
    if "trade_ts" in out.columns:
        out["dt_et"] = pd.to_datetime(out["trade_ts"], unit="ns", utc=True).dt.tz_convert("America/New_York")
    return out


def load_quote_csv(path: Union[str, Path], nrows: Optional[int] = None) -> pd.DataFrame:
    """加载 quote CSV（可选限制行数）"""
    if nrows is not None:
        return pd.read_csv(path, nrows=nrows)
    return pd.read_csv(path)


def filter_quotes_sp100(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    codes: Optional[List[str]] = None,
    threads: int = 8,
    memory_limit: str = "32GB",
) -> None:
    """DuckDB 过滤 gzip CSV，只保留 SP100 成分股期权 quote，输出 parquet"""
    if codes is None:
        codes = SP100
    pattern = "^O:(" + "|".join(codes) + ")[0-9]{6}[CP]"
    input_path, output_path = str(input_path), str(output_path)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads};")
    con.execute(f"PRAGMA memory_limit='{memory_limit}';")
    query = f"""
    COPY (
        SELECT ticker, ask_exchange, ask_price, ask_size, bid_exchange, bid_price,
               bid_size, sequence_number, sip_timestamp
        FROM read_csv_auto('{input_path}', compression='gzip', header=true)
        WHERE regexp_matches(ticker, '{pattern}')
    ) TO '{output_path}' (FORMAT PARQUET);
    """
    con.execute(query)
    con.close()


def merge_trades_quotes(
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    trade_ts_col: str = "sip_timestamp",
    quote_ts_col: str = "sip_timestamp",
) -> pd.DataFrame:
    """merge_asof：每笔 trade 对齐到其发生前的最近一条 quote"""
    t = trades.copy()
    q = quotes.copy()
    t = t.rename(columns={trade_ts_col: "trade_ts"})
    q = q.rename(columns={quote_ts_col: "quote_ts"})
    t["ticker"] = t["ticker"].astype(str)
    q["ticker"] = q["ticker"].astype(str)
    t = t.dropna(subset=["trade_ts", "ticker"]).copy()
    q = q.dropna(subset=["quote_ts", "ticker"]).copy()
    t["trade_ts"] = t["trade_ts"].astype("int64")
    q["quote_ts"] = q["quote_ts"].astype("int64")
    t["trade_dt"] = pd.to_datetime(t["trade_ts"])
    q["quote_dt"] = pd.to_datetime(q["quote_ts"])
    t = t.sort_values(["trade_ts", "ticker"]).reset_index(drop=True)
    q = q.sort_values(["quote_ts", "ticker"]).reset_index(drop=True)
    # direction='backward': 每笔 trade 取发生时刻之前最近的 quote
    return pd.merge_asof(
        t, q, left_on="trade_ts", right_on="quote_ts", by="ticker",
        direction="backward", allow_exact_matches=True,
    )


def filter_rth(
    df: pd.DataFrame,
    dt_col: str = "dt_et",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
) -> pd.DataFrame:
    """只保留 RTH 时段内数据，默认 09:30–16:00 ET"""
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])
    return df[df[dt_col].dt.time.between(
        pd.to_datetime(start_time).time(),
        pd.to_datetime(end_time).time(),
    )].copy()


def clean_merged_option_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    清洗 merged 数据，剔除异常记录。
    构造变量:
      spread = ask_price - bid_price
      mid = (ask_price + bid_price) / 2  买卖中间价
      rolling_med_mid: 按 ticker 滚动 50 条的中心化中位数，用于检测 mid 离群
      abs_dev_mid = |mid - rolling_med_mid|
      rolling_mean_abs_dev_mid: abs_dev_mid 的滚动均值，离群阈值 = 10× 该值
    异常规则见下方 cond_* 变量
    """
    x = df.copy()
    time_col = "trade_ts"
    x = x.sort_values(["ticker", "date", time_col]).reset_index(drop=True)
    num_cols = ["price", "size", "bid_price", "ask_price", "bid_size", "ask_size"]
    for c in num_cols:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x["spread"] = x["ask_price"] - x["bid_price"]
    x["mid"] = (x["ask_price"] + x["bid_price"]) / 2

    # 异常规则：满足任一条即剔除
    cond_zero_price_or_size = (
        (x["price"] <= 0) | (x["size"] <= 0) | (x["bid_price"] <= 0) | (x["ask_price"] <= 0)
        | (x["bid_size"] <= 0) | (x["ask_size"] <= 0)
    )
    cond_negative_spread = x["spread"] < 0  # 买卖价倒挂
    cond_small_both_sizes = (x["bid_size"] <= 1) & (x["ask_size"] <= 1)  # 双边深度过薄
    day_median_spread = x.groupby(["ticker", "date"])["spread"].transform("median")
    cond_spread_too_large = x["spread"] > 50 * day_median_spread  # 单条 spread 远超当日中位数

    # mid 离群：|mid - 滚动中位数| > 10 × 滚动 MAD
    x["rolling_med_mid"] = (
        x.groupby("ticker", group_keys=False)["mid"]
        .apply(lambda s: s.rolling(window=50, center=True, min_periods=25).median())
    )
    x["abs_dev_mid"] = (x["mid"] - x["rolling_med_mid"]).abs()
    x["rolling_mean_abs_dev_mid"] = (
        x.groupby("ticker", group_keys=False)["abs_dev_mid"]
        .apply(lambda s: s.rolling(window=50, center=True, min_periods=25).mean())
    )
    cond_mid_outlier = (
        x["rolling_mean_abs_dev_mid"].notna()
        & (x["abs_dev_mid"] > 10 * x["rolling_mean_abs_dev_mid"])
    )
    cond_ratio_too_large = (x["bid_price"] != 0) & ((x["ask_price"] / x["bid_price"]) > 5)  # ask/bid 比异常
    # 成交价超出 [bid-spread, ask+spread] 合理范围
    cond_trade_price_outside = (
        (x["price"] < (x["bid_price"] - x["spread"]))
        | (x["price"] > (x["ask_price"] + x["spread"]))
    )
    delete_mask = (
        cond_zero_price_or_size | cond_negative_spread | cond_small_both_sizes
        | cond_spread_too_large | cond_mid_outlier | cond_ratio_too_large | cond_trade_price_outside
    )

    summary = pd.DataFrame({
        "rule": [
            "zero_or_nonpositive_price_or_size", "negative_spread", "both_bid_ask_size_le_1",
            "spread_gt_50x_daily_median", "mid_gt_10x_rolling_mean_abs_dev_from_centered_median",
            "ask_bid_ratio_gt_5", "trade_price_outside_bid_minus_spread__ask_plus_spread", "any_rule",
        ],
        "count": [
            int(cond_zero_price_or_size.sum()), int(cond_negative_spread.sum()),
            int(cond_small_both_sizes.sum()), int(cond_spread_too_large.sum()),
            int(cond_mid_outlier.sum()), int(cond_ratio_too_large.sum()),
            int(cond_trade_price_outside.sum()), int(delete_mask.sum()),
        ],
    })

    cleaned = x.loc[~delete_mask].copy()
    return cleaned, summary


def deduplicate_by_ticker_trade_ts(df: pd.DataFrame, agg_dict: dict) -> pd.DataFrame:
    """
    同一 (ticker, trade_ts) 可能有多条（不同 quote_ts 对齐），按 agg_dict 聚合。
    聚合后重新计算: spread = ask_price - bid_price, mid = (ask+bid)/2
    """
    out = (
        df.sort_values(["ticker", "trade_ts", "quote_ts"])
        .groupby(["ticker", "trade_ts"], as_index=False)
        .agg(agg_dict)
    )
    out["spread"] = out["ask_price"] - out["bid_price"]
    out["mid"] = (out["ask_price"] + out["bid_price"]) / 2
    return out


def add_y_5s_return(
    df: pd.DataFrame,
    horizon_seconds: int = 5,
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    构造 y 标签：未来 horizon_seconds 内成交均价相对当前 mid 的收益率。
    变量构造:
      future_avg_trade_price_5s: 窗口 (T, T+horizon] 内所有 trade 的 price 的成交量加权平均
        （此处简化为等权平均：sum(px)/cnt，因单合约内 size 差异不大）
      n_future_trades_5s: 窗口内 trade 条数
      y_5s_ret = future_avg_trade_price_5s / mid - 1
    按 group_cols 分组，每组内按 trade_ts 排序后逐行计算。
    """
    out = df.copy()
    if group_cols is None:
        group_cols = ["ticker", "date", "underlying", "expiry", "cp_flag", "strike"]
    out["trade_ts"] = pd.to_numeric(out["trade_ts"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["mid"] = pd.to_numeric(out["mid"], errors="coerce")
    out = out.dropna(subset=["trade_ts", "price", "mid"] + group_cols).copy()
    out = out.sort_values(group_cols + ["trade_ts", "sequence_number"]).reset_index(drop=True)
    horizon_ns = int(horizon_seconds * 1e9)

    def _one_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["trade_ts", "sequence_number"]).copy()
        ts = g["trade_ts"].to_numpy(dtype=np.int64)
        px = g["price"].to_numpy(dtype=float)
        mid = g["mid"].to_numpy(dtype=float)
        # 对每行 i，窗口 (ts[i], ts[i]+horizon_ns] 内的 trade 索引
        # searchsorted(ts, x, side="right") 返回第一个 ts[j] > x 的下标
        # start_idx[i] = 第一个 > ts[i] 的位置, end_idx[i] = 第一个 > ts[i]+horizon_ns 的位置
        start_idx = np.searchsorted(ts, ts, side="right")
        end_idx = np.searchsorted(ts, ts + horizon_ns, side="right")
        # csum[k] = sum(px[0:k])，故 fut_sum = csum[end]-csum[start] = sum(px[start:end])
        csum = np.concatenate([[0.0], np.cumsum(px)])
        cnt = end_idx - start_idx
        fut_sum = csum[end_idx] - csum[start_idx]
        fut_avg = np.full(len(g), np.nan, dtype=float)
        ok = cnt > 0
        fut_avg[ok] = fut_sum[ok] / cnt[ok]
        y = np.full(len(g), np.nan, dtype=float)
        ok2 = ok & np.isfinite(mid) & (mid > 0)
        y[ok2] = fut_avg[ok2] / mid[ok2] - 1.0
        g["future_avg_trade_price_5s"] = fut_avg
        g["n_future_trades_5s"] = cnt
        g["y_5s_ret"] = y
        return g

    out = out.groupby(group_cols, group_keys=False).apply(_one_group)
    return out
