# -*- coding: utf-8 -*-
"""特征工程：交易方向、日历预测变量、predictor 列提取"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import SPANS, DEFAULT_GROUP_COLS


def get_predictor_cols(df: pd.DataFrame) -> List[str]:
    """按列名正则匹配 build_calendar_predictors 生成的 predictor 列。

    调用方: model_training._rolling_train_predict_core、run_data_pipeline._process_one_underlying、
    carryforward_experiment_common（rolling_train_predict_carryforward 与 iter_mo_horizon_features 路径）、
    run_carryforward_experiment（本地 rolling_train_predict_carryforward 与 _process_one_underlying_cf）。
    """
    patt = re.compile(
        r"^(breadth|immediacy|volume_all|volume_avg|volume_max|lambda_|"
        r"lob_imbalance|txn_imbalance|past_return|turnover|autocov|"
        r"quoted_spread|effective_spread)__"
    )
    cols = [c for c in df.columns if patt.match(c)]
    return sorted(cols)


def _span_label(left_sec: float, right_sec: float) -> str:
    """生成 span 标签，如 0_5s。小数用 p 代替点，如 0p5_1s。

    调用方: build_calendar_predictors（本模块）。
    """
    def fmt(x):
        """将秒数格式化为列名安全字符串（小数点替换为 p）。"""
        s = f"{x:g}"
        return s.replace(".", "p")
    return f"{fmt(left_sec)}_{fmt(right_sec)}s"


def _safe_nanmean(x) -> float:
    """忽略 nan/inf 后取均值，空则返回 nan。

    调用方: build_calendar_predictors（本模块）。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.nan if x.size == 0 else float(x.mean())


def _safe_nanmax(x) -> float:
    """忽略 nan/inf 后取最大值，空则返回 nan。

    调用方: build_calendar_predictors（本模块）。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.nan if x.size == 0 else float(x.max())


def _safe_nansum(x) -> float:
    """忽略 nan/inf 后求和，空则返回 0。

    调用方: build_calendar_predictors（本模块）。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return 0.0 if x.size == 0 else float(x.sum())


def add_trade_direction_proxy(
    df: pd.DataFrame,
    price_col: str = "price",
    mid_col: str = "mid",
    out_col: str = "dir_proxy",
    group_cols: Optional[List[str]] = None,
    ts_col: str = "trade_ts",
    seq_col: str = "sequence_number",
) -> pd.DataFrame:
    """Lee-Ready：quote rule 先，tick rule 兜底，输出 +1/-1/0。

    调用方: build_calendar_predictors（本模块）、run_data_pipeline._process_one_underlying、
    carryforward_experiment_common.iter_mo_horizon_features、run_carryforward_experiment._process_one_underlying_cf。
    """
    out = df.copy()
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out[mid_col] = pd.to_numeric(out[mid_col], errors="coerce")
    out[ts_col] = pd.to_numeric(out[ts_col], errors="coerce")
    if group_cols is None:
        group_cols = DEFAULT_GROUP_COLS
    sort_cols = group_cols + [ts_col]
    if seq_col in out.columns:
        sort_cols.append(seq_col)
    out = out.sort_values(sort_cols).copy()

    def _one_group(g: pd.DataFrame) -> pd.DataFrame:
        """单分组内按时间排序，用 quote/tick 规则生成逐笔买卖方向代理 dir_proxy。"""
        g = g.copy()
        px = g[price_col].to_numpy(dtype=float)
        mid = g[mid_col].to_numpy(dtype=float)
        d = np.zeros(len(g), dtype=float)

        # quote rule：price vs mid 判断买卖方向
        # gt: price > mid → 买方发起，dir=+1；lt: price < mid → 卖方发起，dir=-1
        gt = np.isfinite(px) & np.isfinite(mid) & (px > mid)
        lt = np.isfinite(px) & np.isfinite(mid) & (px < mid)
        eq = np.isfinite(px) & np.isfinite(mid) & (px == mid)
        d[gt], d[lt] = 1.0, -1.0

        # tick rule：price==mid 时无法用 quote rule，用价格变动方向
        # price_diff[i] = px[i] - px[i-1]，prepend=np.nan 使 diff[0]=nan
        price_diff = np.diff(px, prepend=np.nan)
        tick_sign = np.sign(price_diff)
        # 零变动沿用上次非零方向（uptick/downtick）
        last_nonzero = 0.0
        for i in range(len(tick_sign)):
            if np.isfinite(tick_sign[i]) and tick_sign[i] != 0:
                last_nonzero = tick_sign[i]
            elif np.isfinite(tick_sign[i]) and tick_sign[i] == 0:
                tick_sign[i] = last_nonzero
        use_tick = eq | (~gt & ~lt & np.isfinite(px))
        d[use_tick] = tick_sign[use_tick]
        d[~np.isfinite(d)] = 0.0
        d[d > 0], d[d < 0] = 1.0, -1.0
        g[out_col] = d.astype(int)
        return g

    return out.groupby(group_cols, group_keys=False).apply(_one_group)


def build_calendar_predictors(
    df: pd.DataFrame,
    spans: Optional[Sequence[Tuple[float, float]]] = None,
    group_cols: Optional[List[str]] = None,
    ts_col: str = "trade_ts",
    price_col: str = "price",
    size_col: str = "size",
    mid_col: str = "mid",
    bid_col: str = "bid_price",
    ask_col: str = "ask_price",
    bid_size_col: str = "bid_size",
    ask_size_col: str = "ask_size",
    trade_dir_col: str = "dir_proxy",
    turnover_base_col: Optional[str] = None,
    seq_col: str = "sequence_number",
    clock_type: str = "calendar",  # "calendar" | "transaction" | "volume"
) -> pd.DataFrame:
    """对每个 anchor 时刻 T，在 lookback (T−right, T−left] 内算 13 类特征。
    支持三种时钟类型：
    - calendar: 时间窗口（秒）
    - transaction: 交易笔数窗口（笔）
    - volume: 成交量窗口（份）
    特征含义：breadth 条数，immediacy 时长/条数，volume_all/avg/max 成交量，
    lambda_ 价格变动/成交量，lob_imbalance 订单簿失衡，txn_imbalance 买卖失衡，
    past_return 过去收益，turnover 成交量/基准，autocov 收益自协方差，
    quoted_spread 报价价差，effective_spread 有效价差。

    调用方: run_data_pipeline._process_one_underlying、carryforward_experiment_common.iter_mo_horizon_features、
    run_carryforward_experiment._process_one_underlying_cf。
    """
    if spans is None:
        spans = SPANS
    out = df.copy()
    if group_cols is None:
        group_cols = DEFAULT_GROUP_COLS

    # 预处理：为 transaction/volume clock 添加索引列
    if clock_type in ("transaction", "volume"):
        # 在每组内计算交易序号和累积成交量
        out["_txn_idx"] = out.groupby(group_cols).cumcount()
        out["_cum_volume"] = out.groupby(group_cols)[size_col].cumsum()

    numeric_cols = [ts_col, price_col, size_col, mid_col, bid_col, ask_col, bid_size_col, ask_size_col]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if trade_dir_col not in out.columns:
        out = add_trade_direction_proxy(  # 定义于本模块 add_trade_direction_proxy
            out, price_col=price_col, mid_col=mid_col, out_col=trade_dir_col,
            group_cols=group_cols, ts_col=ts_col, seq_col=seq_col,
        )

    sort_cols = group_cols + [ts_col]
    if seq_col in out.columns:
        sort_cols.append(seq_col)
    out = out.sort_values(sort_cols).reset_index(drop=True)

    feature_names = [
        "breadth", "immediacy", "volume_all", "volume_avg", "volume_max",
        "lambda_", "lob_imbalance", "txn_imbalance", "past_return",
        "turnover", "autocov",         "quoted_spread", "effective_spread",
    ]

    def _one_group(g: pd.DataFrame) -> pd.DataFrame:
        """单分组内对每个 anchor 时刻，按时钟类型在 lookback 窗口聚合 13 类微观结构特征。"""
        g = g.copy()
        g = g.sort_values([ts_col] + ([seq_col] if seq_col in g.columns else [])).copy()
        n = len(g)
        ts = g[ts_col].to_numpy(dtype=np.int64)
        px = g[price_col].to_numpy(dtype=float)
        sz = g[size_col].to_numpy(dtype=float)
        mid = g[mid_col].to_numpy(dtype=float)
        bid = g[bid_col].to_numpy(dtype=float)
        ask = g[ask_col].to_numpy(dtype=float)
        bid_sz = g[bid_size_col].to_numpy(dtype=float)
        ask_sz = g[ask_size_col].to_numpy(dtype=float)
        direction = g[trade_dir_col].to_numpy(dtype=float)

        # 预计算各特征所需的逐行项（每笔 trade 一行），窗口内聚合时取 mean/sum
        # autocov: 相邻 trade return 的乘积，窗口内取 mean 得 autocov
        # r1[i]=log(px[i]/px[i-1]), r2[i]=log(px[i-1]/px[i-2]), autocov_term[i]=r1*r2
        prev_px = np.roll(px, 1)
        prev2_px = np.roll(px, 2)
        prev_px[0], prev2_px[:2] = np.nan, np.nan
        r1 = np.where((px > 0) & (prev_px > 0), np.log(px / prev_px), np.nan)
        r2 = np.where((prev_px > 0) & (prev2_px > 0), np.log(prev_px / prev2_px), np.nan)
        autocov_term = r1 * r2

        # quoted_spread: (ask-bid)/mid，窗口内取 mean
        quoted_spread_term = np.where(
            (mid > 0) & np.isfinite(ask) & np.isfinite(bid), (ask - bid) / mid, np.nan)

        # lob_imbalance: (ask_size-bid_size)/(ask_size+bid_size)，窗口内取 mean
        lob_denom = ask_sz + bid_sz
        lob_term = np.where((lob_denom > 0), (ask_sz - bid_sz) / lob_denom, np.nan)

        # effective_spread: 分子 log(price/mid)*dir*size*price，分母 size*price，加权平均
        eff_num_term = np.where(
            (px > 0) & (mid > 0) & np.isfinite(direction) & np.isfinite(sz),
            np.log(px / mid) * direction * sz * px, np.nan)
        eff_den_term = np.where((px > 0) & np.isfinite(sz), sz * px, np.nan)

        turnover_base = (
            pd.to_numeric(g[turnover_base_col], errors="coerce").to_numpy(dtype=float)
            if turnover_base_col and turnover_base_col in g.columns
            else np.full(n, np.nan)
        )

        feat_store: Dict[str, np.ndarray] = {}
        # 根据时钟类型选择标签前缀
        clock_prefix = clock_type  # "calendar", "transaction", "volume"
        for left_sec, right_sec in spans:
            lab = f"{clock_prefix}_{_span_label(left_sec, right_sec)}"
            for fn in feature_names:
                key = f"lambda__{lab}" if fn == "lambda_" else f"{fn}__{lab}"
                feat_store[key] = np.full(n, np.nan, dtype=float)

        # 根据时钟类型准备索引数组
        if clock_type == "calendar":
            idx_arr = ts  # 时间戳（纳秒）
            scale = 1e9
        elif clock_type == "transaction":
            idx_arr = g["_txn_idx"].to_numpy(dtype=np.int64)
            scale = 1  # 笔数直接比较
        elif clock_type == "volume":
            idx_arr = g["_cum_volume"].to_numpy(dtype=np.float64)
            scale = 1  # 成交量直接比较
        else:
            raise ValueError(f"Unknown clock_type: {clock_type}")

        for left_span, right_span in spans:
            lab = f"{clock_prefix}_{_span_label(left_span, right_span)}"

            # 根据时钟类型计算窗口索引
            if clock_type == "calendar":
                left_ns, right_ns = int(left_span * scale), int(right_span * scale)
                start_idx = np.searchsorted(idx_arr, idx_arr - right_ns, side="right")
                end_idx = np.searchsorted(idx_arr, idx_arr - left_ns, side="right")
            else:
                # transaction/volume: 直接按数值查找
                start_idx = np.searchsorted(idx_arr, idx_arr - right_span, side="right")
                end_idx = np.searchsorted(idx_arr, idx_arr - left_span, side="right")

            breadth_arr = feat_store[f"breadth__{lab}"]
            immediacy_arr = feat_store[f"immediacy__{lab}"]
            volume_all_arr = feat_store[f"volume_all__{lab}"]
            volume_avg_arr = feat_store[f"volume_avg__{lab}"]
            volume_max_arr = feat_store[f"volume_max__{lab}"]
            lambda_arr = feat_store[f"lambda__{lab}"]
            lob_imbalance_arr = feat_store[f"lob_imbalance__{lab}"]
            txn_imbalance_arr = feat_store[f"txn_imbalance__{lab}"]
            past_return_arr = feat_store[f"past_return__{lab}"]
            turnover_arr = feat_store[f"turnover__{lab}"]
            autocov_arr = feat_store[f"autocov__{lab}"]
            quoted_spread_arr = feat_store[f"quoted_spread__{lab}"]
            effective_spread_arr = feat_store[f"effective_spread__{lab}"]

            for i in range(n):
                a, b = start_idx[i], end_idx[i]
                if b <= a:
                    continue
                w = slice(a, b)

                # breadth: 窗口内 trade 条数
                breadth = b - a
                breadth_arr[i] = breadth

                # immediacy: 窗口时长/条数，秒/笔，越大表示交易越稀疏
                if breadth > 0:
                    immediacy_arr[i] = (right_sec - left_sec) / breadth

                # volume_all: 窗口内 size 之和
                vol_all = _safe_nansum(sz[w])
                volume_all_arr[i] = vol_all

                # volume_avg: volume_all / breadth
                if breadth > 0:
                    volume_avg_arr[i] = vol_all / breadth

                # volume_max: 窗口内 size 最大值
                volume_max_arr[i] = _safe_nanmax(sz[w])

                first_idx, last_idx = a, b - 1
                # lambda_: (mid_last - mid_first) / volume_all，价格变动 per 单位成交量
                if vol_all > 0 and np.isfinite(mid[first_idx]) and np.isfinite(mid[last_idx]):
                    lambda_arr[i] = (mid[last_idx] - mid[first_idx]) / vol_all

                # lob_imbalance: (ask_size-bid_size)/(ask_size+bid_size) 的窗口均值
                lob_imbalance_arr[i] = _safe_nanmean(lob_term[w])

                # txn_imbalance: sum(size*direction)/volume_all，买卖量加权方向
                if vol_all > 0:
                    txn_imbalance_arr[i] = _safe_nansum(sz[w] * direction[w]) / vol_all

                # past_return: 1 - avg_trade_price/mid_last，过去收益的负值
                if np.isfinite(mid[last_idx]) and mid[last_idx] > 0:
                    avg_trade_price = _safe_nanmean(px[w])
                    if np.isfinite(avg_trade_price):
                        past_return_arr[i] = 1.0 - avg_trade_price / mid[last_idx]

                # turnover: volume_all / turnover_base（若提供基准）
                if np.isfinite(turnover_base[i]) and turnover_base[i] > 0:
                    turnover_arr[i] = vol_all / turnover_base[i]

                # autocov: 相邻 trade return 乘积的窗口均值
                autocov_arr[i] = _safe_nanmean(autocov_term[w])

                # quoted_spread: (ask-bid)/mid 的窗口均值
                quoted_spread_arr[i] = _safe_nanmean(quoted_spread_term[w])

                # effective_spread: sum(log(price/mid)*dir*size*price) / sum(size*price)
                eff_num = _safe_nansum(eff_num_term[w])
                eff_den = _safe_nansum(eff_den_term[w])
                if eff_den > 0:
                    effective_spread_arr[i] = eff_num / eff_den

        feat_df = pd.DataFrame(feat_store, index=g.index)
        g = pd.concat([g, feat_df], axis=1)
        return g

    out = out.groupby(group_cols, group_keys=False).apply(_one_group)
    return out


def check_span_occupancy(
    df: pd.DataFrame,
    spans: Optional[Sequence[Tuple[float, float]]] = None,
    group_cols: Optional[List[str]] = None,
    ts_col: str = "trade_ts",
    seq_col: str = "sequence_number",
) -> pd.DataFrame:
    """统计各 span 窗口内非空比例、平均/中位数 trade 数。

    调用方: 本仓库内暂无；可供独立诊断脚本或 Notebook。
    """
    if spans is None:
        spans = SPANS
    if group_cols is None:
        group_cols = DEFAULT_GROUP_COLS
    x = df.copy()
    x[ts_col] = pd.to_numeric(x[ts_col], errors="coerce")
    sort_cols = group_cols + [ts_col]
    if seq_col in x.columns:
        sort_cols.append(seq_col)
    x = x.dropna(subset=[ts_col] + group_cols).sort_values(sort_cols).copy()

    rows = []
    for _, g in x.groupby(group_cols):
        ts = g[ts_col].to_numpy(dtype=np.int64)
        n = len(ts)
        for left_sec, right_sec in spans:
            left_ns = int(left_sec * 1e9)
            right_ns = int(right_sec * 1e9)
            a = np.searchsorted(ts, ts - right_ns, side="right")
            b = np.searchsorted(ts, ts - left_ns, side="right")
            cnt = b - a
            rows.append({
                "span": _span_label(left_sec, right_sec),
                "n_anchor": n,
                "non_empty_ratio": np.mean(cnt > 0),
                "avg_n_trades_in_window": np.mean(cnt),
                "median_n_trades_in_window": np.median(cnt),
            })

    occ = pd.DataFrame(rows)
    if occ.empty:
        return occ
    return (
        occ.groupby("span", as_index=False)
        .agg({"non_empty_ratio": "mean", "avg_n_trades_in_window": "mean", "median_n_trades_in_window": "mean"})
        .sort_values("span")
        .reset_index(drop=True)
    )
