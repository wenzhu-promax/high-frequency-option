#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据预处理主流程：trade/quote -> 特征矩阵 -> 模型评估"""

from pathlib import Path
from typing import Optional, Tuple

import argparse
import pandas as pd

from config import (
    SP100,
    SPANS,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    DTE_MIN,
    DTE_MAX,
    MONEYNESS_THRESHOLD,
    TRAIN_MINUTES,
    REFIT_EVERY_MINUTES,
    DEFAULT_GROUP_COLS,
    DEFAULT_MODEL,
)
from data_processing import (
    load_trade,
    filter_quotes_sp100,
    merge_trades_quotes,
    filter_rth,
    clean_merged_option_data,
    deduplicate_by_ticker_trade_ts,
    add_y_5s_return,
)
from feature_engineering import add_trade_direction_proxy, build_calendar_predictors
from model_training import rolling_train_predict, get_model_pipelines, evaluate_predictions

# 去重聚合时各列聚合方式
_AGG_DICT = {
    "conditions": "first", "correction": "first", "exchange": "first",
    "price": "median", "size": "sum", "dt": "first", "date": "first",
    "underlying": "first", "expiry": "first", "cp_flag": "first", "strike": "first",
    "dt_utc": "first", "dt_et": "first", "dt_ct": "first", "trade_dt": "first",
    "ask_exchange": "first", "ask_price": "median", "ask_size": "sum",
    "bid_exchange": "first", "bid_price": "median", "bid_size": "sum",
    "sequence_number": "first", "quote_ts": "first", "quote_dt": "first",
}


def _stage_load_trade_quote(
    data_dir: Path,
    output_dir: Path,
    trade_csv: str,
    quote_csv: str,
    underlying: str,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """加载 trade、过滤并加载 quote，返回单标的的 trade 与 quote。"""
    trade_path = data_dir / trade_csv
    if not trade_path.exists():
        print(f"Trade 文件不存在: {trade_path}")
        return None
    trade = load_trade(trade_path, underlying_filter=SP100)
    trade.to_parquet(output_dir / "tradeSP100.parquet")

    quote_parquet = output_dir / "quotesSP100.parquet"
    if not quote_parquet.exists():
        quote_path = data_dir / quote_csv
        if quote_path.exists():
            filter_quotes_sp100(quote_path, quote_parquet, codes=SP100)
        else:
            print(f"Quote 文件不存在: {quote_path}")
            return None

    quote = pd.read_parquet(quote_parquet)
    quote["dt_utc"] = pd.to_datetime(quote["sip_timestamp"], unit="ns", utc=True)
    quote["dt_et"] = quote["dt_utc"].dt.tz_convert("America/New_York")
    quote["dt_ct"] = quote["dt_utc"].dt.tz_convert("America/Chicago")

    quote_subset = quote[quote["ticker"].str.startswith(f"O:{underlying}")].copy()
    trade_subset = trade[trade.underlying == underlying].copy()
    return trade_subset, quote_subset


def _stage_merge_and_clean(
    trade_subset: pd.DataFrame,
    quote_subset: pd.DataFrame,
) -> pd.DataFrame:
    """合并 trade/quote，RTH 过滤，清洗，去重聚合。"""
    merged = merge_trades_quotes(trade_subset, quote_subset)
    df_rth = filter_rth(merged.dropna())

    cleaned, summary = clean_merged_option_data(df_rth)
    print("清洗摘要:")
    print(summary)

    cleaned2 = deduplicate_by_ticker_trade_ts(cleaned, _AGG_DICT)
    return cleaned2


def _stage_select_options(cleaned: pd.DataFrame, spot: float) -> pd.DataFrame:
    """DTE、moneyness 筛选，添加时区列。"""
    df = cleaned.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["dte"] = (df["expiry"] - df["date"]).dt.days
    df["abs_moneyness"] = (df["strike"] / spot - 1).abs()
    cand = df[
        (df["dte"] >= DTE_MIN) & (df["dte"] <= DTE_MAX)
        & (df["abs_moneyness"] <= MONEYNESS_THRESHOLD)
    ].copy()

    cand["trade_dt_et"] = (
        pd.to_datetime(cand["trade_dt"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    )
    cand["quote_dt_et"] = (
        pd.to_datetime(cand["quote_dt"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    )
    return cand


def _stage_build_features(cand: pd.DataFrame) -> pd.DataFrame:
    """构造标签与特征。"""
    aapl_y = add_y_5s_return(cand, group_cols=DEFAULT_GROUP_COLS)
    df1 = add_trade_direction_proxy(aapl_y, group_cols=DEFAULT_GROUP_COLS)
    df_feat = build_calendar_predictors(df1, spans=SPANS, group_cols=DEFAULT_GROUP_COLS)
    return df_feat


def _stage_run_models(df_feat: pd.DataFrame, run_models: bool) -> None:
    """滚动训练（Lasso 等模型）并打印评估指标。"""
    if not run_models:
        return
    pipelines = get_model_pipelines()
    pred_df, _ = rolling_train_predict(
        df_feat,
        pipelines[DEFAULT_MODEL],
        y_col="y_5s_ret",
        time_col="trade_dt_et",
        group_cols=("date",),
        train_minutes=TRAIN_MINUTES,
        refit_every_minutes=REFIT_EVERY_MINUTES,
    )
    eval_df = evaluate_predictions(pred_df)
    if not eval_df.empty:
        print(f"{DEFAULT_MODEL} 评估:")
        print(f"  RMSE = {eval_df['rmse'].iloc[0]:.6f}")
        print(f"  R²   = {eval_df['r2'].iloc[0]:.4f}")


def main(
    data_dir: Path,
    output_dir: Path,
    trade_csv: str = "trades_2025-12-01.csv.gz",
    quote_csv: str = "quotes_2025-12-01.csv.gz",
    underlying: str = "AAPL",
    spot: float = 281.10,
    run_models: bool = True,
) -> None:
    """从原始 trade/quote 到特征矩阵与 Lasso 评估的完整流程"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = _stage_load_trade_quote(data_dir, output_dir, trade_csv, quote_csv, underlying)
    if result is None:
        return
    trade_subset, quote_subset = result

    cleaned = _stage_merge_and_clean(trade_subset, quote_subset)
    cand = _stage_select_options(cleaned, spot)
    df_feat = _stage_build_features(cand)

    cand.to_parquet(output_dir / f"{underlying}.parquet")
    df_feat.to_parquet(output_dir / f"{underlying}_features.parquet")
    print(f"特征矩阵已保存: {output_dir / f'{underlying}_features.parquet'}")

    _stage_run_models(df_feat, run_models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据预处理主流程")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-models", action="store_true", help="不跑模型")
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_models=not args.no_models,
    )
