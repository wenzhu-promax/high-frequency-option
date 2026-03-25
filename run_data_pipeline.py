#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全量实验：dataset CSV → parquet（按需）→ 多标的 × 交易所过滤 × 多档 moneyness × TopN ×
多 horizon × 多模型；单组训练超参（无网格）。汇总 CSV。
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

from config import (
    SP100, SPANS, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SPOT_CSV, DEFAULT_GROUP_COLS,
    TRAIN_MINUTES, REFIT_EVERY_MINUTES, DTE_MIN, DTE_MAX,
    FILTER_SAME_EXCHANGE, MONEYNESS_LIST, TOP_N_CONTRACTS, HORIZON_LIST,
    MIN_VALID_Y, MIN_TRAIN_ROWS, MIN_SAMPLES_UNDERLYING,
)
from data_processing import (
    load_trade, filter_quotes_sp100, merge_trades_quotes, filter_rth,
    clean_merged_option_data, deduplicate_by_ticker_trade_ts,
    add_y_5s_return, fix_dt_et_from_trade_ts,
)
from feature_engineering import add_trade_direction_proxy, build_calendar_predictors, get_predictor_cols
from model_training import rolling_train_predict, get_model_pipelines, evaluate_predictions

AGG: Dict[str, str] = {
    "conditions": "first", "correction": "first", "exchange": "first",
    "price": "median", "size": "sum", "dt": "first", "date": "first",
    "underlying": "first", "expiry": "first", "cp_flag": "first", "strike": "first",
    "dt_utc": "first", "dt_et": "first", "dt_ct": "first", "trade_dt": "first",
    "ask_exchange": "first", "ask_price": "median", "ask_size": "sum",
    "bid_exchange": "first", "bid_price": "median", "bid_size": "sum",
    "sequence_number": "first", "quote_ts": "first", "quote_dt": "first",
}


def _ensure_parquets(
    data_dir: Path, out_dir: Path, trade_csv: str, quote_csv: str,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tp, qp = out_dir / "tradeSP100.parquet", out_dir / "quotesSP100.parquet"
    if not tp.exists():
        tpath = data_dir / trade_csv
        if not tpath.exists():
            raise FileNotFoundError(f"缺少 trade 文件: {tpath}")
        load_trade(tpath, underlying_filter=SP100).to_parquet(tp)
    if not qp.exists():
        qpath = data_dir / quote_csv
        if not qpath.exists():
            raise FileNotFoundError(f"缺少 quote 文件: {qpath}")
        filter_quotes_sp100(qpath, qp, codes=SP100)
    return tp, qp


def _clean(merged: pd.DataFrame) -> Optional[pd.DataFrame]:
    m = fix_dt_et_from_trade_ts(merged)
    if FILTER_SAME_EXCHANGE and "ask_exchange" in m.columns:
        m = m[m["ask_exchange"] == m["bid_exchange"]]
    m = m.dropna()
    rth = filter_rth(m)
    cleaned, _ = clean_merged_option_data(rth)
    if len(cleaned) == 0:
        return None
    return deduplicate_by_ticker_trade_ts(cleaned, AGG)


def _spot(df: pd.DataFrame, u: str, spots: dict) -> Optional[float]:
    s = spots.get(u)
    if s is not None and np.isfinite(s):
        return float(s)
    near = df[(df["dte"] >= DTE_MIN) & (df["dte"] <= DTE_MAX)]
    if len(near) == 0:
        return None
    return float(np.average(near["strike"], weights=np.maximum(near["size"], 1e-9)))


def run_experiment(
    trade_parquet: Path,
    quote_parquet: Path,
    out_dir: Path,
    spots: dict,
    tickers: List[str],
    run_models: bool,
) -> pd.DataFrame:
    trade = pd.read_parquet(trade_parquet)
    out_rows: List[dict] = []
    con = duckdb.connect()
    models = get_model_pipelines() if run_models else {}

    for u in tickers:
        t_u = trade[trade["underlying"] == u]
        if len(t_u) < MIN_SAMPLES_UNDERLYING:
            continue
        q_u = con.execute(
            "SELECT * FROM read_parquet(?) WHERE ticker LIKE ?",
            [str(quote_parquet), f"O:{u}%"],
        ).fetchdf()
        if len(q_u) == 0:
            continue
        c2 = _clean(merge_trades_quotes(t_u, q_u))
        if c2 is None or len(c2) == 0:
            continue
        d0 = c2.copy()
        d0["date"] = pd.to_datetime(d0["date"])
        d0["expiry"] = pd.to_datetime(d0["expiry"])
        d0["dte"] = (d0["expiry"] - d0["date"]).dt.days
        sp = _spot(d0, u, spots)
        if sp is None:
            continue
        d0["abs_moneyness"] = (d0["strike"] / sp - 1).abs()

        for mo in MONEYNESS_LIST:
            c = d0[(d0["dte"] >= DTE_MIN) & (d0["dte"] <= DTE_MAX) & (d0["abs_moneyness"] <= mo)]
            vol = c.groupby("ticker")["size"].sum().sort_values(ascending=False)
            top = vol.head(min(TOP_N_CONTRACTS, len(vol))).index
            c = c[c["ticker"].isin(top)]
            if len(c) < MIN_SAMPLES_UNDERLYING:
                continue
            cand = c.copy()
            cand["trade_dt_et"] = (
                pd.to_datetime(cand["trade_dt"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
            )

            for h in HORIZON_LIST:
                df_w = add_y_5s_return(cand, group_cols=DEFAULT_GROUP_COLS, horizon_seconds=h)
                df1 = add_trade_direction_proxy(df_w, group_cols=DEFAULT_GROUP_COLS)
                feat = build_calendar_predictors(df1, spans=SPANS, group_cols=DEFAULT_GROUP_COLS)
                if feat["y_5s_ret"].notna().sum() < MIN_VALID_Y:
                    continue
                fcols = [x for x in get_predictor_cols(feat) if x in feat.columns and not feat[x].isna().all()]
                if not fcols:
                    continue
                if not run_models:
                    out_rows.append({"underlying": u, "moneyness": mo, "horizon": h, "n_feat_rows": len(feat)})
                    continue
                for name, pipe in models.items():
                    pred, _ = rolling_train_predict(
                        feat, pipe, y_col="y_5s_ret", time_col="trade_dt_et",
                        feature_cols=fcols, group_cols=("date",),
                        train_minutes=TRAIN_MINUTES, refit_every_minutes=REFIT_EVERY_MINUTES,
                        min_train_rows=MIN_TRAIN_ROWS,
                    )
                    ev = evaluate_predictions(pred)
                    if ev.empty:
                        continue
                    row = ev.iloc[0].to_dict()
                    row.update({"underlying": u, "moneyness": mo, "horizon": h, "model": name})
                    out_rows.append(row)

    con.close()
    df_out = pd.DataFrame(out_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary_experiment.csv"
    df_out.to_csv(summary_path, index=False)
    print(f"写入 {summary_path} 行数 {len(df_out)}")
    return df_out


def main() -> None:
    p = argparse.ArgumentParser(description="全量实验主流程")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--trade-csv", type=str, default="trades_2025-12-01.csv.gz")
    p.add_argument("--quote-csv", type=str, default="quotes_2025-12-01.csv.gz")
    p.add_argument("--no-prepare", action="store_true", help="不生成 parquet，需已有 trade/quote parquet")
    p.add_argument("--trade-parquet", type=Path, default=None)
    p.add_argument("--quote-parquet", type=Path, default=None)
    p.add_argument(
        "--spot-csv",
        type=Path,
        default=None,
        help=f"现货 CSV（Ticker + 价格列）；省略且存在 {DEFAULT_SPOT_CSV.name} 时自动使用",
    )
    p.add_argument("--spot-price-col", type=str, default="Close_Price_2025-12-01")
    p.add_argument("--tickers", type=str, default=None, help="逗号分隔，默认 SP100 全部")
    p.add_argument("--no-models", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    if args.trade_parquet and args.quote_parquet:
        tp, qp = args.trade_parquet, args.quote_parquet
    elif not args.no_prepare:
        tp, qp = _ensure_parquets(Path(args.data_dir), out_dir, args.trade_csv, args.quote_csv)
    else:
        tp = out_dir / "tradeSP100.parquet"
        qp = out_dir / "quotesSP100.parquet"

    spot_path = args.spot_csv
    if spot_path is None and DEFAULT_SPOT_CSV.exists():
        spot_path = DEFAULT_SPOT_CSV
    spots: dict = {}
    if spot_path and spot_path.exists():
        sc = pd.read_csv(spot_path)
        spots = dict(zip(sc["Ticker"], sc[args.spot_price_col].astype(float)))

    tickers = [x.strip() for x in args.tickers.split(",")] if args.tickers else SP100
    run_experiment(tp, qp, out_dir, spots, tickers, run_models=not args.no_models)


if __name__ == "__main__":
    main()
