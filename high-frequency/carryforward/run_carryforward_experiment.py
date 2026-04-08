#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Carry-forward 实验入口：单标的管线在 carryforward_experiment_common（load_d0、特征网格、rolling_train_predict_carryforward），
与主线数据/特征一致，仅滚动评估为 carry-forward。

旧版单体脚本（含内联 rolling_train_predict_carryforward）已移至备份：
  carryforward/backup/run_carryforward_experiment_monolithic_legacy.py

请使用独立输出目录，工作目录为 ``high-frequency/``，例如：
  python carryforward/run_carryforward_experiment.py --output-dir output_carryforward --max-tickers 3
"""

from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_HF_ROOT = Path(__file__).resolve().parent.parent
if str(_HF_ROOT) not in sys.path:
    sys.path.insert(0, str(_HF_ROOT))
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from carryforward.carryforward_experiment_common import (
    iter_mo_horizon_features,
    load_d0_for_cf,
    rolling_train_predict_carryforward,
    write_by_ticker_csv,
)
from config import (
    SP100,
    SPANS,
    SPANS_CALENDAR_EXT,
    SPANS_TRANSACTION,
    SPANS_VOLUME,
    DEFAULT_DATA_DIR,
    DEFAULT_TRADE_PARQUET,
    DEFAULT_QUOTE_PARQUET,
    DEFAULT_SPOT_CSV,
    LOCAL_SPOT_CSV,
    DEFAULT_CLOCK_TYPE,
    TRAIN_MINUTES,
    REFIT_EVERY_MINUTES,
    FILTER_SAME_EXCHANGE,
    MONEYNESS_LIST,
    HORIZON_LIST,
    MIN_VALID_Y,
    MIN_TRAIN_ROWS,
    DEFAULT_EXPERIMENT_WORKERS,
)
from model_training import evaluate_predictions, get_model_pipelines
from run_data_pipeline import _read_trade_for_tickers

_W_POOL_TRADE: Optional[pd.DataFrame] = None
_W_POOL_QUOTE: Optional[str] = None
_W_POOL_SPOTS: dict = {}
_W_POOL_RUN_MODELS: bool = True
_W_POOL_SHOW_PROGRESS: bool = True
_W_POOL_MONEYNESS: List[float] = list(MONEYNESS_LIST)
_W_POOL_HORIZONS: List[int] = list(HORIZON_LIST)
_W_POOL_TRAIN_MINUTES: int = TRAIN_MINUTES
_W_POOL_REFIT_EVERY: int = REFIT_EVERY_MINUTES
_W_POOL_MIN_TRAIN_ROWS: int = MIN_TRAIN_ROWS
_W_POOL_MIN_VALID_Y: int = MIN_VALID_Y
_W_POOL_FILTER_SAME_EXCHANGE: bool = FILTER_SAME_EXCHANGE
_W_POOL_SPANS = SPANS
_W_POOL_CLOCK_TYPE: str = DEFAULT_CLOCK_TYPE


def _pool_worker_init_cf(
    trade_path: str,
    quote_path: str,
    spots: dict,
    run_models: bool,
    show_progress: bool,
    tickers: List[str],
    moneyness_list: List[float],
    horizon_list: List[int],
    train_minutes: int,
    refit_every_minutes: int,
    min_train_rows: int,
    min_valid_y: int,
    filter_same_exchange: bool,
    spans: List[Tuple[float, float]],
    clock_type: str,
) -> None:
    global _W_POOL_TRADE, _W_POOL_QUOTE, _W_POOL_SPOTS, _W_POOL_RUN_MODELS, _W_POOL_SHOW_PROGRESS
    global _W_POOL_MONEYNESS, _W_POOL_HORIZONS
    global _W_POOL_TRAIN_MINUTES, _W_POOL_REFIT_EVERY, _W_POOL_MIN_TRAIN_ROWS, _W_POOL_MIN_VALID_Y
    global _W_POOL_FILTER_SAME_EXCHANGE, _W_POOL_SPANS, _W_POOL_CLOCK_TYPE
    _W_POOL_TRADE = _read_trade_for_tickers(trade_path, tickers)
    _W_POOL_QUOTE = quote_path
    _W_POOL_SPOTS = spots
    _W_POOL_RUN_MODELS = run_models
    _W_POOL_SHOW_PROGRESS = show_progress
    _W_POOL_MONEYNESS = moneyness_list
    _W_POOL_HORIZONS = horizon_list
    _W_POOL_TRAIN_MINUTES = train_minutes
    _W_POOL_REFIT_EVERY = refit_every_minutes
    _W_POOL_MIN_TRAIN_ROWS = min_train_rows
    _W_POOL_MIN_VALID_Y = min_valid_y
    _W_POOL_FILTER_SAME_EXCHANGE = filter_same_exchange
    _W_POOL_SPANS = spans
    _W_POOL_CLOCK_TYPE = clock_type


def _process_one_underlying_cf(u: str) -> List[dict]:
    """单标的：load_d0 + 特征网格迭代 + carry-forward 评估。

    调用方: _safe_process_one_cf（本文件）。

    项目内依赖: load_d0_for_cf / iter_mo_horizon_features / write_by_ticker_csv → carryforward_experiment_common；
    get_model_pipelines / evaluate_predictions → model_training；rolling_train_predict_carryforward →
    carryforward_experiment_common；_read_trade_for_tickers → run_data_pipeline。
    """
    trade = _W_POOL_TRADE
    qp = _W_POOL_QUOTE
    spots = _W_POOL_SPOTS
    run_models = _W_POOL_RUN_MODELS
    show_progress = _W_POOL_SHOW_PROGRESS
    if trade is None or qp is None:
        return []

    out_rows: List[dict] = []
    con = duckdb.connect()
    try:
        models = get_model_pipelines() if run_models else {}  # model_training
        d0 = load_d0_for_cf(  # carryforward_experiment_common.load_d0_for_cf
            u, trade, con, qp, spots, _W_POOL_FILTER_SAME_EXCHANGE,
        )
        if d0 is None:
            return out_rows

        for mo, h, feat, fcols in iter_mo_horizon_features(  # carryforward_experiment_common
            d0,
            _W_POOL_MONEYNESS,
            _W_POOL_HORIZONS,
            _W_POOL_SPANS,
            _W_POOL_CLOCK_TYPE,
            _W_POOL_MIN_VALID_Y,
        ):
            if not run_models:
                out_rows.append({"underlying": u, "moneyness": mo, "horizon": h, "n_feat_rows": len(feat)})
                continue
            m_iter = (
                tqdm(models.items(), desc=f"{u} mo={mo}", leave=False, unit="model")
                if show_progress
                else models.items()
            )
            for name, pipe in m_iter:
                pred, _, cf_stats = rolling_train_predict_carryforward(  # carryforward_experiment_common
                    feat,
                    pipe,
                    y_col="y_5s_ret",
                    time_col="trade_dt_et",
                    feature_cols=fcols,
                    group_cols=("date",),
                    train_minutes=_W_POOL_TRAIN_MINUTES,
                    refit_every_minutes=_W_POOL_REFIT_EVERY,
                    min_train_rows=_W_POOL_MIN_TRAIN_ROWS,
                )
                ev = evaluate_predictions(pred)  # model_training.evaluate_predictions
                if ev.empty:
                    continue
                row = ev.iloc[0].to_dict()
                row.update({"underlying": u, "moneyness": mo, "horizon": h, "model": name})
                row.update(cf_stats)
                out_rows.append(row)

        return out_rows
    finally:
        con.close()


def _safe_process_one_cf(u: str) -> List[dict]:
    try:
        return _process_one_underlying_cf(u)
    except Exception as e:
        return [{"underlying": u, "error": str(e)}]


def run_carryforward_experiment(
    trade_parquet: Path,
    quote_parquet: Path,
    out_dir: Path,
    spots: dict,
    tickers: List[str],
    run_models: bool,
    show_progress: bool = True,
    workers: int = DEFAULT_EXPERIMENT_WORKERS,
    moneyness_list: Optional[List[float]] = None,
    horizon_list: Optional[List[int]] = None,
    train_minutes: Optional[int] = None,
    refit_every_minutes: Optional[int] = None,
    min_train_rows: Optional[int] = None,
    min_valid_y: Optional[int] = None,
    filter_same_exchange: Optional[bool] = None,
    spans: Optional[List[Tuple[float, float]]] = None,
    clock_type: Optional[str] = None,
) -> pd.DataFrame:
    out_rows: List[dict] = []
    tp_s, qp_s = str(trade_parquet), str(quote_parquet)
    mo_list = moneyness_list if moneyness_list is not None else list(MONEYNESS_LIST)
    hz_list = horizon_list if horizon_list is not None else list(HORIZON_LIST)
    tm = train_minutes if train_minutes is not None else TRAIN_MINUTES
    re = refit_every_minutes if refit_every_minutes is not None else REFIT_EVERY_MINUTES
    mtr = min_train_rows if min_train_rows is not None else MIN_TRAIN_ROWS
    mvy = min_valid_y if min_valid_y is not None else MIN_VALID_Y
    fex = filter_same_exchange if filter_same_exchange is not None else FILTER_SAME_EXCHANGE
    sp = spans if spans is not None else SPANS
    ct = clock_type if clock_type is not None else DEFAULT_CLOCK_TYPE

    if workers > 1:
        initargs = (
            tp_s, qp_s, spots, run_models, show_progress, tickers, mo_list, hz_list,
            tm, re, mtr, mvy, fex, sp, ct,
        )
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_pool_worker_init_cf,
            initargs=initargs,
        ) as ex:
            it = ex.map(_safe_process_one_cf, tickers)
            if show_progress:
                it = tqdm(it, total=len(tickers), desc="标的", unit="ticker")
            for u, rows in zip(tickers, it):
                write_by_ticker_csv(out_dir, u, rows)
                out_rows.extend(rows)
    else:
        global _W_POOL_TRADE, _W_POOL_QUOTE, _W_POOL_SPOTS, _W_POOL_RUN_MODELS, _W_POOL_SHOW_PROGRESS
        global _W_POOL_MONEYNESS, _W_POOL_HORIZONS
        global _W_POOL_TRAIN_MINUTES, _W_POOL_REFIT_EVERY, _W_POOL_MIN_TRAIN_ROWS, _W_POOL_MIN_VALID_Y
        global _W_POOL_FILTER_SAME_EXCHANGE, _W_POOL_SPANS, _W_POOL_CLOCK_TYPE
        _W_POOL_TRADE = _read_trade_for_tickers(tp_s, tickers)
        _W_POOL_QUOTE = qp_s
        _W_POOL_SPOTS = spots
        _W_POOL_RUN_MODELS = run_models
        _W_POOL_SHOW_PROGRESS = show_progress
        _W_POOL_MONEYNESS = mo_list
        _W_POOL_HORIZONS = hz_list
        _W_POOL_TRAIN_MINUTES = tm
        _W_POOL_REFIT_EVERY = re
        _W_POOL_MIN_TRAIN_ROWS = mtr
        _W_POOL_MIN_VALID_Y = mvy
        _W_POOL_FILTER_SAME_EXCHANGE = fex
        _W_POOL_SPANS = sp
        _W_POOL_CLOCK_TYPE = ct
        try:
            t_iter = tqdm(tickers, desc="标的", unit="ticker") if show_progress else tickers
            for u in t_iter:
                rows = _safe_process_one_cf(u)
                write_by_ticker_csv(out_dir, u, rows)
                out_rows.extend(rows)
        finally:
            _W_POOL_TRADE = None

    df_out = pd.DataFrame(out_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary_experiment.csv"
    df_out.to_csv(summary_path, index=False)
    print(f"[carryforward] 写入 {summary_path} 行数 {len(df_out)}")
    return df_out


def run_synthetic_selftest() -> None:
    """调用 carryforward_experiment_common.rolling_train_predict_carryforward 的合成数据自检。"""
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(42)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", Ridge(alpha=1.0)),
    ])
    day = pd.Timestamp("2025-01-02", tz="America/New_York")
    rows: List[dict] = []
    t0 = day + pd.Timedelta(hours=9, minutes=30)
    for i in range(210):
        ts = t0 + pd.Timedelta(seconds=10 * i)
        rows.append({
            "date": day.date(),
            "trade_dt_et": ts,
            "y_5s_ret": float(rng.standard_normal()),
            "f1": float(rng.standard_normal()),
        })
    t1 = day + pd.Timedelta(hours=10, minutes=35)
    for i in range(5):
        ts = t1 + pd.Timedelta(seconds=10 * i)
        rows.append({
            "date": day.date(),
            "trade_dt_et": ts,
            "y_5s_ret": float(rng.standard_normal()),
            "f1": float(rng.standard_normal()),
        })
    df = pd.DataFrame(rows)
    pred, _fcols, stats = rolling_train_predict_carryforward(
        df,
        pipe,
        y_col="y_5s_ret",
        time_col="trade_dt_et",
        feature_cols=["f1"],
        group_cols=("date",),
        train_minutes=30,
        refit_every_minutes=5,
        min_train_rows=50,
    )
    if len(pred) == 0:
        raise RuntimeError("synthetic selftest: no predictions produced")
    if stats.get("n_fit_windows", 0) < 1:
        raise RuntimeError("synthetic selftest: expected at least one fit window")
    print("[carryforward smoke/synthetic] OK | pred_rows=", len(pred), "| stats:", stats)


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    p = argparse.ArgumentParser(
        description="Carry-forward：共用 carryforward_experiment_common；与主线数据/特征一致",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--trade-csv", type=str, default="trades_2025-12-01.csv.gz")
    p.add_argument("--quote-csv", type=str, default="quotes_2025-12-01.csv.gz")
    p.add_argument("--no-prepare", action="store_true")
    p.add_argument("--trade-parquet", type=Path, default=None)
    p.add_argument("--quote-parquet", type=Path, default=None)
    p.add_argument("--spot-csv", type=Path, default=None)
    p.add_argument("--spot-price-col", type=str, default="Close_Price_2025-12-01")
    p.add_argument("--tickers", type=str, default=None)
    p.add_argument("--no-models", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--workers", type=int, default=DEFAULT_EXPERIMENT_WORKERS, metavar="N")
    p.add_argument("--max-tickers", type=int, default=None, metavar="N")
    p.add_argument("--horizons", type=str, default=None)
    p.add_argument("--moneyness", type=str, default=None)
    p.add_argument("--train-minutes", type=int, default=None, metavar="M")
    p.add_argument("--refit-every-minutes", type=int, default=None, metavar="M")
    p.add_argument("--min-train-rows", type=int, default=None, metavar="N")
    p.add_argument("--min-valid-y", type=int, default=None, metavar="N")
    p.add_argument("--no-exchange-filter", action="store_true")
    p.add_argument("--clock-type", type=str, default=None, choices=["calendar", "transaction", "volume"])
    p.add_argument(
        "--spans-type",
        type=str,
        default=None,
        choices=["default", "calendar_ext", "transaction", "volume"],
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-synthetic-only", action="store_true")

    args = p.parse_args()

    if args.smoke_synthetic_only:
        run_synthetic_selftest()
        return

    from run_data_pipeline import _ensure_parquets

    if args.smoke:
        run_synthetic_selftest()
        if args.max_tickers is None:
            args.max_tickers = 1
        if not args.horizons or not str(args.horizons).strip():
            args.horizons = "5"
        if not args.moneyness or not str(args.moneyness).strip():
            args.moneyness = "0.10"
        args.workers = 1
        args.no_progress = True
        if args.train_minutes is None:
            args.train_minutes = 15
        if args.refit_every_minutes is None:
            args.refit_every_minutes = 5
        if args.min_train_rows is None:
            args.min_train_rows = 50
        if args.output_dir is None:
            args.output_dir = Path(__file__).resolve().parent / "output_carryforward_smoke"

    if args.output_dir is None:
        out_dir = Path(__file__).resolve().parent / "output_carryforward"
    else:
        out_dir = Path(args.output_dir).resolve()

    if args.trade_parquet and args.quote_parquet:
        tp, qp = args.trade_parquet, args.quote_parquet
    elif not args.no_prepare:
        if DEFAULT_TRADE_PARQUET.exists() and DEFAULT_QUOTE_PARQUET.exists():
            tp, qp = DEFAULT_TRADE_PARQUET, DEFAULT_QUOTE_PARQUET
        else:
            tp, qp = _ensure_parquets(Path(args.data_dir), out_dir, args.trade_csv, args.quote_csv)
    else:
        tp = out_dir / "tradeSP100.parquet"
        qp = out_dir / "quotesSP100.parquet"

    spot_path = args.spot_csv
    if spot_path is None:
        if DEFAULT_SPOT_CSV.exists():
            spot_path = DEFAULT_SPOT_CSV
        elif LOCAL_SPOT_CSV.exists():
            spot_path = LOCAL_SPOT_CSV
    spots: dict = {}
    if spot_path and Path(spot_path).exists():
        sc = pd.read_csv(spot_path)
        spots = dict(zip(sc["Ticker"], sc[args.spot_price_col].astype(float)))

    tickers = [x.strip() for x in args.tickers.split(",")] if args.tickers else list(SP100)
    if args.max_tickers is not None and args.max_tickers > 0:
        tickers = tickers[: args.max_tickers]

    horizon_override: Optional[List[int]] = None
    if args.horizons and str(args.horizons).strip():
        horizon_override = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    moneyness_override: Optional[List[float]] = None
    if args.moneyness and str(args.moneyness).strip():
        moneyness_override = [float(x.strip()) for x in args.moneyness.split(",") if x.strip()]

    spans_override = None
    spans_type = args.spans_type or "default"
    if spans_type == "calendar_ext":
        spans_override = SPANS_CALENDAR_EXT
    elif spans_type == "transaction":
        spans_override = SPANS_TRANSACTION
    elif spans_type == "volume":
        spans_override = SPANS_VOLUME

    clock_type_override = args.clock_type
    if clock_type_override is None:
        if spans_type == "transaction":
            clock_type_override = "transaction"
        elif spans_type == "volume":
            clock_type_override = "volume"

    fex_cli: Optional[bool] = False if args.no_exchange_filter else None
    filter_on = False if args.no_exchange_filter else FILTER_SAME_EXCHANGE
    resolved_horizons = horizon_override if horizon_override is not None else list(HORIZON_LIST)
    resolved_moneyness = moneyness_override if moneyness_override is not None else list(MONEYNESS_LIST)
    resolved_min_train_rows = args.min_train_rows if args.min_train_rows is not None else MIN_TRAIN_ROWS
    resolved_min_valid_y = args.min_valid_y if args.min_valid_y is not None else MIN_VALID_Y
    resolved_clock_type = clock_type_override if clock_type_override is not None else DEFAULT_CLOCK_TYPE

    print(f"[carryforward] 输出目录: {out_dir} | ask/bid 交易所一致过滤: {filter_on}")
    print(
        "[carryforward] 实验设定: "
        f"clock_type={resolved_clock_type}, spans_type={spans_type}, "
        f"horizons={resolved_horizons}, moneyness={resolved_moneyness}, "
        f"min_train_rows={resolved_min_train_rows}, min_valid_y={resolved_min_valid_y}"
    )

    run_carryforward_experiment(
        tp,
        qp,
        out_dir,
        spots,
        tickers,
        run_models=not args.no_models,
        show_progress=not args.no_progress,
        workers=max(1, args.workers),
        moneyness_list=moneyness_override,
        horizon_list=horizon_override,
        train_minutes=args.train_minutes,
        refit_every_minutes=args.refit_every_minutes,
        min_train_rows=args.min_train_rows,
        min_valid_y=args.min_valid_y,
        filter_same_exchange=fex_cli,
        spans=spans_override,
        clock_type=clock_type_override,
    )


if __name__ == "__main__":
    main()
