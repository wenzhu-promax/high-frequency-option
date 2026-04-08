#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
正式实验主流程：dataset CSV → parquet（按需）→ SP100 × 交易所过滤 × 多档 moneyness × TopN ×
多 horizon × 多模型；单组训练超参（无网格）。结果：每个标的即时写入 output/by_ticker/{TICKER}.csv，最后汇总 output/summary_experiment.csv。

网格与路径以 config 为准；可选 --tickers / --max-tickers / --horizons / --moneyness 做子集或覆盖；
--train-minutes / --refit-every-minutes / --min-train-rows / --min-valid-y 覆盖滚动训练；
--workers 按标的并行；trade 按本次标的列表谓词加载。
--no-exchange-filter 关闭 ask/bid 交易所一致过滤；未指定 --output-dir 时默认写入 output_no_exchange/，避免与 output/ 混用。
"""

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

from config import (
    SP100, SPANS, SPANS_CALENDAR_EXT, SPANS_TRANSACTION, SPANS_VOLUME, DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_DIR_NO_EXCHANGE, DEFAULT_TRADE_PARQUET, DEFAULT_QUOTE_PARQUET,
    DEFAULT_SPOT_CSV, LOCAL_SPOT_CSV, DEFAULT_GROUP_COLS, DEFAULT_CLOCK_TYPE,
    TRAIN_MINUTES, REFIT_EVERY_MINUTES, DTE_MIN, DTE_MAX,
    FILTER_SAME_EXCHANGE, MONEYNESS_LIST, TOP_N_CONTRACTS, HORIZON_LIST,
    MIN_VALID_Y, MIN_TRAIN_ROWS, MIN_SAMPLES_UNDERLYING,
    DEFAULT_EXPERIMENT_WORKERS,
)
from data_processing import (
    load_trade, filter_quotes_sp100, merge_trades_quotes, filter_rth,
    clean_merged_option_data, deduplicate_by_ticker_trade_ts,
    add_y_5s_return, fix_dt_et_from_trade_ts,
)
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from feature_engineering import add_trade_direction_proxy, build_calendar_predictors, get_predictor_cols
from model_training import (
    evaluate_predictions,
    get_model_pipelines,
    rolling_train_predict,
    rolling_train_predict_with_explain,
)

# 定义于本文件；传入 data_processing.deduplicate_by_ticker_trade_ts 的聚合规则
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
    """若 output 目录下尚无 trade/quote parquet，则从 dataset 中 CSV 生成并返回路径。

    调用方: main（本文件）。
    """
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


def _clean(merged: pd.DataFrame, filter_same_exchange: bool = True) -> Optional[pd.DataFrame]:
    """合并后数据：修正美东时间、可选买卖盘交易所一致、去缺失、RTH、清洗、按 ticker+trade_ts 去重。

    调用方: _process_one_underlying（本文件）。
    """
    m = fix_dt_et_from_trade_ts(merged)
    if filter_same_exchange and "ask_exchange" in m.columns:
        m = m[m["ask_exchange"] == m["bid_exchange"]]
    m = m.dropna()
    rth = filter_rth(m)
    cleaned, _ = clean_merged_option_data(rth)
    if len(cleaned) == 0:
        return None
    return deduplicate_by_ticker_trade_ts(cleaned, AGG)


def _spot(df: pd.DataFrame, u: str, spots: dict) -> Optional[float]:
    """标的 u 的现货价：优先 spots 字典；否则用 dte 窗口内成交量加权行权价近似。

    调用方: _process_one_underlying（本文件）。
    """
    s = spots.get(u)
    if s is not None and np.isfinite(s):
        return float(s)
    near = df[(df["dte"] >= DTE_MIN) & (df["dte"] <= DTE_MAX)]
    if len(near) == 0:
        return None
    return float(np.average(near["strike"], weights=np.maximum(near["size"], 1e-9)))


def _read_trade_for_tickers(trade_path: str, tickers: List[str]) -> pd.DataFrame:
    """按本次标的列表加载 trade（谓词下推；失败则全表读入后 isin 过滤）。

    调用方: _pool_worker_init、run_experiment 单进程路径（本文件）；run_carryforward_experiment、
    run_carryforward_experiment_v2（自各模块导入）。
    """
    if not tickers:
        return pd.read_parquet(trade_path)
    try:
        return pd.read_parquet(trade_path, filters=[("underlying", "in", tickers)])
    except Exception:
        df = pd.read_parquet(trade_path)
        if "underlying" in df.columns:
            return df[df["underlying"].isin(tickers)]
        return df


@dataclass
class WorkerConfig:
    """子进程内单份实验网格配置（由 run_experiment 构造，经 _pool_worker_init 注入）。"""

    moneyness_list: List[float]
    horizon_list: List[int]
    train_minutes: int
    refit_every_minutes: int
    min_train_rows: int
    min_valid_y: int
    filter_same_exchange: bool
    spans: List[Tuple[float, float]]
    clock_type: str
    explain: bool = False
    shap_max_samples: int = 256

    # 调用方: run_experiment（本文件）构造并传入 _pool_worker_init


# workers>1 时每个子进程 init 各持有一份（已按 tickers 过滤的）trade，内存约 workers×该子表大小
_W_POOL_TRADE: Optional[pd.DataFrame] = None
_W_POOL_QUOTE: Optional[str] = None
_W_POOL_SPOTS: dict = {}
_W_POOL_RUN_MODELS: bool = True
_W_POOL_SHOW_PROGRESS: bool = True
_W_POOL_CFG = WorkerConfig(
    moneyness_list=list(MONEYNESS_LIST),
    horizon_list=list(HORIZON_LIST),
    train_minutes=TRAIN_MINUTES,
    refit_every_minutes=REFIT_EVERY_MINUTES,
    min_train_rows=MIN_TRAIN_ROWS,
    min_valid_y=MIN_VALID_Y,
    filter_same_exchange=FILTER_SAME_EXCHANGE,
    spans=list(SPANS),
    clock_type=DEFAULT_CLOCK_TYPE,
)


def _pool_worker_init(
    trade_path: str,
    quote_path: str,
    spots: dict,
    run_models: bool,
    show_progress: bool,
    tickers: List[str],
    cfg: WorkerConfig,
) -> None:
    """多进程池 initializer：每个子进程加载过滤后的 trade、路径与实验配置到模块全局变量。

    调用方: run_experiment 内 ProcessPoolExecutor（本文件）。
    """
    global _W_POOL_TRADE, _W_POOL_QUOTE, _W_POOL_SPOTS, _W_POOL_RUN_MODELS, _W_POOL_SHOW_PROGRESS, _W_POOL_CFG
    _W_POOL_TRADE = _read_trade_for_tickers(trade_path, tickers)
    _W_POOL_QUOTE = quote_path
    _W_POOL_SPOTS = spots
    _W_POOL_RUN_MODELS = run_models
    _W_POOL_SHOW_PROGRESS = show_progress
    _W_POOL_CFG = cfg


def _write_explain_csv(out_dir: Path, explain_parts: List[pd.DataFrame]) -> None:
    """将各标的分段的解释 DataFrame 纵向合并后写入 explain_features.csv。

    调用方: _collect_results（本文件）。
    """
    if not explain_parts:
        return
    ex_all = pd.concat(explain_parts, ignore_index=True)
    ex_path = out_dir / "explain_features.csv"
    ex_all.to_csv(ex_path, index=False)
    print(f"写入 {ex_path} 行数 {len(ex_all)}")


def _collect_results(
    out_dir: Path,
    tickers: List[str],
    result_iter,
) -> List[dict]:
    """遍历各标的 (rows, explain_df)，写 by_ticker CSV、汇总 explain，返回扁平化指标行列表。

    调用方: run_experiment（本文件）。
    """
    out_rows: List[dict] = []
    explain_parts: List[pd.DataFrame] = []
    for u, (rows, ex) in zip(tickers, result_iter):
        _write_ticker_csv(out_dir, u, rows)
        out_rows.extend(rows)
        if not ex.empty:
            explain_parts.append(ex)
    _write_explain_csv(out_dir, explain_parts)
    return out_rows


def _process_one_underlying(u: str) -> Tuple[List[dict], pd.DataFrame]:
    """对单一标的跑 moneyness×horizon×模型 网格：特征、滚动训练/解释、评估指标行。

    调用方: _safe_process_one（本文件）。

    项目内依赖（定义文件）::
      merge_trades_quotes / add_y_5s_return → data_processing.py
      _clean / _spot / _read_trade_for_tickers → 本文件 run_data_pipeline.py
      add_trade_direction_proxy / build_calendar_predictors / get_predictor_cols → feature_engineering.py
      get_model_pipelines / rolling_train_predict / rolling_train_predict_with_explain / evaluate_predictions
      → model_training.py
    """
    trade = _W_POOL_TRADE
    qp = _W_POOL_QUOTE
    spots = _W_POOL_SPOTS
    run_models = _W_POOL_RUN_MODELS
    show_progress = _W_POOL_SHOW_PROGRESS
    cfg = _W_POOL_CFG
    if trade is None or qp is None:
        return [], pd.DataFrame()

    out_rows: List[dict] = []
    explain_parts: List[pd.DataFrame] = []
    con = duckdb.connect()
    try:
        models = get_model_pipelines() if run_models else {}  # model_training.get_model_pipelines

        t_u = trade[trade["underlying"] == u]
        if len(t_u) < MIN_SAMPLES_UNDERLYING:
            return out_rows, pd.DataFrame()
        q_u = con.execute(
            "SELECT * FROM read_parquet(?) WHERE ticker LIKE ?",
            [qp, f"O:{u}%"],
        ).fetchdf()
        if len(q_u) == 0:
            return out_rows, pd.DataFrame()
        c2 = _clean(  # _clean 定义于本文件；merge_trades_quotes 定义于 data_processing
            merge_trades_quotes(t_u, q_u),
            filter_same_exchange=cfg.filter_same_exchange,
        )
        if c2 is None or len(c2) == 0:
            return out_rows, pd.DataFrame()
        d0 = c2.copy()
        d0["date"] = pd.to_datetime(d0["date"])
        d0["expiry"] = pd.to_datetime(d0["expiry"])
        d0["dte"] = (d0["expiry"] - d0["date"]).dt.days
        sp = _spot(d0, u, spots)  # 定义于本文件 _spot
        if sp is None:
            return out_rows, pd.DataFrame()
        d0["abs_moneyness"] = (d0["strike"] / sp - 1).abs()

        for mo in cfg.moneyness_list:
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

            for h in cfg.horizon_list:
                df_w = add_y_5s_return(  # data_processing.add_y_5s_return
                    cand, group_cols=DEFAULT_GROUP_COLS, horizon_seconds=h,
                )
                df1 = add_trade_direction_proxy(df_w, group_cols=DEFAULT_GROUP_COLS)  # feature_engineering
                feat = build_calendar_predictors(  # feature_engineering.build_calendar_predictors
                    df1,
                    spans=cfg.spans,
                    group_cols=DEFAULT_GROUP_COLS,
                    clock_type=cfg.clock_type,
                )
                if feat["y_5s_ret"].notna().sum() < cfg.min_valid_y:
                    continue
                fcols = [
                    x
                    for x in get_predictor_cols(feat)  # feature_engineering.get_predictor_cols
                    if x in feat.columns and not feat[x].isna().all()
                ]
                if not fcols:
                    continue
                if not run_models:
                    out_rows.append({"underlying": u, "moneyness": mo, "horizon": h, "n_feat_rows": len(feat)})
                    continue
                m_iter = (
                    tqdm(models.items(), desc=f"{u} mo={mo}", leave=False, unit="model")
                    if show_progress
                    else models.items()
                )
                for name, pipe in m_iter:
                    if cfg.explain:
                        pred, _, ex = rolling_train_predict_with_explain(  # model_training
                            feat,
                            pipe,
                            y_col="y_5s_ret",
                            time_col="trade_dt_et",
                            feature_cols=fcols,
                            group_cols=("date",),
                            train_minutes=cfg.train_minutes,
                            refit_every_minutes=cfg.refit_every_minutes,
                            min_train_rows=cfg.min_train_rows,
                            shap_max_samples=cfg.shap_max_samples,
                        )
                    else:
                        pred, _ = rolling_train_predict(  # model_training
                            feat,
                            pipe,
                            y_col="y_5s_ret",
                            time_col="trade_dt_et",
                            feature_cols=fcols,
                            group_cols=("date",),
                            train_minutes=cfg.train_minutes,
                            refit_every_minutes=cfg.refit_every_minutes,
                            min_train_rows=cfg.min_train_rows,
                        )
                        ex = None
                    ev = evaluate_predictions(pred)  # model_training.evaluate_predictions
                    if ev.empty:
                        continue
                    row = ev.iloc[0].to_dict()
                    row.update({"underlying": u, "moneyness": mo, "horizon": h, "model": name})
                    out_rows.append(row)
                    if ex is not None and not ex.empty:
                        explain_parts.append(
                            ex.assign(underlying=u, moneyness=mo, horizon=h, model=name)
                        )

        ex_df = pd.concat(explain_parts, ignore_index=True) if explain_parts else pd.DataFrame()
        return out_rows, ex_df
    finally:
        con.close()


def _safe_process_one(u: str) -> Tuple[List[dict], pd.DataFrame]:
    """包装单标的处理，异常时返回带 error 字段的单行记录与空 explain。

    调用方: run_experiment（本文件）。
    """
    try:
        return _process_one_underlying(u)
    except Exception as e:
        return [{"underlying": u, "error": str(e)}], pd.DataFrame()


def _write_ticker_csv(out_dir: Path, u: str, rows: List[dict]) -> None:
    """将单标的所有实验结果行写入 out_dir/by_ticker/{u}.csv。

    调用方: _collect_results（本文件）；carryforward_experiment_common.write_by_ticker_csv 为等价封装。
    """
    sub = out_dir / "by_ticker"
    sub.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(sub / f"{u}.csv", index=False)


def run_experiment(
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
    explain: bool = False,
    shap_max_samples: int = 256,
) -> pd.DataFrame:
    """按标的串行或并行跑完整实验，写 by_ticker 与 summary_experiment.csv，返回汇总 DataFrame。

    调用方: main（本文件）。
    """
    tp_s, qp_s = str(trade_parquet), str(quote_parquet)
    cfg = WorkerConfig(
        moneyness_list=moneyness_list if moneyness_list is not None else list(MONEYNESS_LIST),
        horizon_list=horizon_list if horizon_list is not None else list(HORIZON_LIST),
        train_minutes=train_minutes if train_minutes is not None else TRAIN_MINUTES,
        refit_every_minutes=refit_every_minutes if refit_every_minutes is not None else REFIT_EVERY_MINUTES,
        min_train_rows=min_train_rows if min_train_rows is not None else MIN_TRAIN_ROWS,
        min_valid_y=min_valid_y if min_valid_y is not None else MIN_VALID_Y,
        filter_same_exchange=filter_same_exchange if filter_same_exchange is not None else FILTER_SAME_EXCHANGE,
        spans=spans if spans is not None else list(SPANS),
        clock_type=clock_type if clock_type is not None else DEFAULT_CLOCK_TYPE,
        explain=explain,
        shap_max_samples=shap_max_samples,
    )

    if workers > 1:
        initargs = (tp_s, qp_s, spots, run_models, show_progress, tickers, cfg)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_pool_worker_init,
            initargs=initargs,
        ) as ex:
            it = ex.map(_safe_process_one, tickers)
            if show_progress:
                it = tqdm(it, total=len(tickers), desc="标的", unit="ticker")
            out_rows = _collect_results(out_dir, tickers, it)
    else:
        global _W_POOL_TRADE, _W_POOL_QUOTE, _W_POOL_SPOTS, _W_POOL_RUN_MODELS, _W_POOL_SHOW_PROGRESS, _W_POOL_CFG
        _W_POOL_TRADE = _read_trade_for_tickers(tp_s, tickers)
        _W_POOL_QUOTE = qp_s
        _W_POOL_SPOTS = spots
        _W_POOL_RUN_MODELS = run_models
        _W_POOL_SHOW_PROGRESS = show_progress
        _W_POOL_CFG = cfg
        try:
            t_iter = tqdm(tickers, desc="标的", unit="ticker") if show_progress else tickers
            result_iter = (_safe_process_one(u) for u in t_iter)
            out_rows = _collect_results(out_dir, tickers, result_iter)
        finally:
            _W_POOL_TRADE = None

    df_out = pd.DataFrame(out_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary_experiment.csv"
    df_out.to_csv(summary_path, index=False)
    print(f"写入 {summary_path} 行数 {len(df_out)}")
    return df_out


def main() -> None:
    """解析 CLI：数据路径、标的子集、网格、滚动参数、时钟与 explain，调用 run_experiment。

    调用方: ``if __name__ == "__main__"``（本文件）。
    """
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    p = argparse.ArgumentParser(description="SP100 正式实验：特征 + 滚动训练 + 汇总 CSV")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"结果目录；默认 {DEFAULT_OUTPUT_DIR.name}；与 --no-exchange-filter 且未指定本项时为 {DEFAULT_OUTPUT_DIR_NO_EXCHANGE.name}（避免与有过滤实验混写）",
    )
    p.add_argument("--trade-csv", type=str, default="trades_2025-12-01.csv.gz")
    p.add_argument("--quote-csv", type=str, default="quotes_2025-12-01.csv.gz")
    p.add_argument("--no-prepare", action="store_true", help="不生成 parquet，需已有 trade/quote parquet")
    p.add_argument("--trade-parquet", type=Path, default=None)
    p.add_argument("--quote-parquet", type=Path, default=None)
    p.add_argument(
        "--spot-csv",
        type=Path,
        default=None,
        help="现货 CSV（Ticker + 价格列）；省略时优先 KaidiZhang 目录下默认文件，否则用本仓库内副本",
    )
    p.add_argument("--spot-price-col", type=str, default="Close_Price_2025-12-01")
    p.add_argument("--tickers", type=str, default=None, help="逗号分隔，默认 SP100 全部")
    p.add_argument("--no-models", action="store_true")
    p.add_argument("--no-progress", action="store_true", help="关闭进度条（适合重定向到日志）")
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_EXPERIMENT_WORKERS,
        metavar="N",
        help="并行进程数（按标的）；>1 时每个子进程各加载一份已过滤 trade，内存约 N×子表（默认见 config.DEFAULT_EXPERIMENT_WORKERS）",
    )
    p.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        metavar="N",
        help="仅处理前 N 个标的（在 --tickers 或默认 SP100 顺序上截断）；默认不截断",
    )
    p.add_argument(
        "--horizons",
        type=str,
        default=None,
        help="覆盖 horizon 列表（逗号分隔整数）；省略则使用 config.HORIZON_LIST",
    )
    p.add_argument(
        "--moneyness",
        type=str,
        default=None,
        help="覆盖 moneyness 阈值列表（逗号分隔浮点）；省略则使用 config.MONEYNESS_LIST",
    )
    p.add_argument(
        "--train-minutes",
        type=int,
        default=None,
        metavar="M",
        help=f"滚动训练窗口长度（分钟）；默认 config.TRAIN_MINUTES={TRAIN_MINUTES}",
    )
    p.add_argument(
        "--refit-every-minutes",
        type=int,
        default=None,
        metavar="M",
        help=f"滚动重拟合步长（分钟）；默认 config.REFIT_EVERY_MINUTES={REFIT_EVERY_MINUTES}",
    )
    p.add_argument(
        "--min-train-rows",
        type=int,
        default=None,
        metavar="N",
        help=f"每段训练最少行数；默认 config.MIN_TRAIN_ROWS={MIN_TRAIN_ROWS}",
    )
    p.add_argument(
        "--min-valid-y",
        type=int,
        default=None,
        metavar="N",
        help=f"有效 y 个数下限（少于此则跳过该 horizon）；默认 config.MIN_VALID_Y={MIN_VALID_Y}",
    )
    p.add_argument(
        "--no-exchange-filter",
        action="store_true",
        help="不做 ask/bid 交易所一致过滤；未指定 --output-dir 时默认写入 output_no_exchange/",
    )
    p.add_argument(
        "--clock-type",
        type=str,
        default=None,
        choices=["calendar", "transaction", "volume"],
        help="特征工程时钟类型：calendar(时间窗口), transaction(交易笔数), volume(成交量)；默认 calendar",
    )
    p.add_argument(
        "--spans-type",
        type=str,
        default=None,
        choices=["default", "calendar_ext", "transaction", "volume"],
        help="窗口配置类型：default(原SPANS), calendar_ext(扩展时间), transaction(交易笔数), volume(成交量)；默认 default",
    )
    p.add_argument("--explain", action="store_true", help="保存末次滚动窗的 coef / importance / SHAP 到 explain_features.csv")
    p.add_argument("--shap-max-samples", type=int, default=256, metavar="N", help="SHAP 用测试集前 N 行")
    args = p.parse_args()

    if args.output_dir is not None:
        out_dir = Path(args.output_dir).resolve()
    elif args.no_exchange_filter:
        out_dir = DEFAULT_OUTPUT_DIR_NO_EXCHANGE
    else:
        out_dir = DEFAULT_OUTPUT_DIR
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
    if spot_path and spot_path.exists():
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

    # 选择 spans 配置
    spans_override = None
    spans_type = args.spans_type or "default"
    if spans_type == "calendar_ext":
        spans_override = SPANS_CALENDAR_EXT
    elif spans_type == "transaction":
        spans_override = SPANS_TRANSACTION
    elif spans_type == "volume":
        spans_override = SPANS_VOLUME

    # 如果 clock-type 未指定但 spans-type 暗示了，自动对齐
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
    print(f"输出目录: {out_dir} | ask/bid 交易所一致过滤: {filter_on}")
    print(
        "实验设定: "
        f"clock_type={resolved_clock_type}, spans_type={spans_type}, "
        f"horizons={resolved_horizons}, moneyness={resolved_moneyness}, "
        f"min_train_rows={resolved_min_train_rows}, min_valid_y={resolved_min_valid_y}"
    )

    run_experiment(
        tp, qp, out_dir, spots, tickers,
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
        explain=args.explain,
        shap_max_samples=args.shap_max_samples,
    )


if __name__ == "__main__":
    main()
