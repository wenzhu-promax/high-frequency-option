#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Carry-forward 滚动实验（与 run_data_pipeline 数据与特征一致，仅评估规则不同）。

与主线的差异
    主线 ``model_training.rolling_train_predict``：某滚动步若训练样本数
    ``len(train_df) < min_train_rows``，该步不产生预测（整段测试窗跳过）。
    本模块 ``rolling_train_predict_carryforward``：同样先尝试用当前窗训练；
    若样本不足但此前在同日内已成功 ``fit`` 过模型，则用上一份已拟合模型
    对当前测试窗做预测（carry-forward）；若从未成功 fit，行为与主线一致（跳过）。

滚动时间轴（每个 ``group_cols`` 分组内，默认按交易日 ``date``）
    对每个 ``refit_time``（从 ``day_start + train_minutes`` 起每隔
    ``refit_every_minutes`` 一格直到日末）：
    - 训练区间 ``[train_start, refit_time)``，长度 ``train_minutes``；
    - 测试区间 ``[refit_time, pred_end)``，长度 ``refit_every_minutes``；
    ``last_model`` 在同组内跨步保留，故 carry-forward 使用的是上一成功拟合的副本
    （``sklearn.base.clone`` 后的 pipeline）。

输出
    - ``{out_dir}/by_ticker/{TICKER}.csv``：该标的下所有 (moneyness, horizon, model) 的指标行；
    - ``{out_dir}/summary_experiment.csv``：全表汇总；每行除 ``rmse``/``r2``/``dir_acc`` 外还包含
      ``n_fit_windows``、``n_carryforward_windows``、``n_pred_rows_*`` 等 carry-forward 统计。
    请使用独立 ``--output-dir``（如 ``output_carryforward``），勿覆盖主线 ``output/``。

并行
    ``workers>1`` 时每个子进程通过 ``_pool_worker_init_cf`` 各加载一份按 tickers 过滤的
    trade，内存约 ``workers × 单份表大小``，与 ``run_data_pipeline`` 相同。

依赖复用
    从 ``run_data_pipeline`` 仅导入 ``AGG``、``_clean``、``_spot``、``_read_trade_for_tickers``、
    （``main`` 内）``_ensure_parquets``，不修改 ``model_training`` / ``run_data_pipeline`` 源码。

示例
    python carryforward/run_carryforward_experiment.py --output-dir output_carryforward --max-tickers 3
    python carryforward/run_carryforward_experiment.py --smoke-synthetic-only   # 仅内存合成，验证逻辑
    python carryforward/run_carryforward_experiment.py --smoke                  # 合成自检 + 最小真实数据跑通
"""

from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from tqdm import tqdm

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
    DEFAULT_GROUP_COLS,
    DEFAULT_CLOCK_TYPE,
    TRAIN_MINUTES,
    REFIT_EVERY_MINUTES,
    DTE_MIN,
    DTE_MAX,
    FILTER_SAME_EXCHANGE,
    MONEYNESS_LIST,
    TOP_N_CONTRACTS,
    HORIZON_LIST,
    MIN_VALID_Y,
    MIN_TRAIN_ROWS,
    MIN_SAMPLES_UNDERLYING,
    DEFAULT_EXPERIMENT_WORKERS,
)
from data_processing import (
    merge_trades_quotes,
    add_y_5s_return,
)
from feature_engineering import add_trade_direction_proxy, build_calendar_predictors, get_predictor_cols
from model_training import evaluate_predictions, get_model_pipelines

from run_data_pipeline import (
    AGG,
    _clean,
    _spot,
    _read_trade_for_tickers,
)


def rolling_train_predict_carryforward(
    df: pd.DataFrame,
    model_pipeline: Pipeline,
    y_col: str = "y_5s_ret",
    time_col: str = "trade_dt_et",
    feature_cols: Optional[List[str]] = None,
    group_cols: Tuple[str, ...] = ("date",),
    train_minutes: int = 30,
    refit_every_minutes: int = 5,
    min_train_rows: int = 200,
) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """按日历滚动训练/测试，训练行不足时用上一成功步的模型对当前测试窗预测。

    与 ``rolling_train_predict`` 相同的预处理：有效时间/y、特征列、按 ``group_cols``+``time_col``
    排序，并保留 ``orig_index`` 以便回溯行。

    每个 ``group_cols`` 分组（通常为单日）内维护 ``last_model``：
    - ``len(train_df) >= min_train_rows``：``clone`` pipeline 后 ``fit``，更新 ``last_model``，
      预测记为 ``fit``；
    - 否则若 ``last_model is not None``：用 ``last_model.predict(X_test)``，预测记为
      ``carryforward``；
    - 否则：跳过该步（与主线一致）。

    Returns
    -------
    pred_out : DataFrame
        含 ``orig_index``、``time_col``、``y_col``、``y_pred``，仅测试窗行（无 ``_pred_method``）。
    feature_cols : list of str
        实际使用的特征列名。
    stats : dict
        ``n_fit_windows`` / ``n_carryforward_windows``：按滚动步计数；
        ``n_pred_rows_fit`` / ``n_pred_rows_carryforward``：按预测行计数（一步可对应多行）。

    调用方: 本文件 _process_one_underlying_cf。逻辑副本见 carryforward_experiment_common.rolling_train_predict_carryforward
    （修改时请两处同步）。
    """
    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data = data.dropna(subset=[time_col, y_col]).copy()
    if feature_cols is None:
        feature_cols = get_predictor_cols(data)  # feature_engineering.get_predictor_cols
    feature_cols = [c for c in feature_cols if c in data.columns and not data[c].isna().all()]
    data = data[np.isfinite(pd.to_numeric(data[y_col], errors="coerce"))].copy()
    data = data.sort_values(list(group_cols) + [time_col]).reset_index(drop=False).rename(columns={"index": "orig_index"})

    all_preds: List[pd.DataFrame] = []
    n_fit_windows = 0
    n_cf_windows = 0

    for _day_key, g in data.groupby(list(group_cols), sort=True):
        g = g.sort_values(time_col).copy()
        day_start, day_end = g[time_col].min(), g[time_col].max()
        first_train_end = day_start + pd.Timedelta(minutes=train_minutes)
        if first_train_end >= day_end:
            continue

        last_model: Optional[Pipeline] = None

        # 与 model_training._rolling_train_predict_core 同网格；此处允许 train 不足时用 last_model 预测
        for refit_time in pd.date_range(start=first_train_end, end=day_end, freq=f"{refit_every_minutes}min"):
            train_start = refit_time - pd.Timedelta(minutes=train_minutes)
            pred_end = refit_time + pd.Timedelta(minutes=refit_every_minutes)
            train_mask = (g[time_col] >= train_start) & (g[time_col] < refit_time)
            test_mask = (g[time_col] >= refit_time) & (g[time_col] < pred_end)
            train_df = g.loc[train_mask].copy()
            test_df = g.loc[test_mask].copy()
            if len(test_df) == 0:
                continue

            X_test = test_df[feature_cols]

            if len(train_df) >= min_train_rows:
                X_train = train_df[feature_cols]
                y_train = train_df[y_col].astype(float).to_numpy()
                model = clone(model_pipeline)
                model.fit(X_train, y_train)
                last_model = model
                y_pred = model.predict(X_test)
                n_fit_windows += 1
                method = "fit"
            elif last_model is not None:
                y_pred = last_model.predict(X_test)
                n_cf_windows += 1
                method = "carryforward"
            else:
                continue

            pred_part = test_df[["orig_index", time_col, y_col]].copy()
            pred_part["y_pred"] = y_pred
            pred_part["_pred_method"] = method
            all_preds.append(pred_part)

    if not all_preds:
        return (
            pd.DataFrame(),
            feature_cols,
            {
                "n_fit_windows": 0,
                "n_carryforward_windows": 0,
                "n_pred_rows_fit": 0,
                "n_pred_rows_carryforward": 0,
            },
        )

    pred_out = pd.concat(all_preds, ignore_index=True)
    n_pred_fit = int((pred_out["_pred_method"] == "fit").sum())
    n_pred_cf = int((pred_out["_pred_method"] == "carryforward").sum())
    pred_out = pred_out.drop(columns=["_pred_method"])

    stats = {
        "n_fit_windows": n_fit_windows,
        "n_carryforward_windows": n_cf_windows,
        "n_pred_rows_fit": n_pred_fit,
        "n_pred_rows_carryforward": n_pred_cf,
    }
    return pred_out, feature_cols, stats


# ---------- 与 run_data_pipeline 相同的 worker 全局量（本模块独立一份） ----------
# 多进程下无法在 worker 间共享大 DataFrame，故 initializer 向各子进程注入一份过滤后的 trade；
# 单进程路径则直接在 run_carryforward_experiment 内赋值，结束时将 _W_POOL_TRADE 置 None 释放。

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
    """ProcessPoolExecutor 子进程入口：加载本批 tickers 的 trade 并写入本模块全局变量。

    与 ``run_data_pipeline._pool_worker_init`` 对应，但额外传入 moneyness/horizon/滚动与特征
    超参（本实验未使用 WorkerConfig 数据类，改为平铺全局变量）。
    """
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
    """处理单个标的 ``u``：merge→清洗→moneyness/TopN→多 horizon 特征→多模型 carry-forward 评估。

    流程与 ``run_data_pipeline._process_one_underlying`` 对齐，区别仅为调用
    ``rolling_train_predict_carryforward`` 并将返回的 ``cf_stats`` 合并进每行结果字典
    （故 summary 中含 ``n_fit_windows`` 等列）。若 ``run_models`` 为 False，仅记录特征行数占位行。

    调用方: _safe_process_one_cf（本文件）。

    项目内依赖: 与 run_data_pipeline._process_one_underlying 相同的数据/特征函数；滚动为
    本文件 ``rolling_train_predict_carryforward``；评估为 ``model_training.evaluate_predictions``。
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

        t_u = trade[trade["underlying"] == u]
        if len(t_u) < MIN_SAMPLES_UNDERLYING:
            return out_rows
        q_u = con.execute(
            "SELECT * FROM read_parquet(?) WHERE ticker LIKE ?",
            [qp, f"O:{u}%"],
        ).fetchdf()
        if len(q_u) == 0:
            return out_rows
        c2 = _clean(  # run_data_pipeline._clean；merge_trades_quotes → data_processing
            merge_trades_quotes(t_u, q_u),
            filter_same_exchange=_W_POOL_FILTER_SAME_EXCHANGE,
        )
        if c2 is None or len(c2) == 0:
            return out_rows
        d0 = c2.copy()
        d0["date"] = pd.to_datetime(d0["date"])
        d0["expiry"] = pd.to_datetime(d0["expiry"])
        d0["dte"] = (d0["expiry"] - d0["date"]).dt.days
        sp = _spot(d0, u, spots)  # run_data_pipeline._spot
        if sp is None:
            return out_rows
        d0["abs_moneyness"] = (d0["strike"] / sp - 1).abs()

        for mo in _W_POOL_MONEYNESS:
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

            for h in _W_POOL_HORIZONS:
                df_w = add_y_5s_return(  # data_processing
                    cand, group_cols=DEFAULT_GROUP_COLS, horizon_seconds=h,
                )
                df1 = add_trade_direction_proxy(df_w, group_cols=DEFAULT_GROUP_COLS)  # feature_engineering
                feat = build_calendar_predictors(  # feature_engineering
                    df1,
                    spans=_W_POOL_SPANS,
                    group_cols=DEFAULT_GROUP_COLS,
                    clock_type=_W_POOL_CLOCK_TYPE,
                )
                if feat["y_5s_ret"].notna().sum() < _W_POOL_MIN_VALID_Y:
                    continue
                fcols = [
                    x
                    for x in get_predictor_cols(feat)  # feature_engineering
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
                    pred, _, cf_stats = rolling_train_predict_carryforward(  # 本文件（与 common 重复维护）
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
                    ev = evaluate_predictions(pred)  # model_training
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
    """调用 ``_process_one_underlying_cf``；异常时返回 ``[{"underlying": u, "error": ...}]`` 避免整池中断。"""
    try:
        return _process_one_underlying_cf(u)
    except Exception as e:
        return [{"underlying": u, "error": str(e)}]


def _write_ticker_csv(out_dir: Path, u: str, rows: List[dict]) -> None:
    """将 ``rows`` 写成 ``out_dir/by_ticker/{u}.csv``（多进程/单进程路径均每标的一次写入）。"""
    sub = out_dir / "by_ticker"
    sub.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(sub / f"{u}.csv", index=False)


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
    """对 ``tickers`` 列表跑完整 carry-forward 实验并落盘。

    参数未传时默认使用 ``config`` 中 ``MONEYNESS_LIST``、``HORIZON_LIST``、``TRAIN_MINUTES`` 等。
    ``workers==1`` 时在本进程设置全局池变量后顺序处理；``workers>1`` 时用进程池并行按标的处理。

    Returns
    -------
    DataFrame
        与 ``summary_experiment.csv`` 内容一致的全部指标行（含 carry-forward 统计列）。
    """
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
                _write_ticker_csv(out_dir, u, rows)
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
                _write_ticker_csv(out_dir, u, rows)
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
    """内存自检：构造单日两段密集成交（中段长间隙），使部分滚动步训练样本 < min_train_rows。

    断言至少产生一条预测且 ``n_fit_windows >= 1``；用于 CI 或本地快速验证
    ``rolling_train_predict_carryforward`` 与 ``--smoke-synthetic-only`` 路径。
    """
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
    # 210×10s = 35min：覆盖到 10:05，使首个测试窗 [10:00,10:05) 内有点（否则 9:59:50 后断档会导致 test_df 为空）
    for i in range(210):
        ts = t0 + pd.Timedelta(seconds=10 * i)
        rows.append({
            "date": day.date(),
            "trade_dt_et": ts,
            "y_5s_ret": float(rng.standard_normal()),
            "f1": float(rng.standard_normal()),
        })
    # 长间隙后再来少量点：后续某窗训练行数不足时可走 carry-forward
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
    print(
        "[smoke/synthetic] OK | pred_rows=",
        len(pred),
        "| stats:",
        stats,
    )


def main() -> None:
    """命令行入口：解析路径与网格、现货、``--no-exchange-filter``、``spans-type``/``clock-type``。

    - ``--smoke-synthetic-only``：只运行 ``run_synthetic_selftest`` 后退出；
    - ``--smoke``：先自检，再收紧为单标的、短滚动、默认写入 ``output_carryforward_smoke/``；
    - 否则解析 parquet/CSV、spot，调用 ``run_carryforward_experiment``。
    """
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    p = argparse.ArgumentParser(
        description="Carry-forward 滚动实验：训练行数不足时用上一段模型预测；结果写入独立目录",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="必填或显式指定；建议 output_carryforward 等，避免覆盖 output/",
    )
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
    p.add_argument(
        "--smoke",
        action="store_true",
        help="快速跑通：先跑合成自检，再 1 标的、horizon=5、moneyness=0.10、单进程、较短滚动窗，输出到 output_carryforward_smoke/",
    )
    p.add_argument(
        "--smoke-synthetic-only",
        action="store_true",
        help="仅运行内存合成自检（不读 parquet），验证 carry-forward 代码路径",
    )

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
