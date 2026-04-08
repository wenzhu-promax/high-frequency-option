# -*- coding: utf-8 -*-
"""
Carry-forward 实验共用：load_d0、特征网格迭代、rolling_train_predict_carryforward。
由 ``run_carryforward_experiment.py`` 唯一入口调用。

旧版单体脚本（内联重复逻辑）见 ``carryforward/backup/run_carryforward_experiment_monolithic_legacy.py``（仅备份）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_HF_ROOT = Path(__file__).resolve().parent.parent
if str(_HF_ROOT) not in sys.path:
    sys.path.insert(0, str(_HF_ROOT))

from typing import Iterator, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from config import (
    DEFAULT_GROUP_COLS,
    DTE_MAX,
    DTE_MIN,
    MIN_SAMPLES_UNDERLYING,
    TOP_N_CONTRACTS,
)
from data_processing import add_y_5s_return, merge_trades_quotes
from feature_engineering import add_trade_direction_proxy, build_calendar_predictors, get_predictor_cols

from run_data_pipeline import _clean, _spot


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
) -> Tuple[pd.DataFrame, List[str], dict]:
    """滚动 + carry-forward 训练预测（滚动步内 train 不足时用上一成功 fit 的模型预测）。

    调用方: run_carryforward_experiment._process_one_underlying_cf。
    """
    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data = data.dropna(subset=[time_col, y_col]).copy()
    if feature_cols is None:
        feature_cols = get_predictor_cols(data)  # feature_engineering.get_predictor_cols
    feature_cols = [c for c in feature_cols if c in data.columns and not data[c].isna().all()]
    data = data[np.isfinite(pd.to_numeric(data[y_col], errors="coerce"))].copy()
    data = data.sort_values(list(group_cols) + [time_col]).reset_index(drop=False).rename(
        columns={"index": "orig_index"}
    )

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

        # 与 model_training._rolling_train_predict_core 同结构；区别：train 不足时可 carry-forward
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


def load_d0_for_cf(
    u: str,
    trade: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    quote_parquet_path: str,
    spots: dict,
    filter_same_exchange: bool,
) -> Optional[pd.DataFrame]:
    """等价于原版 _process_one_underlying_cf 中从 t_u 到 d0（含 abs_moneyness）的片段；失败返回 None。

    调用方: run_carryforward_experiment._process_one_underlying_cf。
    """
    t_u = trade[trade["underlying"] == u]
    if len(t_u) < MIN_SAMPLES_UNDERLYING:
        return None
    q_u = con.execute(
        "SELECT * FROM read_parquet(?) WHERE ticker LIKE ?",
        [quote_parquet_path, f"O:{u}%"],
    ).fetchdf()
    if len(q_u) == 0:
        return None
    c2 = _clean(  # run_data_pipeline._clean；merge_trades_quotes → data_processing
        merge_trades_quotes(t_u, q_u),
        filter_same_exchange=filter_same_exchange,
    )
    if c2 is None or len(c2) == 0:
        return None
    d0 = c2.copy()
    d0["date"] = pd.to_datetime(d0["date"])
    d0["expiry"] = pd.to_datetime(d0["expiry"])
    d0["dte"] = (d0["expiry"] - d0["date"]).dt.days
    sp = _spot(d0, u, spots)  # run_data_pipeline._spot
    if sp is None:
        return None
    d0["abs_moneyness"] = (d0["strike"] / sp - 1).abs()
    return d0


def iter_mo_horizon_features(
    d0: pd.DataFrame,
    moneyness_list: List[float],
    horizon_list: List[int],
    spans: List[Tuple[float, float]],
    clock_type: str,
    min_valid_y: int,
) -> Iterator[Tuple[float, int, pd.DataFrame, List[str]]]:
    """等价于原版 mo/h 嵌套循环中构造 feat、fcols 并 continue 的规则；有效时 yield 一组。

    调用方: run_carryforward_experiment._process_one_underlying_cf。
    """
    for mo in moneyness_list:
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

        for h in horizon_list:
            df_w = add_y_5s_return(  # data_processing
                cand, group_cols=DEFAULT_GROUP_COLS, horizon_seconds=h,
            )
            df1 = add_trade_direction_proxy(df_w, group_cols=DEFAULT_GROUP_COLS)  # feature_engineering
            feat = build_calendar_predictors(  # feature_engineering
                df1,
                spans=spans,
                group_cols=DEFAULT_GROUP_COLS,
                clock_type=clock_type,
            )
            if feat["y_5s_ret"].notna().sum() < min_valid_y:
                continue
            fcols = [
                x
                for x in get_predictor_cols(feat)  # feature_engineering
                if x in feat.columns and not feat[x].isna().all()
            ]
            if not fcols:
                continue
            yield mo, h, feat, fcols


def write_by_ticker_csv(out_dir: Path, u: str, rows: List[dict]) -> None:
    """与 run_data_pipeline._write_ticker_csv 相同。

    调用方: run_carryforward_experiment.run_carryforward_experiment。
    """
    out_dir = Path(out_dir)
    sub = out_dir / "by_ticker"
    sub.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(sub / f"{u}.csv", index=False)
