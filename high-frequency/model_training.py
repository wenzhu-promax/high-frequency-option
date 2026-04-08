# -*- coding: utf-8 -*-
"""模型训练与评估：滚动训练、多模型、评估。绘图见 model_plots 模块。"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from feature_engineering import get_predictor_cols
from model_explain import explain_snapshot

# 共享 pipeline 步骤
_IMPUTER = ("imputer", SimpleImputer(strategy="median"))
_SCALER = ("scaler", StandardScaler())
_CV_SPLITS = TimeSeriesSplit(n_splits=3)
# 坐标下降：max_iter + tol 均为 sklearn 支持的收敛控制（RidgeCV 无 max_iter/tol）
_LINEAR_MAX_ITER = 50_000
_LINEAR_TOL = 1e-3


def _linear_pipeline(model):
    """构造线性类模型流水线：median 填充 → 标准化 → 回归器。

    调用方: get_model_pipelines（本模块）。
    """
    return Pipeline([_IMPUTER, _SCALER, ("model", model)])


def get_model_pipelines() -> Dict[str, Pipeline]:
    """Lasso/Ridge/ElasticNet/RandomForest/XGBoost 的 pipeline。
    线性模型：imputer(median) → scaler → 模型；树模型：imputer → 模型（无 scaler）；
    XGBoost 含 scaler 与线性模型路径一致（便于与 Lasso 等对比，非必须）。

    调用方: run_data_pipeline._process_one_underlying、carryforward.run_carryforward_experiment._process_one_underlying_cf。
    """
    return {
        "Lasso": _linear_pipeline(
            LassoCV(
                cv=_CV_SPLITS, n_alphas=50,
                max_iter=_LINEAR_MAX_ITER, tol=_LINEAR_TOL, random_state=42,
            )
        ),
        "Ridge": _linear_pipeline(
            RidgeCV(alphas=np.logspace(-3, 3, 50), cv=_CV_SPLITS)
        ),
        "ElasticNet": _linear_pipeline(
            ElasticNetCV(
                cv=_CV_SPLITS,
                max_iter=_LINEAR_MAX_ITER, tol=_LINEAR_TOL, random_state=42,
            )
        ),
        "RandomForest": Pipeline([
            _IMPUTER,
            ("model", RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=42)),
        ]),
        "XGBoost": Pipeline([
            _IMPUTER,
            _SCALER,  # 与线性模型统一缩放，便于对照；树模型通常可不缩放
            ("model", XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.01, random_state=42,
                verbosity=0, n_jobs=-1,
            )),
        ]),
    }


def _rolling_train_predict_core(
    df: pd.DataFrame,
    model_pipeline: Pipeline,
    y_col: str = "y_5s_ret",
    time_col: str = "trade_dt_et",
    feature_cols: Optional[List[str]] = None,
    group_cols: Tuple[str, ...] = ("date",),
    train_minutes: int = 30,
    refit_every_minutes: int = 5,
    min_train_rows: int = 200,
    capture_explain: bool = False,
    shap_max_samples: int = 256,
) -> Tuple[pd.DataFrame, List[str], Optional[pd.DataFrame]]:
    """滚动训练预测；可选返回末次成功 fit 的解释结果。

    在「同一 group_cols 分组（如单日）」内按 refit_time 网格滑动；若某步 train 行数 < min_train_rows
    则整步跳过（与 carry-forward 版不同，见 carryforward/run_carryforward_experiment.py）。

    调用方: rolling_train_predict、rolling_train_predict_with_explain（本模块）。
    """
    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data = data.dropna(subset=[time_col, y_col]).copy()
    if feature_cols is None:
        feature_cols = get_predictor_cols(data)  # 定义于 feature_engineering.get_predictor_cols
    feature_cols = [c for c in feature_cols if c in data.columns and not data[c].isna().all()]
    data = data[np.isfinite(pd.to_numeric(data[y_col], errors="coerce"))].copy()
    data = data.sort_values(list(group_cols) + [time_col]).reset_index(drop=False).rename(columns={"index": "orig_index"})

    all_preds = []
    last_explain: Optional[pd.DataFrame] = None
    # 按 group_cols（如 date）分日；日内再按 refit_time 网格滚动 train/test
    for day_key, g in data.groupby(list(group_cols), sort=True):
        g = g.sort_values(time_col).copy()
        day_start, day_end = g[time_col].min(), g[time_col].max()
        first_train_end = day_start + pd.Timedelta(minutes=train_minutes)
        if first_train_end >= day_end:
            continue
        # refit_time：从「首段训练窗结束」到日末，每隔 refit_every_minutes 一步
        for refit_time in pd.date_range(start=first_train_end, end=day_end, freq=f"{refit_every_minutes}min"):
            train_start = refit_time - pd.Timedelta(minutes=train_minutes)
            pred_end = refit_time + pd.Timedelta(minutes=refit_every_minutes)
            train_mask = (g[time_col] >= train_start) & (g[time_col] < refit_time)
            test_mask = (g[time_col] >= refit_time) & (g[time_col] < pred_end)
            train_df = g.loc[train_mask].copy()
            test_df = g.loc[test_mask].copy()
            if len(train_df) < min_train_rows or len(test_df) == 0:
                continue
            X_train = train_df[feature_cols]
            y_train = train_df[y_col].astype(float).to_numpy()
            X_test = test_df[feature_cols]
            model = clone(model_pipeline)
            model.fit(X_train, y_train)
            pred_part = test_df[["orig_index", time_col, y_col]].copy()
            pred_part["y_pred"] = model.predict(X_test)
            all_preds.append(pred_part)
            if capture_explain:
                last_explain = explain_snapshot(  # 定义于 model_explain.explain_snapshot
                    model, feature_cols, X_test, shap_max_samples=shap_max_samples
                )

    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return pred_df, feature_cols, last_explain if capture_explain else None


def rolling_train_predict(
    df: pd.DataFrame,
    model_pipeline: Pipeline,
    y_col: str = "y_5s_ret",
    time_col: str = "trade_dt_et",
    feature_cols: Optional[List[str]] = None,
    group_cols: Tuple[str, ...] = ("date",),
    train_minutes: int = 30,
    refit_every_minutes: int = 5,
    min_train_rows: int = 200,
) -> Tuple[pd.DataFrame, List[str]]:
    """任意 sklearn pipeline 的滚动训练预测（封装 _rolling_train_predict_core，无解释）。

    调用方: run_data_pipeline._process_one_underlying（explain 关闭时）。
    """
    pred_df, used_features, _ = _rolling_train_predict_core(
        df,
        model_pipeline,
        y_col=y_col,
        time_col=time_col,
        feature_cols=feature_cols,
        group_cols=group_cols,
        train_minutes=train_minutes,
        refit_every_minutes=refit_every_minutes,
        min_train_rows=min_train_rows,
    )
    return pred_df, used_features


def rolling_train_predict_with_explain(
    df: pd.DataFrame,
    model_pipeline: Pipeline,
    y_col: str = "y_5s_ret",
    time_col: str = "trade_dt_et",
    feature_cols: Optional[List[str]] = None,
    group_cols: Tuple[str, ...] = ("date",),
    train_minutes: int = 30,
    refit_every_minutes: int = 5,
    min_train_rows: int = 200,
    shap_max_samples: int = 256,
) -> Tuple[pd.DataFrame, List[str], Optional[pd.DataFrame]]:
    """滚动训练预测，并返回末次成功 fit 的解释结果。

    调用方: run_data_pipeline._process_one_underlying（explain 开启时）。
    """
    return _rolling_train_predict_core(
        df,
        model_pipeline,
        y_col=y_col,
        time_col=time_col,
        feature_cols=feature_cols,
        group_cols=group_cols,
        train_minutes=train_minutes,
        refit_every_minutes=refit_every_minutes,
        min_train_rows=min_train_rows,
        capture_explain=True,
        shap_max_samples=shap_max_samples,
    )


def evaluate_predictions(pred_df: pd.DataFrame, y_col: str = "y_5s_ret") -> pd.DataFrame:
    """RMSE、R²、方向准确率。dir_acc = sign(y)==sign(y_pred) 的比例。

    调用方: run_data_pipeline._process_one_underlying、carryforward.run_carryforward_experiment._process_one_underlying_cf。
    """
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()
    if y_col not in pred_df.columns or "y_pred" not in pred_df.columns:
        return pd.DataFrame()
    p = pred_df.dropna(subset=[y_col, "y_pred"]).copy()
    if len(p) == 0:
        return pd.DataFrame()
    return pd.DataFrame([{
        "n": len(p),
        "rmse": np.sqrt(mean_squared_error(p[y_col], p["y_pred"])),
        "r2": r2_score(p[y_col], p["y_pred"]),
        "dir_acc": (np.sign(p[y_col]) == np.sign(p["y_pred"])).mean(),
    }])

