# -*- coding: utf-8 -*-
"""末次滚动窗：线性模型 coef；树模型 feature_importances_ + TreeExplainer mean |SHAP|。"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def _X_transformed(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """对 sklearn Pipeline 中除最后一步 estimator 外的各步依次 transform，返回模型输入矩阵。

    调用方: explain_snapshot（本模块）。
    """
    Xv: pd.DataFrame | np.ndarray = X.copy()
    for _, step in pipe.steps[:-1]:
        Xv = step.transform(Xv)
        if not isinstance(Xv, pd.DataFrame):
            Xv = pd.DataFrame(Xv, columns=X.columns, index=X.index)
    return Xv.to_numpy(dtype=float)


def explain_snapshot(
    pipe: Pipeline,
    feature_names: List[str],
    X_test: pd.DataFrame,
    shap_max_samples: int = 256,
) -> pd.DataFrame:
    """取测试集前若干行，输出线性系数或树模型 importance + TreeExplainer 的 |SHAP| 均值表。

    调用方: model_training._rolling_train_predict_core（capture_explain=True 时）。
    """
    est = pipe.named_steps["model"]
    n = min(len(X_test), shap_max_samples)
    if n == 0:
        return pd.DataFrame(columns=["feature", "coef", "importance", "shap_mean"])

    Xt = _X_transformed(pipe, X_test.iloc[:n])
    coef = np.full(len(feature_names), np.nan)
    imp = np.full(len(feature_names), np.nan)
    shap_m = np.full(len(feature_names), np.nan)

    if hasattr(est, "coef_"):
        c = np.asarray(est.coef_).ravel()
        if c.size == len(feature_names):
            coef = c
    elif hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_).ravel()
        try:
            import shap

            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(Xt)
            if isinstance(sv, list):
                sv = sv[0]
            shap_m = np.abs(np.asarray(sv)).mean(axis=0).ravel()
        except Exception:
            # 未安装 shap、树不支持或样本异常时跳过 SHAP，coef/importance 仍可用
            pass

    return pd.DataFrame(
        {"feature": feature_names, "coef": coef, "importance": imp, "shap_mean": shap_m}
    )
