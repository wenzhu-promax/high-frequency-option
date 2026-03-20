# -*- coding: utf-8 -*-
"""模型评估可视化：预测 vs 实际、残差、滚动 R²、系数热力图等"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import r2_score


def plot_pred_vs_actual(
    pred_df: pd.DataFrame,
    y_col: str = "y_5s_ret",
    title: str = "Predicted vs Actual (Lasso)",
    ax: Optional[Axes] = None,
) -> Axes:
    """y_pred vs y_true 散点"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    p = pred_df.dropna(subset=[y_col, "y_pred"]).copy()
    ax.scatter(p[y_col], p["y_pred"], alpha=0.2, s=3)
    ax.plot([-0.1, 0.1], [-0.1, 0.1], "r--", lw=2, label="y=x")
    ax.set_xlabel(f"y_true ({y_col})")
    ax.set_ylabel("y_pred")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    return ax


def plot_residuals(
    pred_df: pd.DataFrame,
    y_col: str = "y_5s_ret",
    ax1: Optional[Axes] = None,
    ax2: Optional[Axes] = None,
) -> tuple:
    """残差直方图 + 残差 vs y_pred"""
    p = pred_df.dropna(subset=[y_col, "y_pred"]).copy()
    resid = p[y_col] - p["y_pred"]
    if ax1 is None or ax2 is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax1, ax2 = axes[0], axes[1]
    ax1.hist(resid, bins=80, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Residual")
    ax1.set_title("Residual Distribution")
    ax2.scatter(p["y_pred"], resid, alpha=0.2, s=3)
    ax2.axhline(0, color="red", linestyle="--")
    ax2.set_xlabel("y_pred")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residual vs Predicted")
    return ax1, ax2


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: tuple = ("rmse", "r2", "dir_acc"),
    ax: Optional[Axes] = None,
) -> Axes:
    """多模型 rmse/r2/dir_acc 柱状对比"""
    if ax is None:
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        axes = [axes] if n == 1 else axes
    else:
        axes = [ax]
    models = comparison_df["model"].tolist()
    x = np.arange(len(models))
    w = 0.6
    titles = {"rmse": "RMSE by Model", "r2": "R² by Model", "dir_acc": "Direction Accuracy by Model"}
    colors = {"rmse": "steelblue", "r2": "darkgreen", "dir_acc": "coral"}
    for i, m in enumerate(metrics):
        if i < len(axes):
            axes[i].bar(x, comparison_df[m], width=w, color=colors.get(m, "gray"), alpha=0.8)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(models, rotation=45, ha="right")
            axes[i].set_ylabel(m.upper())
            axes[i].set_title(titles.get(m, m))
            if m == "dir_acc":
                axes[i].axhline(0.5, color="gray", linestyle="--", label="Random")
                axes[i].legend()
    plt.tight_layout()
    return axes[0] if axes else ax


def plot_rolling_r2(
    pred_df: pd.DataFrame,
    time_col: str = "trade_dt_et",
    y_col: str = "y_5s_ret",
    freq: str = "5min",
    rolling_window: int = 6,
    min_samples: int = 30,
    ax: Optional[Axes] = None,
) -> Axes:
    """按时间分桶算 R²，再滚动平滑。bin_r2 为每 freq 内的 R²，rolling_r2 为滚动均值。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    plot_df = pred_df.dropna(subset=[time_col, y_col, "y_pred"]).copy()
    plot_df[time_col] = pd.to_datetime(plot_df[time_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[time_col]).sort_values(time_col)
    bin_r2 = (
        plot_df.set_index(time_col)
        .groupby(pd.Grouper(freq=freq))
        .apply(lambda g: r2_score(g[y_col], g["y_pred"]) if len(g) >= min_samples else np.nan)
    )
    rolling_r2 = bin_r2.rolling(rolling_window, min_periods=rolling_window // 2).mean()
    ax.plot(rolling_r2.index, rolling_r2.values, lw=1.5, label="Rolling R²")
    ax.axhline(0, color="gray", linestyle="--", lw=1)
    ax.set_title("Rolling R² Over Time")
    ax.set_xlabel("Time (ET)")
    ax.set_ylabel("R²")
    ax.legend()
    return ax


def plot_hourly_direction_accuracy(
    pred_df: pd.DataFrame,
    time_col: str = "trade_dt_et",
    y_col: str = "y_5s_ret",
    ax: Optional[Axes] = None,
) -> Axes:
    """各小时方向准确率"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    p = pred_df.dropna(subset=[time_col, y_col, "y_pred"]).copy()
    p[time_col] = pd.to_datetime(p[time_col], errors="coerce")
    p["hour"] = p[time_col].dt.hour
    p["correct"] = (np.sign(p[y_col]) == np.sign(p["y_pred"])).astype(int)
    hourly = p.groupby("hour")["correct"].mean()
    ax.plot(hourly.index, hourly.values, "o-", color="darkgreen")
    ax.axhline(0.5, color="gray", linestyle="--", label="Random")
    ax.set_xlabel("Hour (ET)")
    ax.set_ylabel("Direction Accuracy")
    ax.set_title("Direction Accuracy by Hour")
    ax.legend()
    return ax


def plot_coef_heatmap(
    coef_df: pd.DataFrame,
    top_n: int = 20,
    ax: Optional[Axes] = None,
) -> Axes:
    """Lasso 系数随时间热力图"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    z = coef_df.copy()
    z["refit_time"] = pd.to_datetime(z["refit_time"], errors="coerce")
    z = z.dropna(subset=["refit_time"])
    top_feats = (
        z.assign(abs_coef=lambda t: t["coef"].abs())
        .groupby("feature", observed=True)["abs_coef"]
        .mean().nlargest(top_n).index.tolist()
    )
    mat = (
        z[z["feature"].isin(top_feats)]
        .pivot_table(index="feature", columns="refit_time", values="coef", aggfunc="mean")
        .reindex(top_feats)
    )
    im = ax.imshow(mat.to_numpy(), aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_title("Lasso Coefficient Stability Heatmap (Top 20 Features)")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    xt = np.arange(mat.shape[1])
    step = max(1, len(xt) // 10)
    ax.set_xticks(xt[::step])
    ax.set_xticklabels([str(mat.columns[i])[:16] for i in xt[::step]], rotation=45, ha="right")
    plt.colorbar(im, ax=ax, label="Coefficient Value")
    return ax


def plot_multi_model_rolling_r2(
    model_frames: Dict[str, pd.DataFrame],
    time_col: str = "trade_dt_et",
    y_col: str = "y_5s_ret",
    freq: str = "5min",
    rolling_window: int = 6,
    min_samples: int = 30,
    ax: Optional[Axes] = None,
) -> Axes:
    """多模型滚动 R² 曲线"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    for name, d in model_frames.items():
        if d is None or len(d) == 0:
            continue
        need = [time_col, y_col, "y_pred"]
        if not all(c in d.columns for c in need):
            continue
        x = d.dropna(subset=need).copy()
        x[time_col] = pd.to_datetime(x[time_col], errors="coerce")
        x = x.dropna(subset=[time_col]).sort_values(time_col)
        if len(x) == 0:
            continue
        r2_5m = (
            x.set_index(time_col)
            .groupby(pd.Grouper(freq=freq))
            .apply(lambda g: r2_score(g[y_col], g["y_pred"]) if len(g) >= min_samples else np.nan)
        )
        r2_roll = r2_5m.rolling(rolling_window, min_periods=rolling_window // 2).mean()
        ax.plot(r2_roll.index, r2_roll.values, lw=1.4, label=name)
    ax.axhline(0, color="gray", linestyle="--", lw=1)
    ax.set_title("Rolling R² Over Time (All Models)")
    ax.set_xlabel("Time (ET)")
    ax.set_ylabel("R²")
    ax.legend(ncol=3, fontsize=9)
    return ax
