# -*- coding: utf-8 -*-
"""模型评估可视化：预测 vs 实际、残差、滚动 R²、系数热力图、按单维分组的 summary 箱线图等。

（moneyness×horizon）分面箱线图已迁至 analyze_summary_eqweight.boxplot_facets；本模块
plot_summary_boxplot_facets / plot_summary_metric_boxplots 为弃用桩。

调用方: 本仓库内无其它 .py 导入；供 Notebook/独立脚本或调试使用。
"""

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import r2_score


def _has_y_pred_cols(pred_df: pd.DataFrame, y_col: str, *extra: str) -> bool:
    """判断预测表是否非空且包含 y 真值、y_pred 及可选的额外列。

    调用方: plot_pred_vs_actual、plot_residuals、plot_rolling_r2、plot_hourly_direction_accuracy（本模块）。
    """
    if pred_df is None or pred_df.empty:
        return False
    cols = (y_col, "y_pred") + extra
    return all(c in pred_df.columns for c in cols)


def plot_pred_vs_actual(
    pred_df: pd.DataFrame,
    y_col: str = "y_5s_ret",
    title: str = "Predicted vs Actual (Lasso)",
    ax: Optional[Axes] = None,
) -> Axes:
    """绘制真实 y 与预测 y 的散点图，并叠加 y=x 参考线。

    调用方: 外部 Notebook/脚本（本仓库无 .py 引用）。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    if not _has_y_pred_cols(pred_df, y_col):  # 本模块 _has_y_pred_cols
        ax.set_title(title + " (无预测数据)")
        return ax
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
    """左：残差分布直方图；右：残差对预测值的散点。

    调用方: 外部 Notebook/脚本。
    """
    if ax1 is None or ax2 is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax1, ax2 = axes[0], axes[1]
    if not _has_y_pred_cols(pred_df, y_col):
        ax1.set_title("Residual (无预测数据)")
        ax2.set_title("Residual vs pred (无预测数据)")
        return ax1, ax2
    p = pred_df.dropna(subset=[y_col, "y_pred"]).copy()
    resid = p[y_col] - p["y_pred"]
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
    """按模型绘制多指标（rmse、r2、dir_acc）柱状对比图。

    调用方: 外部 Notebook/脚本。
    """
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
    """按时间频率分桶计算 R²，再对分桶 R² 做滚动平滑并绘制曲线。

    调用方: 外部 Notebook/脚本。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    if not _has_y_pred_cols(pred_df, y_col, time_col):
        ax.set_title("Rolling R² (无预测数据)")
        return ax
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
    """按交易小时聚合方向准确率并折线展示。

    调用方: 外部 Notebook/脚本。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    if not _has_y_pred_cols(pred_df, y_col, time_col):
        ax.set_title("Direction Accuracy by Hour (无预测数据)")
        return ax
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
    """对含 refit_time、feature、coef 的表，取平均绝对系数 Top-N 特征绘制系数热力图。

    调用方: 外部 Notebook/脚本。
    """
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
    """在同一坐标系叠加多条模型的滚动 R² 时间序列。

    调用方: 外部 Notebook/脚本。
    """
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


def plot_summary_metrics_boxplot(
    summary_df: pd.DataFrame,
    group_col: str,
    metrics: Sequence[str] = ("rmse", "r2", "dir_acc"),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """对 summary 表按 group_col 分组，为每个指标绘制箱线图，可选保存到文件。

    调用方: 外部 Notebook/脚本。
    """
    df = summary_df.dropna(subset=[group_col]).copy()
    levels = sorted(df[group_col].unique(), key=lambda x: (str(type(x)), str(x)))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    titles = {"rmse": "RMSE", "r2": "R²", "dir_acc": "Dir. acc."}
    for ax, m in zip(axes, metrics):
        if m not in df.columns:
            ax.set_visible(False)
            continue
        series = [df.loc[df[group_col] == lv, m].dropna().to_numpy() for lv in levels]
        labels = [str(lv) for lv in levels]
        ax.boxplot(series, labels=labels)
        ax.set_title(titles.get(m, m))
        ax.set_xlabel(group_col)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# --- summary 分面箱线图：与 analyze_summary_eqweight.boxplot_facets 重复，原实现已移除 ---
# 请使用: python analyze_summary_eqweight.py --mode eqweight
# 或: from analyze_summary_eqweight import boxplot_facets, run_eqweight_analysis


def plot_summary_boxplot_facets(
    df: pd.DataFrame,
    metric: str,
    ref_line: float,
    ylabel: str,
    title: str,
    save_path: Path,
    y_mode: str = "auto",
    ylim: tuple | None = None,
) -> None:
    """已弃用：与 analyze_summary_eqweight 重复，调用即报错以引导迁移。

    调用方: 不应调用；请改用 analyze_summary_eqweight.boxplot_facets。
    """
    raise NotImplementedError(
        "分面箱线图请使用 analyze_summary_eqweight.boxplot_facets 或 "
        "python analyze_summary_eqweight.py --mode eqweight"
    )


def plot_summary_metric_boxplots(
    input_path: Path,
    output_dir: Path,
    metric: str = "r2",
    y_mode: str | None = None,
    ylim: tuple | None = None,
    title: str | None = None,
    legacy_name: str | None = None,
) -> Path:
    """已弃用：请使用 analyze_summary_eqweight 生成 summary 分面箱线图。

    调用方: 不应调用。
    """
    raise NotImplementedError(
        "请使用 analyze_summary_eqweight.py（CLI）或其中 boxplot_facets / run_eqweight_analysis。"
    )


plot_boxplot_facets = plot_summary_boxplot_facets
plot_metric_boxplots = plot_summary_metric_boxplots
