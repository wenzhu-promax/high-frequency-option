#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified summary analysis for summary_experiment.csv.

Modes:
1. eqweight: 原有 equal-weight 分析
2. rigorous: 原有 rigorous 画图分析
3. all: 同时执行两套分析，分别写入子目录

分面箱线图（moneyness × horizon）以本模块 `boxplot_facets` 为准；
`model_plots.plot_summary_boxplot_facets` 已弃用，避免重复实现。

各函数 docstring 中含「调用方: …」与本文件内交叉引用；跨模块调用较少（仅标准库与 pandas 等）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CELL_COLS = ["underlying", "moneyness", "horizon"]
METRIC_COLS = ["n", "rmse", "r2", "dir_acc"]
DEFAULT_MODEL_ORDER = ["Lasso", "Ridge", "ElasticNet", "RandomForest", "XGBoost"]


def parse_args() -> argparse.Namespace:
    """解析 summary 分析脚本的输入路径、输出目录、模式与各类 min_n 阈值。

    调用方: main（本模块）。
    """
    parser = argparse.ArgumentParser(description="Unified summary analysis")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/summary_experiment.csv"),
        help="Path to summary_experiment.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/analysis_eqweight"),
        help="Output directory. In --mode all, this is the parent directory.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eqweight", "rigorous", "all"],
        default="eqweight",
        help="Analysis mode",
    )
    parser.add_argument(
        "--main-min-n",
        type=float,
        default=50.0,
        help="Equal-weight minimum n threshold",
    )
    parser.add_argument(
        "--rigorous-min-n",
        type=float,
        default=50.0,
        help="Rigorous mode minimum n threshold",
    )
    parser.add_argument(
        "--rigorous-output-dir",
        type=Path,
        default=None,
        help="Optional output directory for rigorous mode",
    )
    return parser.parse_args()


def model_order_present(df: pd.DataFrame) -> list[str]:
    """返回表中出现的模型名，按默认优先级排序，其余字母序接在后面。

    调用方: complete_cells、overall_equal_weight_summary、equal_weight_win_rate、weighted_summary、
    boxplot_facets 等（本模块）。
    """
    present = set(df["model"].dropna().unique().tolist())
    ordered = [m for m in DEFAULT_MODEL_ORDER if m in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """将指标与分组列转为数值/字符串，便于后续聚合与排序。

    调用方: main（本模块）。
    """
    out = df.copy()
    for col in METRIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["moneyness"] = pd.to_numeric(out["moneyness"], errors="coerce")
    out["horizon"] = pd.to_numeric(out["horizon"], errors="coerce")
    out["underlying"] = out["underlying"].astype(str)
    out["model"] = out["model"].astype(str)
    return out


def complete_cells(df: pd.DataFrame, min_n: float) -> pd.DataFrame:
    """保留 n>=min_n 且每个 (underlying,moneyness,horizon) 单元格上全部模型都存在的行。

    调用方: run_eqweight_analysis、run_rigorous_analysis、sensitivity_analysis（本模块）。
    """
    models = model_order_present(df)
    filtered = df[df["n"] >= min_n].copy()
    cell_counts = filtered.groupby(CELL_COLS)["model"].nunique().reset_index(name="n_models")
    complete = cell_counts[cell_counts["n_models"] == len(models)][CELL_COLS]
    return filtered.merge(complete, on=CELL_COLS, how="inner")


def equal_weight_over_underlyings(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_map: dict[str, str],
) -> pd.DataFrame:
    """先在每个标的内按 group_cols 聚合，再对标的等权平均（跨标的简单平均）。

    调用方: horizon_decay_equal_weight、moneyness_equal_weight（本模块）。
    """
    per_underlying = df.groupby(["underlying"] + group_cols, as_index=False).agg(metric_map)
    across_underlying = per_underlying.groupby(group_cols, as_index=False).mean(numeric_only=True)
    n_by_group = (
        per_underlying.groupby(group_cols, as_index=False)["underlying"]
        .nunique()
        .rename(columns={"underlying": "n_underlyings"})
    )
    return across_underlying.merge(n_by_group, on=group_cols, how="left")


def overall_equal_weight_summary(df: pd.DataFrame) -> pd.DataFrame:
    """先按标的×模型聚合 r2/dir_acc 等，再对标的等权平均得到每模型总览。

    调用方: run_eqweight_analysis、sensitivity_analysis（本模块）。
    """
    per_underlying = df.groupby(["underlying", "model"], as_index=False).agg(
        eq_avg_r2=("r2", "mean"),
        eq_avg_dir_acc=("dir_acc", "mean"),
        eq_r2_positive_share=("r2", lambda s: float((s > 0).mean())),
        n_cells=("r2", "size"),
    )
    overall = per_underlying.groupby("model", as_index=False).agg(
        eq_avg_r2=("eq_avg_r2", "mean"),
        eq_avg_dir_acc=("eq_avg_dir_acc", "mean"),
        eq_r2_positive_share=("eq_r2_positive_share", "mean"),
        n_underlyings=("underlying", "nunique"),
        avg_cells_per_underlying=("n_cells", "mean"),
    )
    overall["model"] = pd.Categorical(overall["model"], categories=model_order_present(df), ordered=True)
    return overall.sort_values("model").reset_index(drop=True)


def equal_weight_win_rate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """在每个 (underlying,moneyness,horizon) 内看谁 metric 最大，再按标的等权得到模型胜率。

    调用方: run_eqweight_analysis、sensitivity_analysis（本模块）。
    """
    tmp = df[CELL_COLS + ["model", metric]].copy()
    tmp["cell_best"] = tmp.groupby(CELL_COLS)[metric].transform("max")
    tmp["is_winner"] = (tmp[metric] == tmp["cell_best"]).astype(float)
    tmp["n_winners"] = tmp.groupby(CELL_COLS)["is_winner"].transform("sum")
    tmp["win_fraction"] = np.where(tmp["is_winner"] > 0, 1.0 / tmp["n_winners"], 0.0)

    per_underlying = tmp.groupby(["underlying", "model"], as_index=False).agg(eq_win_rate=("win_fraction", "mean"))
    out = per_underlying.groupby("model", as_index=False).agg(
        eq_win_rate=("eq_win_rate", "mean"),
        n_underlyings=("underlying", "nunique"),
    )
    out = out.rename(columns={"eq_win_rate": f"eq_win_rate_{metric}"})
    out["model"] = pd.Categorical(out["model"], categories=model_order_present(df), ordered=True)
    return out.sort_values("model").reset_index(drop=True)


def horizon_decay_equal_weight(df: pd.DataFrame) -> pd.DataFrame:
    """按 horizon×模型对标的等权聚合 r2 与 dir_acc，用于观察预测步长衰减。

    调用方: run_eqweight_analysis（本模块）。
    """
    return equal_weight_over_underlyings(
        df,
        group_cols=["horizon", "model"],
        metric_map={"r2": "mean", "dir_acc": "mean"},
    ).rename(columns={"r2": "eq_avg_r2", "dir_acc": "eq_avg_dir_acc"})


def moneyness_equal_weight(df: pd.DataFrame) -> pd.DataFrame:
    """按 moneyness×模型对标的等权聚合 r2 与 dir_acc。

    调用方: run_eqweight_analysis（本模块）。
    """
    return equal_weight_over_underlyings(
        df,
        group_cols=["moneyness", "model"],
        metric_map={"r2": "mean", "dir_acc": "mean"},
    ).rename(columns={"r2": "eq_avg_r2", "dir_acc": "eq_avg_dir_acc"})


def weighted_summary(df: pd.DataFrame) -> pd.DataFrame:
    """按样本量 n 对 r2、dir_acc 加权平均（非 pooled R²），按模型汇总。

    调用方: run_eqweight_analysis、compare_equal_vs_weighted（本模块）。
    """
    rows = []
    for model, sub in df.groupby("model"):
        w = sub["n"].to_numpy(dtype=float)
        r2 = sub["r2"].to_numpy(dtype=float)
        dir_acc = sub["dir_acc"].to_numpy(dtype=float)
        r2_mask = np.isfinite(w) & np.isfinite(r2)
        da_mask = np.isfinite(w) & np.isfinite(dir_acc)
        rows.append(
            {
                "model": model,
                "weighted_avg_r2": float(np.average(r2[r2_mask], weights=w[r2_mask])) if r2_mask.any() else np.nan,
                "weighted_avg_dir_acc": float(np.average(dir_acc[da_mask], weights=w[da_mask])) if da_mask.any() else np.nan,
                "n_complete_rows": len(sub),
                "note": "weighted_avg_r2 is NOT pooled_r2",
            }
        )
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], categories=model_order_present(df), ordered=True)
    return out.sort_values("model").reset_index(drop=True)


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    """对 x 以 w 为权做加权平均，忽略非有限值。

    调用方: plot_weighted_summary_by_horizon（本模块，内部聚合）。
    """
    mask = np.isfinite(x.to_numpy()) & np.isfinite(w.to_numpy())
    if not mask.any():
        return np.nan
    xv = x.to_numpy(dtype=float)[mask]
    wv = w.to_numpy(dtype=float)[mask]
    if wv.sum() <= 0:
        return np.nan
    return float(np.average(xv, weights=wv))


def sensitivity_analysis(raw: pd.DataFrame, thresholds: Iterable[int]) -> pd.DataFrame:
    """对多个 min_n 阈值重复 complete_cells 与总览指标，检验结论对样本量门槛的敏感性。

    调用方: run_eqweight_analysis（本模块）。
    """
    blocks = []
    for min_n in thresholds:
        comp = complete_cells(raw, min_n=min_n)
        overall = overall_equal_weight_summary(comp)
        win_r2 = equal_weight_win_rate(comp, "r2")[["model", "eq_win_rate_r2"]]
        win_da = equal_weight_win_rate(comp, "dir_acc")[["model", "eq_win_rate_dir_acc"]]
        merged = overall.merge(win_r2, on="model", how="left").merge(win_da, on="model", how="left")
        merged["min_n"] = min_n
        merged["n_complete_rows"] = len(comp)
        merged["n_complete_cells"] = comp[CELL_COLS].drop_duplicates().shape[0]
        blocks.append(merged)
    return pd.concat(blocks, ignore_index=True)


def boxplot_facets(
    df: pd.DataFrame,
    metric: str,
    ref_line: float,
    ylabel: str,
    title: str,
    save_path: Path,
    sharey: bool | None = None,
) -> None:
    """(moneyness × horizon) 分面箱线图。

    调用方: run_eqweight_analysis（本模块）。替代 model_plots 中已弃用的分面实现。
    """
    models = model_order_present(df)  # 本模块 model_order_present
    moneyness_vals = sorted(df["moneyness"].dropna().unique().tolist())
    horizon_vals = sorted(df["horizon"].dropna().unique().tolist())
    if sharey is None:
        sharey = metric != "r2"

    fig, axes = plt.subplots(
        len(moneyness_vals),
        len(horizon_vals),
        figsize=(3.2 * len(horizon_vals), 3.4 * len(moneyness_vals)),
        sharey=sharey,
    )
    if len(moneyness_vals) == 1:
        axes = np.array([axes])

    rng = np.random.default_rng(42)
    for i, mo in enumerate(moneyness_vals):
        for j, hz in enumerate(horizon_vals):
            ax = axes[i, j]
            sub = df[(df["moneyness"] == mo) & (df["horizon"] == hz)].copy()
            series = [sub.loc[sub["model"] == model, metric].dropna().to_numpy(dtype=float) for model in models]
            ax.boxplot(series, labels=models, showfliers=False)
            ax.axhline(ref_line, color="red", linestyle="--", linewidth=1)

            for k, model in enumerate(models, start=1):
                vals = sub.loc[sub["model"] == model, metric].dropna().to_numpy(dtype=float)
                if len(vals) == 0:
                    continue
                x = k + rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(x, vals, s=8, alpha=0.45, color="black")

            if i == 0:
                ax.set_title(f"h={int(hz)}")
            if j == 0:
                ax.set_ylabel(f"m={mo}\n{ylabel}")
            ax.tick_params(axis="x", rotation=30, labelsize=8)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def lineplot_metrics(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    save_path: Path,
    title_prefix: str,
) -> None:
    """双面板折线：横轴为 x_col，纵轴分别为等权平均 R² 与等权平均方向准确率。

    调用方: run_eqweight_analysis（本模块）。
    """
    models = model_order_present(df)  # 本模块
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for model in models:
        sub = df[df["model"] == model].sort_values(x_col)
        axes[0].plot(sub[x_col], sub["eq_avg_r2"], marker="o", label=model)
        axes[1].plot(sub[x_col], sub["eq_avg_dir_acc"], marker="o", label=model)
    axes[0].axhline(0.0, color="red", linestyle="--", linewidth=1)
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=1)
    axes[0].set_title(f"{title_prefix}: Equal-weight Avg R²")
    axes[1].set_title(f"{title_prefix}: Equal-weight Avg Dir. Acc.")
    axes[0].set_xlabel(x_label)
    axes[1].set_xlabel(x_label)
    axes[0].set_ylabel("R²")
    axes[1].set_ylabel("Direction Accuracy")
    axes[1].legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def compare_equal_vs_weighted(eq_df: pd.DataFrame, wt_df: pd.DataFrame, save_path: Path) -> None:
    """并排柱状图对比「标的等权」与「按 n 加权」的 R² 与方向准确率。

    调用方: run_eqweight_analysis（本模块）。
    """
    models = eq_df["model"].astype(str).tolist()
    x = np.arange(len(models))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(x - width / 2, eq_df["eq_avg_r2"], width=width, label="Equal-weight")
    axes[0].bar(x + width / 2, wt_df["weighted_avg_r2"], width=width, label="Weighted by n")
    axes[0].axhline(0.0, color="red", linestyle="--", linewidth=1)
    axes[0].set_title("R²: Equal-weight vs Weighted")
    axes[0].set_ylabel("R²")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=25)

    axes[1].bar(x - width / 2, eq_df["eq_avg_dir_acc"], width=width, label="Equal-weight")
    axes[1].bar(x + width / 2, wt_df["weighted_avg_dir_acc"], width=width, label="Weighted by n")
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=1)
    axes[1].set_title("Dir. Acc.: Equal-weight vs Weighted")
    axes[1].set_ylabel("Direction Accuracy")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_metric_facets_rigorous(
    complete: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    ref_line: float | None,
    save_path: Path,
) -> None:
    """Rigorous 分面图：仅对有数据的模型画箱线；R² 分面独立 y 轴。

    调用方: run_rigorous_analysis（本模块）。
    """
    models = sorted(complete["model"].dropna().unique().tolist())
    moneyness_values = sorted(complete["moneyness"].dropna().unique().tolist())
    horizons = sorted(complete["horizon"].dropna().unique().tolist())

    sharey = metric_col != "r2"
    fig, axes = plt.subplots(
        len(moneyness_values),
        len(horizons),
        figsize=(3.2 * len(horizons), 3.4 * len(moneyness_values)),
        sharey=sharey,
    )
    if len(moneyness_values) == 1 and len(horizons) == 1:
        axes = np.array([[axes]])
    elif len(moneyness_values) == 1:
        axes = np.array([axes])
    elif len(horizons) == 1:
        axes = np.array([[ax] for ax in axes])

    for i, mo in enumerate(moneyness_values):
        for j, h in enumerate(horizons):
            ax = axes[i, j]
            sub = complete[(complete["moneyness"] == mo) & (complete["horizon"] == h)]
            series = []
            labels = []
            for model in models:
                vals = sub.loc[sub["model"] == model, metric_col].dropna().to_numpy(dtype=float)
                if len(vals) == 0:
                    continue
                series.append(vals)
                labels.append(model)

            if series:
                ax.boxplot(series, labels=labels, showfliers=False)
            if ref_line is not None:
                ax.axhline(ref_line, color="gray", linestyle="--", linewidth=1)
            if i == 0:
                ax.set_title(f"h={int(h)}")
            if j == 0:
                ax.set_ylabel(f"m={mo}\n{ylabel}")
            ax.tick_params(axis="x", rotation=30)
            if not series:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=9)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_n_distribution(filtered: pd.DataFrame, min_n: float, save_path: Path) -> None:
    """按 horizon 分组的样本量 n 取 log10 后箱线图，并标注 min_n 参考线。

    调用方: run_rigorous_analysis（本模块）。
    """
    horizons = sorted(filtered["horizon"].dropna().unique().tolist())
    data = []
    labels = []
    for h in horizons:
        vals = filtered.loc[filtered["horizon"] == h, "n"].dropna()
        if len(vals) == 0:
            continue
        data.append(np.log10(vals.to_numpy(dtype=float)))
        labels.append(str(int(h)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.axhline(np.log10(min_n), color="red", linestyle="--", linewidth=1, label=f"log10(min_n={min_n:g})")
    ax.set_title("Sample Size Distribution by Horizon")
    ax.set_xlabel("Horizon (seconds)")
    ax.set_ylabel("log10(n)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_weighted_summary_by_horizon(complete: pd.DataFrame, save_path: Path) -> None:
    """对每个模型×horizon 用 n 加权 RMSE/R²/方向准确率，三面板折线。

    调用方: run_rigorous_analysis（本模块）。
    """
    horizons = sorted(complete["horizon"].dropna().unique().tolist())
    models = model_order_present(complete)
    rows = []
    for model in models:
        for h in horizons:
            sub = complete[(complete["model"] == model) & (complete["horizon"] == h)]
            rows.append(
                {
                    "model": model,
                    "horizon": h,
                    "weighted_rmse": weighted_mean(sub["rmse"], sub["n"]),  # 本模块 weighted_mean
                    "weighted_r2": weighted_mean(sub["r2"], sub["n"]),
                    "weighted_dir_acc": weighted_mean(sub["dir_acc"], sub["n"]),
                }
            )
    agg = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    for model in models:
        sub = agg[agg["model"] == model].sort_values("horizon")
        axes[0].plot(sub["horizon"], sub["weighted_rmse"], marker="o", label=model)
        axes[1].plot(sub["horizon"], sub["weighted_r2"], marker="o", label=model)
        axes[2].plot(sub["horizon"], sub["weighted_dir_acc"], marker="o", label=model)

    axes[0].set_title("Weighted Mean RMSE by Horizon")
    axes[0].set_xlabel("Horizon (seconds)")
    axes[0].set_ylabel("Weighted RMSE")
    axes[1].set_title("Weighted Mean R² by Horizon")
    axes[1].set_xlabel("Horizon (seconds)")
    axes[1].set_ylabel("Weighted R²")
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[2].set_title("Weighted Mean Direction Accuracy by Horizon")
    axes[2].set_xlabel("Horizon (seconds)")
    axes[2].set_ylabel("Weighted Dir. Acc.")
    axes[2].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[2].legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_eqweight_readme(out_dir: Path, main_min_n: float, raw: pd.DataFrame, complete: pd.DataFrame) -> None:
    """在输出目录写入 equal-weight 分析说明与数据统计的 README.md。

    调用方: run_eqweight_analysis（本模块）。
    """
    models = model_order_present(raw)
    text = f"""# Equal-weight analysis

Main rules:
- equal-weight across underlyings
- cell = (underlying, moneyness, horizon)
- keep only complete cells with all models present
- main min_n = {main_min_n:g}

Benchmarks:
- R² reference line is 0 (mean predictor benchmark)
- dir_acc reference line is 0.5 (random-direction benchmark)

Warnings:
- weighted_avg_r2 is NOT pooled_r2
- no significance tests / confidence intervals / residual analysis

Raw rows: {len(raw)}
Complete rows (main): {len(complete)}
Complete cells (main): {complete[CELL_COLS].drop_duplicates().shape[0]}
Models: {", ".join(models)}
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def write_rigorous_readme(
    raw_df: pd.DataFrame,
    filtered: pd.DataFrame,
    complete: pd.DataFrame,
    min_n: float,
    save_path: Path,
) -> None:
    """写入 rigorous 诊断说明：行数、配对单元格、按 horizon 的完整 cell 数量等。

    调用方: run_rigorous_analysis（本模块）。
    """
    models = model_order_present(raw_df)
    full_cells = complete[CELL_COLS].drop_duplicates()
    lines = [
        "# Rigorous Plot Diagnostics",
        "",
        "- benchmark lines:",
        "  - `R² = 0` 对应均值预测器",
        "  - `dir_acc = 0.5` 对应随机猜方向",
        f"- min_n: `{min_n:g}`",
        f"- raw rows: `{len(raw_df)}`",
        f"- filtered rows (n >= min_n): `{len(filtered)}`",
        f"- raw distinct cells: `{raw_df[CELL_COLS].drop_duplicates().shape[0]}`",
        f"- paired distinct cells (all {len(models)} models present): `{len(full_cells)}`",
        f"- models: `{', '.join(models)}`",
        f"- underlyings: `{raw_df['underlying'].nunique()}`",
        "",
        "## Rows Filtered Out By n",
        f"- rows with n < min_n: `{int((raw_df['n'] < min_n).sum())}`",
        "",
        "## Paired Cell Counts By Horizon",
    ]
    counts = full_cells.groupby("horizon").size().sort_index()
    for h, c in counts.items():
        lines.append(f"- horizon {int(h)}: `{int(c)}` cells")
    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_eqweight_analysis(raw: pd.DataFrame, out_dir: Path, min_n: float) -> None:
    """执行等权分析：导出 CSV、箱线图、衰减线、敏感性表与 README。

    调用方: main（本模块，--mode eqweight 或 all）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    complete = complete_cells(raw, min_n=min_n)
    overall_eq = overall_equal_weight_summary(complete)
    win_r2 = equal_weight_win_rate(complete, "r2")
    win_da = equal_weight_win_rate(complete, "dir_acc")
    horizon_eq = horizon_decay_equal_weight(complete)
    moneyness_eq = moneyness_equal_weight(complete)
    weighted = weighted_summary(complete)
    sensitivity = sensitivity_analysis(raw, thresholds=(20, 50, 100))
    overall_main = (
        overall_eq
        .merge(win_r2[["model", "eq_win_rate_r2"]], on="model", how="left")
        .merge(win_da[["model", "eq_win_rate_dir_acc"]], on="model", how="left")
    )

    complete.sort_values(CELL_COLS + ["model"]).to_csv(out_dir / f"complete_cells_min_n_{int(min_n)}.csv", index=False)
    overall_main.to_csv(out_dir / "overall_equal_weight_summary.csv", index=False)
    win_r2.to_csv(out_dir / "win_rate_r2_equal_weight.csv", index=False)
    win_da.to_csv(out_dir / "win_rate_dir_acc_equal_weight.csv", index=False)
    horizon_eq.to_csv(out_dir / "horizon_decay_equal_weight.csv", index=False)
    moneyness_eq.to_csv(out_dir / "moneyness_heterogeneity_equal_weight.csv", index=False)
    weighted.to_csv(out_dir / "weighted_summary_robustness.csv", index=False)
    sensitivity.to_csv(out_dir / "sensitivity_min_n_20_50_100.csv", index=False)

    boxplot_facets(
        complete,
        metric="r2",
        ref_line=0.0,
        ylabel="R²",
        title="Cross-sectional Boxplots of R² by Model",
        save_path=fig_dir / "boxplot_r2_by_model_facets.png",
    )
    boxplot_facets(
        complete,
        metric="dir_acc",
        ref_line=0.5,
        ylabel="Direction Accuracy",
        title="Cross-sectional Boxplots of Direction Accuracy by Model",
        save_path=fig_dir / "boxplot_dir_acc_by_model_facets.png",
    )
    lineplot_metrics(
        horizon_eq,
        x_col="horizon",
        x_label="Horizon (seconds)",
        save_path=fig_dir / "horizon_decay_equal_weight.png",
        title_prefix="Horizon Decay",
    )
    lineplot_metrics(
        moneyness_eq,
        x_col="moneyness",
        x_label="Moneyness",
        save_path=fig_dir / "moneyness_heterogeneity_equal_weight.png",
        title_prefix="Moneyness Heterogeneity",
    )
    compare_equal_vs_weighted(overall_eq, weighted, save_path=fig_dir / "equal_vs_weighted_overall.png")
    write_eqweight_readme(out_dir, min_n, raw, complete)


def run_rigorous_analysis(raw: pd.DataFrame, out_dir: Path, min_n: float) -> None:
    """执行 rigorous 诊断图：n 分布、分面箱线图、按 horizon 加权曲线与 README。

    调用方: main（本模块，--mode rigorous 或 all）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered = raw[raw["n"] >= min_n].copy()
    complete = complete_cells(raw, min_n=min_n)

    plot_n_distribution(filtered, min_n=min_n, save_path=out_dir / "n_distribution_by_horizon.png")
    plot_metric_facets_rigorous(
        complete,
        metric_col="r2",
        ylabel="R²",
        title="R² by Moneyness and Horizon (0 = mean benchmark)",
        ref_line=0.0,
        save_path=out_dir / "r2_facets_mean_benchmark.png",
    )
    plot_metric_facets_rigorous(
        complete,
        metric_col="dir_acc",
        ylabel="Dir. Acc.",
        title="Direction Accuracy by Moneyness and Horizon (0.5 = random benchmark)",
        ref_line=0.5,
        save_path=out_dir / "dir_acc_facets_random_benchmark.png",
    )
    plot_weighted_summary_by_horizon(complete, save_path=out_dir / "weighted_summary_by_horizon.png")
    write_rigorous_readme(raw, filtered, complete, min_n=min_n, save_path=out_dir / "README.md")


def main() -> None:
    """根据 --mode 调用 eqweight、rigorous 或两者并输出到对应子目录。

    调用方: ``if __name__ == "__main__"``（本模块）。
    """
    args = parse_args()
    raw = ensure_numeric(pd.read_csv(args.input))
    outputs = []

    if args.mode in ("eqweight", "all"):
        eq_out = args.output_dir if args.mode == "eqweight" else args.output_dir / "eqweight"
        run_eqweight_analysis(raw, out_dir=eq_out, min_n=args.main_min_n)
        outputs.append(f"eqweight={eq_out}")

    if args.mode in ("rigorous", "all"):
        if args.rigorous_output_dir is not None:
            rig_out = args.rigorous_output_dir
        else:
            rig_out = args.output_dir if args.mode == "rigorous" else args.output_dir / "rigorous"
        run_rigorous_analysis(raw, out_dir=rig_out, min_n=args.rigorous_min_n)
        outputs.append(f"rigorous={rig_out}")

    print("Done. Outputs: " + ", ".join(outputs))


if __name__ == "__main__":
    main()
