# -*- coding: utf-8 -*-
"""DOLG+SupCon 相对 DOLG+ArcFace 的逐牛相关分析。

分析保持同一逻辑：
1. 横轴：Hard gap 改善 = SupCon hard_gap - ArcFace hard_gap
2. 纵轴1：Hard集正常类 Recall 改善
3. 纵轴2：Hard集正常类 F1 改善
4. 统计单位：牛只
5. Pearson 线性相关；30,000 次逐牛 bootstrap 估计 r 的 95% CI
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_PROJECT_ROOT = Path(r"D:\code_project\cow_recognition")
N_BOOTSTRAP = 30_000
RANDOM_SEED = 20260916


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="默认保存到 paper_materials/cow_macro_distance/gap_performance_correlation",
    )
    return parser.parse_args()


def read_unique_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")
    df = pd.read_csv(path, dtype={"cow_id": str})
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{path.name} 缺少字段：{missing}")
    if df["cow_id"].duplicated().any():
        duplicated = df.loc[df["cow_id"].duplicated(), "cow_id"].tolist()[:5]
        raise ValueError(f"{path.name} 出现重复 cow_id，例如：{duplicated}")
    return df[columns].copy()


def bootstrap_pearson_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    """以牛只为单位有放回重抽样，返回 Pearson r 的百分位数 95% CI。"""
    rng = np.random.default_rng(seed)
    n = x.size
    correlations = np.empty(n_bootstrap, dtype=float)
    chunk_size = 2_000

    for start in range(0, n_bootstrap, chunk_size):
        size = min(chunk_size, n_bootstrap - start)
        indices = rng.integers(0, n, size=(size, n))
        xb = x[indices]
        yb = y[indices]
        xb_centered = xb - xb.mean(axis=1, keepdims=True)
        yb_centered = yb - yb.mean(axis=1, keepdims=True)
        numerator = np.sum(xb_centered * yb_centered, axis=1)
        denominator = np.sqrt(
            np.sum(xb_centered**2, axis=1) * np.sum(yb_centered**2, axis=1)
        )
        correlations[start : start + size] = np.divide(
            numerator,
            denominator,
            out=np.full(size, np.nan),
            where=denominator > 0,
        )

    correlations = correlations[np.isfinite(correlations)]
    return tuple(np.quantile(correlations, [0.025, 0.975]))


def regression_line_ci(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """普通最小二乘直线及均值回归线的95% CI。"""
    fit = stats.linregress(x, y)
    predicted = fit.intercept + fit.slope * grid

    fitted_at_x = fit.intercept + fit.slope * x
    residuals = y - fitted_at_x
    n = x.size
    residual_std = np.sqrt(np.sum(residuals**2) / (n - 2))
    sxx = np.sum((x - x.mean()) ** 2)
    standard_error = residual_std * np.sqrt(1 / n + (grid - x.mean()) ** 2 / sxx)
    critical_value = stats.t.ppf(0.975, df=n - 2)
    lower = predicted - critical_value * standard_error
    upper = predicted + critical_value * standard_error
    return predicted, lower, upper, fit.slope, fit.intercept


def analyze_metric(
    x: np.ndarray,
    y: np.ndarray,
    metric_name: str,
    seed: int,
) -> dict[str, float | int | str]:
    pearson_r, pearson_p = stats.pearsonr(x, y)
    ci_low, ci_high = bootstrap_pearson_ci(x, y, N_BOOTSTRAP, seed)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    regression = stats.linregress(x, y)
    return {
        "metric": metric_name,
        "n_cows": x.size,
        "pearson_r": pearson_r,
        "pearson_ci_low": ci_low,
        "pearson_ci_high": ci_high,
        "pearson_p": pearson_p,
        "r_squared": pearson_r**2,
        "slope": regression.slope,
        "intercept": regression.intercept,
        "spearman_rho": spearman_r,
        "spearman_p": spearman_p,
        "n_bootstrap": N_BOOTSTRAP,
        "random_seed": seed,
    }


def p_text(value: float) -> str:
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}"


def draw_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    y_column: str,
    title: str,
    y_label: str,
    color: str,
    statistics: dict[str, float | int | str],
) -> None:
    x = data["hard_gap_improvement"].to_numpy(dtype=float)
    y = data[y_column].to_numpy(dtype=float)

    grid = np.linspace(x.min(), x.max(), 300)
    predicted, lower, upper, _, _ = regression_line_ci(x, y, grid)

    ax.scatter(
        x,
        y,
        s=34,
        color=color,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.45,
        zorder=2,
    )
    ax.fill_between(grid, lower, upper, color=color, alpha=0.18, linewidth=0)
    ax.plot(grid, predicted, color="#202020", linewidth=2.1, zorder=3)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1.1)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1.1)

    annotation = (
        f"Pearson r = {statistics['pearson_r']:.3f}\n"
        f"95% CI [{statistics['pearson_ci_low']:.3f}, "
        f"{statistics['pearson_ci_high']:.3f}]\n"
        f"{p_text(float(statistics['pearson_p']))}\n"
        f"R² = {statistics['r_squared']:.3f}\n"
        f"n = {int(statistics['n_cows'])} cows"
    )
    ax.text(
        0.035,
        0.965,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#B8B8B8",
            "alpha": 0.94,
        },
    )
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(
        "Improvement in Hard gap\n(DOLG+ArcFace+SupCon − DOLG+ArcFace)",
        fontsize=10.5,
    )
    ax.set_ylabel(y_label, fontsize=10.5)
    ax.grid(color="#D9D9D9", linewidth=0.75, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root
        / "paper_materials"
        / "cow_macro_distance"
        / "gap_performance_correlation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 如果文件位置发生变化，只修改以下四个路径。
    arcface_distance_csv = (
        project_root / "paper_materials/cow_macro_distance/dolg_arcface_per_cow.csv"
    )
    supcon_distance_csv = (
        project_root / "paper_materials/cow_macro_distance/dolg_supcon_per_cow.csv"
    )
    arcface_metric_csv = (
        project_root
        / "outputs/dolg_selected_loss_10ep/openset_hard/dolg_arcface_per_cow_metrics.csv"
    )
    supcon_metric_csv = (
        project_root
        / "outputs/dolg_selected_loss_10ep/openset_hard/"
        "dolg_arcface_supcon_lam01_per_cow_metrics.csv"
    )

    distance_arcface = read_unique_csv(
        arcface_distance_csv, ["cow_id", "hard_gap"]
    ).rename(columns={"hard_gap": "hard_gap_arcface"})
    distance_supcon = read_unique_csv(
        supcon_distance_csv, ["cow_id", "hard_gap"]
    ).rename(columns={"hard_gap": "hard_gap_supcon"})
    metric_arcface = read_unique_csv(
        arcface_metric_csv, ["cow_id", "recall_normal", "f1_normal"]
    ).rename(
        columns={
            "recall_normal": "recall_arcface",
            "f1_normal": "f1_arcface",
        }
    )
    metric_supcon = read_unique_csv(
        supcon_metric_csv, ["cow_id", "recall_normal", "f1_normal"]
    ).rename(
        columns={
            "recall_normal": "recall_supcon",
            "f1_normal": "f1_supcon",
        }
    )

    data = distance_arcface.merge(
        distance_supcon, on="cow_id", how="inner", validate="one_to_one"
    )
    data = data.merge(
        metric_arcface, on="cow_id", how="inner", validate="one_to_one"
    )
    data = data.merge(
        metric_supcon, on="cow_id", how="inner", validate="one_to_one"
    )

    source_counts = [
        len(distance_arcface),
        len(distance_supcon),
        len(metric_arcface),
        len(metric_supcon),
    ]
    if len(set(source_counts + [len(data)])) != 1:
        raise ValueError(
            f"cow_id未完全匹配。四个源文件及合并后行数分别为：{source_counts + [len(data)]}"
        )

    data["hard_gap_improvement"] = (
        data["hard_gap_supcon"] - data["hard_gap_arcface"]
    )
    data["recall_improvement"] = (
        data["recall_supcon"] - data["recall_arcface"]
    )
    data["f1_improvement"] = data["f1_supcon"] - data["f1_arcface"]

    x = data["hard_gap_improvement"].to_numpy(dtype=float)
    recall_statistics = analyze_metric(
        x,
        data["recall_improvement"].to_numpy(dtype=float),
        "Hard-set normal-class Recall improvement",
        RANDOM_SEED,
    )
    f1_statistics = analyze_metric(
        x,
        data["f1_improvement"].to_numpy(dtype=float),
        "Hard-set normal-class F1 improvement",
        RANDOM_SEED + 1,
    )

    data.to_csv(
        output_dir / "dolg_supcon_gap_recall_f1_per_cow.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([recall_statistics, f1_statistics]).to_csv(
        output_dir / "dolg_supcon_gap_recall_f1_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 350,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.1))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.19, top=0.80, wspace=0.27)
    fig.suptitle(
        "Association between Hard-gap improvement and cleaning-performance improvement",
        fontsize=16,
        y=0.94,
    )

    draw_panel(
        axes[0],
        data,
        "recall_improvement",
        "(a) Normal-class Recall improvement",
        "Improvement in Hard-set normal-class Recall\n"
        "(DOLG+ArcFace+SupCon − DOLG+ArcFace)",
        "#E6863B",
        recall_statistics,
    )
    draw_panel(
        axes[1],
        data,
        "f1_improvement",
        "(b) Normal-class F1 improvement",
        "Improvement in Hard-set normal-class F1\n"
        "(DOLG+ArcFace+SupCon − DOLG+ArcFace)",
        "#2878B5",
        f1_statistics,
    )

    fig.text(
        0.5,
        0.055,
        "Each point represents one cow; shaded bands show 95% confidence intervals for the fitted regression lines.",
        ha="center",
        fontsize=9.7,
        color="#4A4A4A",
    )

    figure_png = output_dir / "dolg_supcon_gap_recall_f1_correlation.png"
    figure_pdf = output_dir / "dolg_supcon_gap_recall_f1_correlation.pdf"
    fig.savefig(figure_png, bbox_inches="tight", facecolor="white")
    fig.savefig(figure_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for result in (recall_statistics, f1_statistics):
        print("=" * 72)
        print(result["metric"])
        print(f"Pearson r = {result['pearson_r']:.6f}")
        print(
            "Bootstrap 95% CI = "
            f"[{result['pearson_ci_low']:.6f}, {result['pearson_ci_high']:.6f}]"
        )
        print(f"p = {result['pearson_p']:.6g}")
        print(f"R^2 = {result['r_squared']:.6f}")
        print(f"n = {result['n_cows']} cows")

    print("=" * 72)
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    main()
