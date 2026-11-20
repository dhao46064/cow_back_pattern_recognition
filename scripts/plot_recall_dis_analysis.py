# -*- coding: utf-8 -*-
"""
只绘制：
Hard gap改善值 与 normal-class Recall改善值的Pearson线性相关图
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


PROJECT_ROOT = Path(r"D:\code_project\cow_recognition")

INPUT_CSV = (
    PROJECT_ROOT
    / "paper_materials"
    / "cow_macro_distance"
    / "supcon_linear_correlation"
    / "supcon_linear_correlation_per_cow.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "paper_materials"
    / "cow_macro_distance"
    / "supcon_linear_correlation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def bootstrap_pearson_ci(x, y, n_bootstrap=10000, seed=42):
    """以牛只为单位计算Pearson r的Bootstrap 95%置信区间。"""
    rng = np.random.default_rng(seed)
    n = len(x)
    bootstrap_r = []

    for _ in range(n_bootstrap):
        index = rng.integers(0, n, size=n)
        bx = x[index]
        by = y[index]

        if np.std(bx) == 0 or np.std(by) == 0:
            continue

        bootstrap_r.append(
            stats.pearsonr(bx, by).statistic
        )

    return np.percentile(bootstrap_r, [2.5, 97.5])


def format_p(p):
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"找不到逐牛结果：{INPUT_CSV}")

    data = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    required_columns = [
        "cow_id",
        "delta_hard_gap",
        "delta_recall",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            f"CSV缺少列：{missing_columns}\n"
            f"当前列名：{list(data.columns)}"
        )

    # 每头牛作为一个独立统计单位
    valid = (
        data[required_columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    x = valid["delta_hard_gap"].to_numpy(dtype=float)
    y = valid["delta_recall"].to_numpy(dtype=float)
    n = len(valid)

    if n < 3:
        raise ValueError(f"有效牛只数量不足：n={n}")

    # Pearson线性相关
    pearson = stats.pearsonr(x, y)
    pearson_r = float(pearson.statistic)
    p_value = float(pearson.pvalue)

    ci_low, ci_high = bootstrap_pearson_ci(x, y)

    # 线性回归
    regression = stats.linregress(x, y)
    slope = float(regression.slope)
    intercept = float(regression.intercept)

    x_line = np.linspace(x.min(), x.max(), 300)
    y_line = intercept + slope * x_line

    # 回归均值的95%置信区间
    fitted_y = intercept + slope * x
    residuals = y - fitted_y

    residual_standard_error = np.sqrt(
        np.sum(residuals ** 2) / (n - 2)
    )

    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean) ** 2)

    mean_prediction_se = residual_standard_error * np.sqrt(
        1 / n + ((x_line - x_mean) ** 2) / sxx
    )

    critical_t = stats.t.ppf(0.975, df=n - 2)

    regression_ci_low = (
        y_line - critical_t * mean_prediction_se
    )
    regression_ci_high = (
        y_line + critical_t * mean_prediction_se
    )

    # ========================================================
    # 绘图
    # ========================================================
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "normal",
    })

    fig, ax = plt.subplots(figsize=(8.5, 7))

    ax.scatter(
        x,
        y,
        s=48,
        color="#e67e22",
        alpha=0.72,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    ax.fill_between(
        x_line,
        regression_ci_low,
        regression_ci_high,
        color="#f4a261",
        alpha=0.25,
        linewidth=0,
        label="95% CI",
        zorder=1,
    )

    ax.plot(
        x_line,
        y_line,
        color="black",
        linewidth=2.4,
        label="Linear regression",
        zorder=4,
    )

    # 改善值为0的参考线
    ax.axvline(
        0,
        color="#777777",
        linestyle="--",
        linewidth=1.2,
        zorder=2,
    )

    ax.axhline(
        0,
        color="#777777",
        linestyle="--",
        linewidth=1.2,
        zorder=2,
    )

    annotation = (
        f"Pearson r = {pearson_r:.3f}\n"
        f"95% CI [{ci_low:.3f}, {ci_high:.3f}]\n"
        f"{format_p(p_value)}\n"
        f"n = {n} cows"
    )

    ax.text(
        0.035,
        0.965,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#bdbdbd",
            "alpha": 0.95,
        },
    )

    ax.set_title(
        "Association between separation-margin improvement\n"
        "and normal-class recall improvement",
        fontsize=16,
        pad=14,
    )

    ax.set_xlabel(
        "Improvement in Hard gap\n"
        "(DOLG+SupCon − DOLG+ArcFace)",
        fontsize=12,
    )

    ax.set_ylabel(
        "Improvement in normal-class Recall\n"
        "(DOLG+SupCon − DOLG+ArcFace)",
        fontsize=12,
    )

    ax.tick_params(axis="both", labelsize=11)
    ax.grid(alpha=0.22)
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()

    png_path = OUTPUT_DIR / "supcon_gap_recall_linear.png"
    pdf_path = OUTPUT_DIR / "supcon_gap_recall_linear.pdf"
    csv_path = OUTPUT_DIR / "supcon_gap_recall_linear_statistics.csv"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    result = pd.DataFrame([{
        "analysis": "delta_hard_gap_vs_delta_recall",
        "n_cows": n,
        "pearson_r": pearson_r,
        "pearson_ci_low": ci_low,
        "pearson_ci_high": ci_high,
        "p_value": p_value,
        "linear_slope": slope,
        "linear_intercept": intercept,
    }])

    result.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 65)
    print("Hard gap改善与Recall改善的线性相关分析")
    print("=" * 65)
    print(f"Pearson r = {pearson_r:.6f}")
    print(f"95% CI = [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"p = {p_value:.8g}")
    print(f"n = {n} cows")
    print(f"回归斜率 = {slope:.6f}")
    print()
    print(f"图片：{png_path}")
    print(f"PDF：{pdf_path}")
    print(f"统计结果：{csv_path}")


if __name__ == "__main__":
    main()