# -*- coding: utf-8 -*-
"""重新绘制SupCon逐牛配对差异图和距离—性能相关图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon


# =========================================================
# 1. 路径
# =========================================================

ROOT = Path(
    r"D:\code_project\cow_recognition"
)

DATA_DIR = (
    ROOT
    / "paper_materials"
    / "cow_macro_distance"
    / "supcon_statistics"
)

PAIR_CSV = (
    DATA_DIR
    / "supcon_per_cow_improvements.csv"
)

CORRELATION_CSV = (
    DATA_DIR
    / "dolg_supcon_distance_performance_improvements.csv"
)

FOREST_PNG = (
    DATA_DIR
    / "supcon_paired_differences_revised.png"
)

FOREST_PDF = (
    DATA_DIR
    / "supcon_paired_differences_revised.pdf"
)

CORRELATION_PNG = (
    DATA_DIR
    / "dolg_gap_recall_correlation_revised.png"
)

CORRELATION_PDF = (
    DATA_DIR
    / "dolg_gap_recall_correlation_revised.pdf"
)

STATISTICS_CSV = (
    DATA_DIR
    / "supcon_paired_statistics_revised.csv"
)

CORRELATION_RESULT_CSV = (
    DATA_DIR
    / "dolg_gap_recall_correlation_revised.csv"
)

SEED = 42
BOOTSTRAP_REPEATS = 10000

R50_COLOR = "#2878B5"
DOLG_COLOR = "#D95F02"


COMPARISON_ORDER = [
    "R50: SupCon vs ArcFace",
    "R50: SupCon vs Center",
    "DOLG: SupCon vs ArcFace",
    "DOLG: SupCon vs Center",
]

SHORT_LABELS = {
    "R50: SupCon vs ArcFace":
        "R50: SupCon − ArcFace",

    "R50: SupCon vs Center":
        "R50: SupCon − Center",

    "DOLG: SupCon vs ArcFace":
        "DOLG: SupCon − ArcFace",

    "DOLG: SupCon vs Center":
        "DOLG: SupCon − Center",
}


# =========================================================
# 2. Bootstrap
# =========================================================

def bootstrap_mean_ci(values, seed):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(
        BOOTSTRAP_REPEATS,
        dtype=np.float64,
    )

    for index in range(BOOTSTRAP_REPEATS):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        bootstrap_means[index] = (
            sample.mean()
        )

    low, high = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return (
        float(values.mean()),
        float(low),
        float(high),
    )


def bootstrap_spearman_ci(x, y, seed):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    rng = np.random.default_rng(seed)
    correlations = []

    for _ in range(BOOTSTRAP_REPEATS):
        indices = rng.integers(
            0,
            len(x),
            size=len(x),
        )

        sampled_x = x[indices]
        sampled_y = y[indices]

        # 某次重采样可能全部相同
        if (
            np.unique(sampled_x).size < 2
            or np.unique(sampled_y).size < 2
        ):
            continue

        result = spearmanr(
            sampled_x,
            sampled_y,
        )

        if np.isfinite(result.statistic):
            correlations.append(
                float(result.statistic)
            )

    low, high = np.quantile(
        correlations,
        [0.025, 0.975],
    )

    return float(low), float(high)


# =========================================================
# 3. 效应量与Holm校正
# =========================================================

def rank_biserial_effect(differences):
    differences = np.asarray(
        differences,
        dtype=np.float64,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    differences = differences[
        differences != 0
    ]

    if len(differences) == 0:
        return 0.0

    ranks = rankdata(
        np.abs(differences)
    )

    positive = ranks[
        differences > 0
    ].sum()

    negative = ranks[
        differences < 0
    ].sum()

    return float(
        (positive - negative)
        / ranks.sum()
    )


def holm_adjust(p_values):
    p_values = np.asarray(
        p_values,
        dtype=np.float64,
    )

    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(count)

    previous = 0.0

    for rank_index, original_index in enumerate(order):
        corrected = min(
            1.0,
            p_values[original_index]
            * (count - rank_index),
        )

        corrected = max(
            corrected,
            previous,
        )

        adjusted[original_index] = corrected
        previous = corrected

    return adjusted


# =========================================================
# 4. 重新计算SupCon配对统计
# =========================================================

def calculate_paired_statistics():
    if not PAIR_CSV.is_file():
        raise FileNotFoundError(
            f"找不到文件：{PAIR_CSV}"
        )

    data = pd.read_csv(
        PAIR_CSV,
        dtype={"cow_id": str},
    )

    required = {
        "cow_id",
        "metric",
        "comparison",
        "supcon_value",
        "reference_value",
    }

    missing = required - set(data.columns)

    if missing:
        raise KeyError(
            f"配对CSV缺少字段：{sorted(missing)}"
        )

    rows = []

    for metric_index, metric in enumerate(
        ["hard_na", "hard_gap"]
    ):
        metric_rows = []

        for comparison_index, comparison in enumerate(
            COMPARISON_ORDER
        ):
            group = data[
                (data["metric"] == metric)
                & (data["comparison"] == comparison)
            ].copy()

            group = group.dropna(
                subset=[
                    "supcon_value",
                    "reference_value",
                ]
            )

            differences = (
                group["supcon_value"].to_numpy(float)
                - group["reference_value"].to_numpy(float)
            )

            mean, ci_low, ci_high = (
                bootstrap_mean_ci(
                    differences,
                    seed=(
                        SEED
                        + metric_index * 100
                        + comparison_index
                    ),
                )
            )

            test = wilcoxon(
                group["supcon_value"],
                group["reference_value"],
                alternative="two-sided",
                zero_method="wilcox",
            )

            improved = int(
                np.sum(differences > 0)
            )

            worsened = int(
                np.sum(differences < 0)
            )

            metric_rows.append({
                "metric": metric,
                "comparison": comparison,
                "cow_count": len(group),

                "reference_mean":
                    float(
                        group[
                            "reference_value"
                        ].mean()
                    ),

                "supcon_mean":
                    float(
                        group[
                            "supcon_value"
                        ].mean()
                    ),

                "mean_difference": mean,
                "ci95_low": ci_low,
                "ci95_high": ci_high,

                "median_difference":
                    float(
                        np.median(differences)
                    ),

                "improved_cows": improved,

                "improved_percent":
                    improved
                    / len(group)
                    * 100,

                "worsened_cows": worsened,

                "wilcoxon_p_raw":
                    float(test.pvalue),

                "rank_biserial":
                    rank_biserial_effect(
                        differences
                    ),
            })

        adjusted = holm_adjust(
            [
                row["wilcoxon_p_raw"]
                for row in metric_rows
            ]
        )

        for row, corrected_p in zip(
            metric_rows,
            adjusted,
        ):
            row["wilcoxon_p_holm"] = (
                float(corrected_p)
            )

            rows.append(row)

    result = pd.DataFrame(rows)

    result.to_csv(
        STATISTICS_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    return result


# =========================================================
# 5. 两面板森林图
# =========================================================

def p_text(p_value):
    if p_value < 0.001:
        return "pHolm < 0.001"
    return f"pHolm = {p_value:.3f}"


def plot_forest(statistics):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6.4),
    )

    panel_settings = [
        {
            "metric": "hard_na",
            "title": "(a) Hard abnormal separation",
            "xlabel": (
                "Mean paired difference in Hard N-A\n"
                "(SupCon − reference)"
            ),
        },
        {
            "metric": "hard_gap",
            "title": "(b) Absolute separation margin",
            "xlabel": (
                "Mean paired difference in Hard gap\n"
                "(SupCon − reference)"
            ),
        },
    ]

    for axis, setting in zip(
        axes,
        panel_settings,
    ):
        panel = (
            statistics[
                statistics["metric"]
                == setting["metric"]
            ]
            .set_index("comparison")
            .loc[COMPARISON_ORDER]
            .reset_index()
        )

        y_positions = np.arange(
            len(panel)
        )[::-1]

        axis.axvline(
            0,
            color="#555555",
            linestyle="--",
            linewidth=1.2,
        )

        maximum_ci = panel[
            "ci95_high"
        ].max()

        minimum_ci = min(
            0,
            panel["ci95_low"].min(),
        )

        data_range = (
            maximum_ci - minimum_ci
        )

        text_offset = max(
            data_range * 0.035,
            0.002,
        )

        for row_index, row in panel.iterrows():
            comparison = row["comparison"]

            color = (
                R50_COLOR
                if comparison.startswith("R50")
                else DOLG_COLOR
            )

            y = y_positions[row_index]
            mean = row["mean_difference"]

            left_error = (
                mean - row["ci95_low"]
            )

            right_error = (
                row["ci95_high"] - mean
            )

            axis.errorbar(
                mean,
                y,
                xerr=[
                    [left_error],
                    [right_error],
                ],
                fmt="o",
                markersize=8,
                color=color,
                ecolor=color,
                elinewidth=2.2,
                capsize=5,
                zorder=3,
            )

            significance = (
                "significant"
                if row["wilcoxon_p_holm"] < 0.05
                else "not significant"
            )

            annotation = (
                f"{p_text(row['wilcoxon_p_holm'])}\n"
                f"r={row['rank_biserial']:.2f}, "
                f"improved={row['improved_percent']:.1f}%\n"
                f"{significance}"
            )

            axis.text(
                row["ci95_high"] + text_offset,
                y,
                annotation,
                va="center",
                fontsize=8.5,
                color=(
                    "#222222"
                    if significance == "significant"
                    else "#777777"
                ),
            )

        axis.set_yticks(
            y_positions
        )

        axis.set_yticklabels(
            [
                SHORT_LABELS[item]
                for item in COMPARISON_ORDER
            ],
            fontsize=9.5,
        )

        axis.set_xlabel(
            setting["xlabel"],
            fontsize=10.5,
        )

        axis.set_title(
            setting["title"],
            fontsize=13,
            pad=12,
        )

        axis.grid(
            axis="x",
            color="#DDDDDD",
            linewidth=0.7,
            alpha=0.75,
        )

        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        # 为右侧统计文字保留空间
        axis.set_xlim(
            minimum_ci - data_range * 0.08,
            maximum_ci + data_range * 0.75,
        )

    figure.suptitle(
        "Cow-level paired differences between "
        "SupCon and comparison models",
        fontsize=15,
        y=0.98,
    )

    figure.text(
        0.5,
        0.015,
        "Points show mean paired differences; "
        "error bars show cow-level bootstrap 95% CIs. "
        "P values are from paired Wilcoxon tests with Holm correction.",
        ha="center",
        fontsize=9,
        color="#555555",
    )

    figure.subplots_adjust(
        left=0.17,
        right=0.98,
        top=0.86,
        bottom=0.18,
        wspace=0.32,
    )

    figure.savefig(
        FOREST_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        FOREST_PDF,
        bbox_inches="tight",
    )

    plt.close(figure)


# =========================================================
# 6. 相关图
# =========================================================

def plot_correlation():
    if not CORRELATION_CSV.is_file():
        raise FileNotFoundError(
            f"找不到文件：{CORRELATION_CSV}"
        )

    data = pd.read_csv(
        CORRELATION_CSV,
        dtype={"cow_id": str},
    )

    required = {
        "cow_id",
        "delta_hard_gap",
        "delta_recall",
    }

    missing = required - set(data.columns)

    if missing:
        raise KeyError(
            f"相关分析CSV缺少字段：{sorted(missing)}"
        )

    data = data.dropna(
        subset=[
            "delta_hard_gap",
            "delta_recall",
        ]
    ).copy()

    x = data[
        "delta_hard_gap"
    ].to_numpy(dtype=float)

    y = data[
        "delta_recall"
    ].to_numpy(dtype=float)

    result = spearmanr(x, y)

    rho = float(result.statistic)
    p_value = float(result.pvalue)

    rho_low, rho_high = (
        bootstrap_spearman_ci(
            x,
            y,
            seed=SEED,
        )
    )

    # 四象限数量
    q1 = int(
        np.sum((x > 0) & (y > 0))
    )
    q2 = int(
        np.sum((x < 0) & (y > 0))
    )
    q3 = int(
        np.sum((x < 0) & (y < 0))
    )
    q4 = int(
        np.sum((x > 0) & (y < 0))
    )

    both_improved_or_equal = int(
        np.sum((x > 0) & (y >= 0))
    )

    correlation_summary = pd.DataFrame([
        {
            "analysis":
                "DOLG SupCon vs ArcFace",

            "cow_count":
                len(data),

            "spearman_rho":
                rho,

            "rho_ci95_low":
                rho_low,

            "rho_ci95_high":
                rho_high,

            "spearman_p":
                p_value,

            "gap_positive_recall_positive":
                q1,

            "gap_negative_recall_positive":
                q2,

            "gap_negative_recall_negative":
                q3,

            "gap_positive_recall_negative":
                q4,

            "gap_positive_recall_nonnegative":
                both_improved_or_equal,

            "gap_positive_recall_nonnegative_percent":
                both_improved_or_equal
                / len(data)
                * 100,
        }
    ])

    correlation_summary.to_csv(
        CORRELATION_RESULT_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    figure, axis = plt.subplots(
        figsize=(8.2, 7.0)
    )

    x_min = min(
        x.min(),
        -0.01,
    )

    x_max = max(
        x.max(),
        0.01,
    )

    y_min = min(
        y.min(),
        -0.01,
    )

    y_max = max(
        y.max(),
        0.01,
    )

    # 四象限浅色背景
    axis.axvspan(
        0,
        x_max,
        ymin=(
            (0 - y_min)
            / (y_max - y_min)
        ),
        ymax=1,
        color="#D9F0D3",
        alpha=0.28,
    )

    axis.axvspan(
        x_min,
        0,
        ymin=0,
        ymax=(
            (0 - y_min)
            / (y_max - y_min)
        ),
        color="#FDD0A2",
        alpha=0.20,
    )

    axis.axvline(
        0,
        color="#888888",
        linestyle="--",
        linewidth=1.1,
    )

    axis.axhline(
        0,
        color="#888888",
        linestyle="--",
        linewidth=1.1,
    )

    axis.scatter(
        x,
        y,
        s=34,
        color=DOLG_COLOR,
        alpha=0.68,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )

    # 非参数趋势：按Hard gap改善量分成8个分位组
    # 每组绘制横纵坐标中位数，不再使用线性回归线
    trend_frame = pd.DataFrame({
        "x": x,
        "y": y,
    })

    trend_frame["bin"] = pd.qcut(
        trend_frame["x"],
        q=8,
        duplicates="drop",
    )

    trend = (
        trend_frame
        .groupby(
            "bin",
            observed=True,
        )
        .agg(
            x_median=("x", "median"),
            y_median=("y", "median"),
        )
        .sort_values("x_median")
    )

    axis.plot(
        trend["x_median"],
        trend["y_median"],
        color="#222222",
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="Quantile-bin median trend",
        zorder=4,
    )

    if p_value < 0.001:
        p_label = "p < 0.001"
    else:
        p_label = f"p = {p_value:.3f}"

    statistics_text = (
        f"Spearman rho = {rho:.3f}\n"
        f"95% CI [{rho_low:.3f}, {rho_high:.3f}]\n"
        f"{p_label}\n"
        f"n = {len(data)} cows"
    )

    axis.text(
        0.03,
        0.97,
        statistics_text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#BBBBBB",
            "alpha": 0.93,
        },
    )

    # 象限解释
    axis.text(
        0.98,
        0.97,
        "Gap improved\nRecall improved",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#3A7D44",
    )

    axis.text(
        0.02,
        0.03,
        "Gap worsened\nRecall worsened",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#A34A00",
    )

    axis.set_xlabel(
        "Improvement in Hard gap\n"
        "(DOLG+SupCon − DOLG+ArcFace)",
        fontsize=11,
    )

    axis.set_ylabel(
        "Improvement in normal-class recall on the Hard set\n"
        "(DOLG+SupCon − DOLG+ArcFace)",
        fontsize=11,
    )

    axis.set_title(
        "Association between separation-margin improvement "
        "and normal-class recall improvement",
        fontsize=13,
        pad=12,
    )

    axis.grid(
        color="#DDDDDD",
        linewidth=0.7,
        alpha=0.65,
    )

    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.legend(
        loc="lower right",
        frameon=False,
        fontsize=9,
    )

    figure.tight_layout()

    figure.savefig(
        CORRELATION_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        CORRELATION_PDF,
        bbox_inches="tight",
    )

    plt.close(figure)

    return correlation_summary


# =========================================================
# 7. 主函数
# =========================================================

def main():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    statistics = (
        calculate_paired_statistics()
    )

    plot_forest(
        statistics
    )

    correlation_summary = (
        plot_correlation()
    )

    print()
    print("=" * 90)
    print("SupCon配对统计")
    print("=" * 90)

    show_columns = [
        "metric",
        "comparison",
        "cow_count",
        "mean_difference",
        "ci95_low",
        "ci95_high",
        "improved_percent",
        "rank_biserial",
        "wilcoxon_p_holm",
    ]

    print(
        statistics[
            show_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print("DOLG距离改善与Recall改善相关性")
    print("=" * 90)

    print(
        correlation_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print(f"森林图PNG：{FOREST_PNG}")
    print(f"森林图PDF：{FOREST_PDF}")
    print(f"相关图PNG：{CORRELATION_PNG}")
    print(f"相关图PDF：{CORRELATION_PDF}")
    print("=" * 90)


if __name__ == "__main__":
    main()