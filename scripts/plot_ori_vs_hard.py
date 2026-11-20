# -*- coding: utf-8 -*-
"""ResNet50+ArcFace基线：Ori与Hard开放集逐牛配对距离分析。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# =========================================================
# 1. 路径及参数
# =========================================================

DATA_DIR = Path(
    r"D:\code_project\cow_recognition"
    r"\paper_materials\cow_macro_distance"
)

INPUT_CSV = DATA_DIR / "r50_arcface_per_cow.csv"

OUTPUT_PNG = DATA_DIR / "baseline_ori_vs_hard_paired.png"
OUTPUT_PDF = DATA_DIR / "baseline_ori_vs_hard_paired.pdf"
OUTPUT_SUMMARY = DATA_DIR / "baseline_ori_vs_hard_summary.csv"

SEED = 42
BOOTSTRAP_REPEATS = 10000

ORI_COLOR = "#4C78A8"
HARD_COLOR = "#E45756"


# =========================================================
# 2. Bootstrap置信区间
# =========================================================

def bootstrap_mean_ci(values, seed=42):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

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
        bootstrap_means[index] = sample.mean()

    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return (
        float(values.mean()),
        float(ci_low),
        float(ci_high),
    )


def paired_bootstrap_drop_ci(ori, hard, seed=42):
    """以牛只为单位对Ori-Hard差值进行配对bootstrap。"""

    ori = np.asarray(ori, dtype=np.float64)
    hard = np.asarray(hard, dtype=np.float64)
    difference = ori - hard

    return bootstrap_mean_ci(
        difference,
        seed=seed,
    )


# =========================================================
# 3. 读取数据
# =========================================================

def load_data():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"找不到逐牛结果：{INPUT_CSV}"
        )

    frame = pd.read_csv(
        INPUT_CSV,
        dtype={"cow_id": str},
    )

    required_columns = {
        "cow_id",
        "ori_na",
        "hard_na",
    }

    missing = required_columns - set(frame.columns)

    if missing:
        raise KeyError(
            f"CSV缺少字段：{sorted(missing)}"
        )

    frame["ori_na"] = pd.to_numeric(
        frame["ori_na"],
        errors="coerce",
    )

    frame["hard_na"] = pd.to_numeric(
        frame["hard_na"],
        errors="coerce",
    )

    # 配对分析只能使用同时具有Ori和Hard结果的牛只
    frame = frame.dropna(
        subset=["ori_na", "hard_na"]
    ).copy()

    frame["distance_drop"] = (
        frame["ori_na"] - frame["hard_na"]
    )

    if frame.empty:
        raise RuntimeError(
            "没有同时具有Ori N-A和Hard N-A的牛只"
        )

    return frame


# =========================================================
# 4. 统计分析
# =========================================================

def calculate_statistics(frame):
    ori = frame["ori_na"].to_numpy(dtype=float)
    hard = frame["hard_na"].to_numpy(dtype=float)
    difference = ori - hard

    ori_mean, ori_low, ori_high = bootstrap_mean_ci(
        ori,
        seed=SEED,
    )

    hard_mean, hard_low, hard_high = bootstrap_mean_ci(
        hard,
        seed=SEED + 1,
    )

    drop_mean, drop_low, drop_high = (
        paired_bootstrap_drop_ci(
            ori,
            hard,
            seed=SEED + 2,
        )
    )

    # 双侧配对Wilcoxon检验
    test = wilcoxon(
        ori,
        hard,
        alternative="two-sided",
        zero_method="wilcox",
    )

    hard_lower_count = int(
        np.sum(hard < ori)
    )

    hard_equal_count = int(
        np.sum(np.isclose(hard, ori))
    )

    hard_higher_count = int(
        np.sum(hard > ori)
    )

    relative_drop = (
        drop_mean / ori_mean * 100
        if ori_mean != 0
        else float("nan")
    )

    return {
        "cow_count": len(frame),

        "ori_mean": ori_mean,
        "ori_ci95_low": ori_low,
        "ori_ci95_high": ori_high,

        "hard_mean": hard_mean,
        "hard_ci95_low": hard_low,
        "hard_ci95_high": hard_high,

        "drop_mean": drop_mean,
        "drop_ci95_low": drop_low,
        "drop_ci95_high": drop_high,
        "relative_drop_percent": relative_drop,

        "hard_lower_count": hard_lower_count,
        "hard_equal_count": hard_equal_count,
        "hard_higher_count": hard_higher_count,

        "wilcoxon_statistic": float(test.statistic),
        "wilcoxon_p": float(test.pvalue),
    }


# =========================================================
# 5. 绘图
# =========================================================

def plot_paired_comparison(frame, statistics):
    ori = frame["ori_na"].to_numpy(dtype=float)
    hard = frame["hard_na"].to_numpy(dtype=float)

    figure, axis = plt.subplots(
        figsize=(7.6, 6.8)
    )

    # -----------------------------------------------------
    # 背景小提琴：表示逐牛距离分布
    # -----------------------------------------------------

    violin = axis.violinplot(
        [ori, hard],
        positions=[0, 1],
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, color in zip(
        violin["bodies"],
        [ORI_COLOR, HARD_COLOR],
    ):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.18)
        body.set_linewidth(1.0)

    # -----------------------------------------------------
    # 每头牛一条配对连线
    # -----------------------------------------------------

    rng = np.random.default_rng(SEED)

    # 同一头牛在两侧使用相同偏移，保持配对关系
    jitter = rng.uniform(
        -0.10,
        0.10,
        size=len(frame),
    )

    for index in range(len(frame)):
        color = (
            "#808080"
            if hard[index] <= ori[index]
            else "#D95F02"
        )

        axis.plot(
            [0 + jitter[index], 1 + jitter[index]],
            [ori[index], hard[index]],
            color=color,
            alpha=0.13,
            linewidth=0.7,
            zorder=1,
        )

    axis.scatter(
        np.zeros(len(ori)) + jitter,
        ori,
        s=13,
        color=ORI_COLOR,
        alpha=0.48,
        edgecolors="none",
        zorder=2,
    )

    axis.scatter(
        np.ones(len(hard)) + jitter,
        hard,
        s=13,
        color=HARD_COLOR,
        alpha=0.48,
        edgecolors="none",
        zorder=2,
    )

    # -----------------------------------------------------
    # 箱线图
    # -----------------------------------------------------

    boxplot = axis.boxplot(
        [ori, hard],
        positions=[0, 1],
        widths=0.20,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={
            "color": "#222222",
            "linewidth": 1.5,
        },
        whiskerprops={
            "color": "#444444",
            "linewidth": 1.1,
        },
        capprops={
            "color": "#444444",
            "linewidth": 1.1,
        },
    )

    for box, color in zip(
        boxplot["boxes"],
        [ORI_COLOR, HARD_COLOR],
    ):
        box.set_facecolor("white")
        box.set_edgecolor(color)
        box.set_linewidth(1.5)

    # -----------------------------------------------------
    # Macro均值及bootstrap 95% CI
    # -----------------------------------------------------

    means = [
        statistics["ori_mean"],
        statistics["hard_mean"],
    ]

    ci_lows = [
        statistics["ori_ci95_low"],
        statistics["hard_ci95_low"],
    ]

    ci_highs = [
        statistics["ori_ci95_high"],
        statistics["hard_ci95_high"],
    ]

    lower_errors = [
        means[i] - ci_lows[i]
        for i in range(2)
    ]

    upper_errors = [
        ci_highs[i] - means[i]
        for i in range(2)
    ]

    axis.errorbar(
        [0, 1],
        means,
        yerr=[
            lower_errors,
            upper_errors,
        ],
        fmt="D",
        markersize=7,
        markerfacecolor="#111111",
        markeredgecolor="#111111",
        ecolor="#111111",
        elinewidth=1.7,
        capsize=5,
        zorder=5,
        label="Cow-macro mean and 95% CI",
    )

    # 连接两个macro均值
    axis.plot(
        [0, 1],
        means,
        color="#111111",
        linewidth=2.0,
        linestyle="--",
        zorder=4,
    )

    # -----------------------------------------------------
    # 统计结果标注
    # -----------------------------------------------------

    p_value = statistics["wilcoxon_p"]

    if p_value < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {p_value:.3f}"

    annotation = (
        f"Mean drop = {statistics['drop_mean']:.4f}\n"
        f"95% CI [{statistics['drop_ci95_low']:.4f}, "
        f"{statistics['drop_ci95_high']:.4f}]\n"
        f"Relative drop = "
        f"{statistics['relative_drop_percent']:.1f}%\n"
        f"Paired Wilcoxon {p_text}\n"
        f"Hard < Ori: "
        f"{statistics['hard_lower_count']}/"
        f"{statistics['cow_count']} cows"
    )

    axis.text(
        0.98,
        0.97,
        annotation,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#BBBBBB",
            "alpha": 0.92,
        },
    )

    axis.set_title(
        "ResNet50 + ArcFace: Ori vs Hard open-set distance",
        fontsize=14,
        pad=14,
    )

    axis.set_ylabel(
        "Normal-abnormal cosine distance",
        fontsize=11,
    )

    axis.set_xticks([0, 1])
    axis.set_xticklabels(
        ["Original open set", "Hard open set"],
        fontsize=11,
    )

    axis.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.7,
        alpha=0.75,
    )

    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.legend(
        loc="lower left",
        frameon=False,
        fontsize=9,
    )

    figure.text(
        0.5,
        0.015,
        "Each point and connecting line represents one cow.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )

    figure.tight_layout(
        rect=[0, 0.04, 1, 1]
    )

    figure.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    plt.close(figure)


# =========================================================
# 6. 保存统计结果
# =========================================================

def save_summary(statistics):
    summary = pd.DataFrame([
        {
            "model": "ResNet50 + ArcFace",
            **statistics,
        }
    ])

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )


# =========================================================
# 7. 主函数
# =========================================================

def main():
    frame = load_data()
    statistics = calculate_statistics(frame)

    plot_paired_comparison(
        frame,
        statistics,
    )

    save_summary(statistics)

    print("=" * 80)
    print("ResNet50 + ArcFace：Ori与Hard开放集对比")
    print("=" * 80)

    print(f"配对牛只数量：{statistics['cow_count']}")

    print(
        f"Ori N-A："
        f"{statistics['ori_mean']:.6f} "
        f"[{statistics['ori_ci95_low']:.6f}, "
        f"{statistics['ori_ci95_high']:.6f}]"
    )

    print(
        f"Hard N-A："
        f"{statistics['hard_mean']:.6f} "
        f"[{statistics['hard_ci95_low']:.6f}, "
        f"{statistics['hard_ci95_high']:.6f}]"
    )

    print(
        f"平均下降："
        f"{statistics['drop_mean']:.6f} "
        f"[{statistics['drop_ci95_low']:.6f}, "
        f"{statistics['drop_ci95_high']:.6f}]"
    )

    print(
        f"相对下降："
        f"{statistics['relative_drop_percent']:.2f}%"
    )

    print(
        f"Hard距离更低的牛只："
        f"{statistics['hard_lower_count']}/"
        f"{statistics['cow_count']}"
    )

    print(
        f"Wilcoxon p值："
        f"{statistics['wilcoxon_p']:.10g}"
    )

    print("=" * 80)
    print(f"PNG：{OUTPUT_PNG}")
    print(f"PDF：{OUTPUT_PDF}")
    print(f"统计结果：{OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()