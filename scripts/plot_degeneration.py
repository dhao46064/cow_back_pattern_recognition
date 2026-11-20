# -*- coding: utf-8 -*-
"""
原始集到Hard集的Macro指标退化图（简洁论文版）

左图：Macro Precision退化
右图：Macro Recall退化
数值越小，模型在Hard集上的稳定性越好。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. 输出目录
# ============================================================
PROJECT_ROOT = Path(r"D:\code_project\cow_recognition")
OUTPUT_DIR = PROJECT_ROOT / "paper_materials" / "main_metric_degradation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "macro_metric_degradation_clean.png"
OUTPUT_PDF = OUTPUT_DIR / "macro_metric_degradation_clean.pdf"
OUTPUT_CSV = OUTPUT_DIR / "macro_metric_degradation_clean.csv"


# ============================================================
# 2. 已确认的Cow-Macro主指标，单位均为%
# ============================================================
data = [
    {
        "model": "ResNet50 + ArcFace",
        "group": "ResNet50",
        "loss": "ArcFace",
        "ori_precision": 99.03,
        "ori_recall": 98.96,
        "hard_precision": 98.07,
        "hard_recall": 90.64,
        "hard_f1": 94.18,
    },
    {
        "model": "ResNet50 + ArcFace + Center",
        "group": "ResNet50",
        "loss": "Center",
        "ori_precision": 99.14,
        "ori_recall": 99.29,
        "hard_precision": 98.27,
        "hard_recall": 92.37,
        "hard_f1": 95.20,
    },
    {
        "model": "ResNet50 + ArcFace + SupCon",
        "group": "ResNet50",
        "loss": "SupCon",
        "ori_precision": 99.18,
        "ori_recall": 98.89,
        "hard_precision": 98.56,
        "hard_recall": 92.47,
        "hard_f1": 95.39,
    },
    {
        "model": "DOLG + ArcFace",
        "group": "DOLG",
        "loss": "ArcFace",
        "ori_precision": 99.24,
        "ori_recall": 99.08,
        "hard_precision": 98.77,
        "hard_recall": 92.70,
        "hard_f1": 95.60,
    },
    {
        "model": "DOLG + ArcFace + Center",
        "group": "DOLG",
        "loss": "Center",
        "ori_precision": 99.23,
        "ori_recall": 99.26,
        "hard_precision": 98.53,
        "hard_recall": 93.00,
        "hard_f1": 95.66,
    },
    {
        "model": "DOLG + ArcFace + SupCon",
        "group": "DOLG",
        "loss": "SupCon",
        "ori_precision": 99.33,
        "ori_recall": 99.27,
        "hard_precision": 98.79,
        "hard_recall": 93.10,
        "hard_f1": 95.83,
    },
]

df = pd.DataFrame(data)

# 退化幅度：Original − Hard，单位为百分点
df["precision_drop"] = df["ori_precision"] - df["hard_precision"]
df["recall_drop"] = df["ori_recall"] - df["hard_recall"]

# 原始集F1与F1退化，仅用于导出统计表
df["ori_f1"] = (
    2 * df["ori_precision"] * df["ori_recall"]
    / (df["ori_precision"] + df["ori_recall"])
)
df["f1_drop"] = df["ori_f1"] - df["hard_f1"]

# 按Hard F1从低到高排列，最终模型位于最上方
df = df.sort_values("hard_f1", ascending=True).reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


# ============================================================
# 3. 绘图样式
# ============================================================
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 400,
    }
)

COLORS = {
    "ArcFace": "#8C8C8C",
    "Center": "#E69F00",
    "SupCon": "#2878B5",
}

MARKERS = {
    "ArcFace": "o",
    "Center": "s",
    "SupCon": "^",
}

y = np.arange(len(df))

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.2, 6.6),
    sharey=True,
    gridspec_kw={"wspace": 0.10},
)

plot_settings = [
    (
        axes[0],
        "precision_drop",
        "(a) Precision degradation",
        "Macro Precision degradation\n(percentage points; lower is better)",
    ),
    (
        axes[1],
        "recall_drop",
        "(b) Recall degradation",
        "Macro Recall degradation\n(percentage points; lower is better)",
    ),
]

for ax, metric, title, xlabel in plot_settings:
    # 每行浅色引导线
    for yi in y:
        ax.axhline(
            yi,
            color="#EEEEEE",
            linewidth=0.8,
            zorder=0,
        )

    # 从0延伸到指标值的线段
    for yi, row in df.iterrows():
        is_final = row["model"] == "DOLG + ArcFace + SupCon"

        ax.hlines(
            y=yi,
            xmin=0,
            xmax=row[metric],
            color=COLORS[row["loss"]],
            linewidth=3.4 if is_final else 2.2,
            alpha=1.0 if is_final else 0.70,
            zorder=1,
        )

        ax.scatter(
            row[metric],
            yi,
            s=145 if is_final else 90,
            marker=MARKERS[row["loss"]],
            color=COLORS[row["loss"]],
            edgecolor="white",
            linewidth=1.3,
            zorder=3,
        )

        ax.text(
            row[metric] + df[metric].max() * 0.025,
            yi,
            f'{row[metric]:.2f}',
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight="bold" if is_final else "normal",
            color="#1F1F1F",
        )

    ax.axvline(
        0,
        color="#444444",
        linewidth=1.0,
    )

    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel, labelpad=10)

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.7,
        alpha=0.28,
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(
        0,
        df[metric].max() * 1.22,
    )

# 模型标签
axes[0].set_yticks(y)
axes[0].set_yticklabels(df["model"])

# 最终模型标签加粗
for label, model in zip(axes[0].get_yticklabels(), df["model"]):
    if model == "DOLG + ArcFace + SupCon":
        label.set_fontweight("bold")
        label.set_color(COLORS["SupCon"])

# 右侧给出Hard F1，既展示退化，也保留最终性能
for yi, row in df.iterrows():
    is_final = row["model"] == "DOLG + ArcFace + SupCon"

    axes[1].text(
        1.02,
        yi,
        f'Hard F1 {row["hard_f1"]:.2f}%',
        transform=axes[1].get_yaxis_transform(),
        va="center",
        ha="left",
        fontsize=10,
        fontweight="bold" if is_final else "normal",
        color=COLORS["SupCon"] if is_final else "#555555",
        clip_on=False,
    )

# 分隔ResNet50和DOLG两组
group_change = None
for i in range(1, len(df)):
    if df.loc[i, "group"] != df.loc[i - 1, "group"]:
        group_change = i - 0.5
        break

if group_change is not None:
    for ax in axes:
        ax.axhline(
            group_change,
            color="#AAAAAA",
            linestyle=":",
            linewidth=1.2,
        )

# 图例
legend_handles = []
for loss_name in ["ArcFace", "Center", "SupCon"]:
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker=MARKERS[loss_name],
            color=COLORS[loss_name],
            linewidth=2.2,
            markersize=8,
            markerfacecolor=COLORS[loss_name],
            markeredgecolor="white",
            label=(
                "ArcFace"
                if loss_name == "ArcFace"
                else f"+ {loss_name}"
            ),
        )
    )

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.015),
    ncol=3,
    frameon=False,
)

fig.suptitle(
    "Macro-level degradation from the Original to Hard open set",
    fontsize=17,
    y=0.985,
)

fig.text(
    0.5,
    0.025,
    "Shorter lines indicate stronger robustness. "
    "Hard-set Macro F1 is reported on the right.",
    ha="center",
    fontsize=10.5,
    color="#555555",
)

fig.subplots_adjust(
    left=0.31,
    right=0.86,
    top=0.87,
    bottom=0.18,
)

fig.savefig(OUTPUT_PNG, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, bbox_inches="tight")
plt.close(fig)


print("绘图完成：")
print(f"PNG: {OUTPUT_PNG}")
print(f"PDF: {OUTPUT_PDF}")
print(f"CSV: {OUTPUT_CSV}")

print("\n指标退化结果：")
print(
    df[
        [
            "model",
            "precision_drop",
            "recall_drop",
            "f1_drop",
            "hard_f1",
        ]
    ].to_string(
        index=False,
        formatters={
            "precision_drop": "{:.2f}".format,
            "recall_drop": "{:.2f}".format,
            "f1_drop": "{:.2f}".format,
            "hard_f1": "{:.2f}".format,
        },
    )
)