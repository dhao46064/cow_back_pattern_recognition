# -*- coding: utf-8 -*-
"""直接读取已有CSV，重新绘制放大坐标的阈值稳定性图。"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import precision_recall_curve, auc


# ============================================================
# 路径
# ============================================================

DATA_DIR = Path(
    r"D:\code_project\cow_recognition"
    r"\paper_materials\threshold_stability\dolg_hard"
)

SWEEP_CSV = DATA_DIR / "threshold_sweep_metrics.csv"
SUMMARY_CSV = DATA_DIR / "threshold_model_summary.csv"
SCORES_CSV = DATA_DIR / "threshold_sample_scores.csv"

ORIGINAL_K = 1.0

MODELS = {
    "dolg_arcface": {
        "name": "DOLG+ArcFace",
        "color": "#7f7f7f",
    },
    "dolg_center": {
        "name": "DOLG+ArcFace+Center",
        "color": "#e67e22",
    },
    "dolg_supcon": {
        "name": "DOLG+ArcFace+SupCon",
        "color": "#2878b5",
    },
}


def configure_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def draw_f1_curve(sweep_df, summary_df):
    configure_style()

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # 放大原始阈值与最优阈值附近
    x_min = 0.70
    x_max = 2.30

    visible = sweep_df[
        sweep_df["std_scale_k"].between(x_min, x_max)
    ]

    # 根据当前显示范围自动确定纵轴
    y_data_min = visible["f1_normal_micro"].min()
    y_data_max = visible["f1_normal_micro"].max()

    y_min = np.floor(
        (y_data_min - 0.003) / 0.005
    ) * 0.005

    y_max = np.ceil(
        (y_data_max + 0.003) / 0.005
    ) * 0.005

    for model_key, config in MODELS.items():
        model_data = (
            sweep_df[
                sweep_df["model_key"] == model_key
            ]
            .sort_values("std_scale_k")
        )

        ax.plot(
            model_data["std_scale_k"],
            model_data["f1_normal_micro"],
            color=config["color"],
            linewidth=2.4,
            label=config["name"],
        )

        summary_row = summary_df[
            summary_df["model_key"] == model_key
        ]

        if not summary_row.empty:
            summary_row = summary_row.iloc[0]

            ax.scatter(
                summary_row["best_k"],
                summary_row["best_f1"],
                s=70,
                color=config["color"],
                edgecolor="white",
                linewidth=0.9,
                zorder=5,
            )

    ax.axvline(
        ORIGINAL_K,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="Original setting (k=1.0)",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 更小的坐标分度
    ax.xaxis.set_major_locator(
        MultipleLocator(0.20)
    )
    ax.xaxis.set_minor_locator(
        MultipleLocator(0.05)
    )
    ax.yaxis.set_major_locator(
        MultipleLocator(0.005)
    )
    ax.yaxis.set_minor_locator(
        MultipleLocator(0.001)
    )

    ax.grid(
        which="major",
        alpha=0.28,
    )
    ax.grid(
        which="minor",
        alpha=0.10,
        linestyle=":",
    )

    ax.set_title(
        "Hard-set F1 across decision thresholds",
        fontsize=16,
        pad=12,
    )

    ax.set_xlabel(
        "Threshold coefficient k\n"
        r"$threshold=mean+k\times std$",
        fontsize=12,
    )

    ax.set_ylabel(
        "Normal-class Micro F1",
        fontsize=12,
    )

    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()

    png_path = DATA_DIR / "hard_threshold_f1_curve_zoomed.png"
    pdf_path = DATA_DIR / "hard_threshold_f1_curve_zoomed.pdf"

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

    print(f"F1图：{png_path}")


def draw_pr_curve(scores_df):
    configure_style()

    fig, ax = plt.subplots(figsize=(8, 7))

    for model_key, config in MODELS.items():
        model_data = scores_df[
            scores_df["model_key"] == model_key
        ]

        y_true = model_data[
            "label"
        ].to_numpy(dtype=np.int64)

        z_score = model_data[
            "z_score"
        ].to_numpy(dtype=float)

        # z越小越像normal，因此取负数作为normal置信度
        normal_confidence = -z_score

        precision, recall, _ = precision_recall_curve(
            y_true,
            normal_confidence,
            pos_label=1,
        )

        pr_auc = auc(
            recall[::-1],
            precision[::-1],
        )

        ax.plot(
            recall[::-1],
            precision[::-1],
            color=config["color"],
            linewidth=2.4,
            label=(
                f"{config['name']} "
                f"(PR-AUC={pr_auc:.4f})"
            ),
        )

    # 放大曲线存在差异的右上角
    ax.set_xlim(0.88, 1.002)
    ax.set_ylim(0.94, 1.002)

    ax.xaxis.set_major_locator(
        MultipleLocator(0.02)
    )
    ax.xaxis.set_minor_locator(
        MultipleLocator(0.005)
    )
    ax.yaxis.set_major_locator(
        MultipleLocator(0.01)
    )
    ax.yaxis.set_minor_locator(
        MultipleLocator(0.002)
    )

    ax.grid(
        which="major",
        alpha=0.28,
    )
    ax.grid(
        which="minor",
        alpha=0.10,
        linestyle=":",
    )

    ax.set_title(
        "Precision–Recall curves on the Hard set",
        fontsize=16,
        pad=12,
    )

    ax.set_xlabel(
        "Normal-class Recall",
        fontsize=12,
    )

    ax.set_ylabel(
        "Normal-class Precision",
        fontsize=12,
    )

    ax.legend(
        loc="lower left",
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()

    png_path = (
        DATA_DIR
        / "hard_precision_recall_curve_zoomed.png"
    )

    pdf_path = (
        DATA_DIR
        / "hard_precision_recall_curve_zoomed.pdf"
    )

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

    print(f"PR图：{png_path}")


def main():
    for path in (
        SWEEP_CSV,
        SUMMARY_CSV,
        SCORES_CSV,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到文件：{path}"
            )

    print("读取已有CSV，不重新运行模型。")

    sweep_df = pd.read_csv(
        SWEEP_CSV,
        encoding="utf-8-sig",
    )

    summary_df = pd.read_csv(
        SUMMARY_CSV,
        encoding="utf-8-sig",
    )

    scores_df = pd.read_csv(
        SCORES_CSV,
        encoding="utf-8-sig",
    )

    draw_f1_curve(
        sweep_df,
        summary_df,
    )

    draw_pr_curve(
        scores_df,
    )

    print("\n绘图完成。")


if __name__ == "__main__":
    main()