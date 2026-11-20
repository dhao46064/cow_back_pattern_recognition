# -*- coding: utf-8 -*-
"""
Hard开放集双视角PR分析。

左图：normal为正类
    分数 = -z_score
    衡量正常样本保留能力

右图：abnormal为正类
    分数 = z_score
    衡量异常样本清除能力

直接读取已有threshold_sample_scores.csv，不重新推理模型。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
)


# ============================================================
# 1. 路径
# ============================================================

DATA_DIR = Path(
    r"D:\code_project\cow_recognition"
    r"\paper_materials\threshold_stability\dolg_hard"
)

SCORES_CSV = (
    DATA_DIR
    / "threshold_sample_scores.csv"
)

OUTPUT_PNG = (
    DATA_DIR
    / "hard_dual_class_precision_recall.png"
)

OUTPUT_PDF = (
    DATA_DIR
    / "hard_dual_class_precision_recall.pdf"
)

OUTPUT_CSV = (
    DATA_DIR
    / "hard_dual_class_pr_auc_summary.csv"
)


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


# ============================================================
# 2. PR指标计算
# ============================================================

def calculate_pr(y_true, confidence):
    """
    计算Precision–Recall曲线、PR-AUC和Average Precision。
    """
    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            confidence,
            pos_label=1,
        )
    )

    # sklearn返回的recall通常从1递减至0
    pr_auc = float(
        auc(
            recall[::-1],
            precision[::-1],
        )
    )

    average_precision = float(
        average_precision_score(
            y_true,
            confidence,
        )
    )

    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "pr_auc": pr_auc,
        "average_precision": average_precision,
    }


# ============================================================
# 3. 主程序
# ============================================================

def main():
    if not SCORES_CSV.is_file():
        raise FileNotFoundError(
            f"找不到逐样本分数：{SCORES_CSV}"
        )

    data = pd.read_csv(
        SCORES_CSV,
        encoding="utf-8-sig",
    )

    required_columns = {
        "model_key",
        "cow_id",
        "filename",
        "label",
        "z_score",
    }

    missing_columns = (
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise KeyError(
            f"CSV缺少列：{sorted(missing_columns)}\n"
            f"当前列名：{list(data.columns)}"
        )

    data = (
        data
        .replace([np.inf, -np.inf], np.nan)
        .dropna(
            subset=[
                "model_key",
                "label",
                "z_score",
            ]
        )
        .copy()
    )

    data["label"] = (
        data["label"].astype(int)
    )

    data["z_score"] = (
        data["z_score"].astype(float)
    )

    results = {}
    summary_rows = []

    for model_key, config in MODELS.items():
        model_data = data[
            data["model_key"] == model_key
        ].copy()

        if model_data.empty:
            raise ValueError(
                f"没有找到模型数据：{model_key}"
            )

        original_label = model_data[
            "label"
        ].to_numpy(dtype=np.int64)

        z_score = model_data[
            "z_score"
        ].to_numpy(dtype=float)

        # ----------------------------------------------------
        # normal为正类
        # 原始label：normal=1，abnormal=0
        # z越小越像normal，因此使用-z作为normal置信度
        # ----------------------------------------------------
        normal_true = original_label
        normal_confidence = -z_score

        normal_result = calculate_pr(
            normal_true,
            normal_confidence,
        )

        # ----------------------------------------------------
        # abnormal为正类
        # 转换后：abnormal=1，normal=0
        # z越大越像abnormal，因此直接使用z作为异常置信度
        # ----------------------------------------------------
        abnormal_true = (
            1 - original_label
        )
        abnormal_confidence = z_score

        abnormal_result = calculate_pr(
            abnormal_true,
            abnormal_confidence,
        )

        results[model_key] = {
            "normal": normal_result,
            "abnormal": abnormal_result,
        }

        summary_rows.append({
            "model_key": model_key,
            "model": config["name"],
            "num_cows": int(
                model_data["cow_id"].nunique()
            ),
            "num_samples": int(
                len(model_data)
            ),
            "num_normal": int(
                (original_label == 1).sum()
            ),
            "num_abnormal": int(
                (original_label == 0).sum()
            ),
            "normal_pr_auc": (
                normal_result["pr_auc"]
            ),
            "normal_average_precision": (
                normal_result[
                    "average_precision"
                ]
            ),
            "abnormal_pr_auc": (
                abnormal_result["pr_auc"]
            ),
            "abnormal_average_precision": (
                abnormal_result[
                    "average_precision"
                ]
            ),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 4. 绘制双子图
    # ========================================================

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6.5),
    )

    # --------------------------------------------------------
    # 左图：normal正类
    # --------------------------------------------------------

    ax = axes[0]

    for model_key, config in MODELS.items():
        result = results[
            model_key
        ]["normal"]

        ax.plot(
            result["recall"][::-1],
            result["precision"][::-1],
            color=config["color"],
            linewidth=2.4,
            label=(
                f"{config['name']} "
                f"(PR-AUC={result['pr_auc']:.4f})"
            ),
        )

    # 正常类指标都很高，放大右上角
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

    ax.set_title(
        "(a) Normal-class retention",
        fontsize=15,
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

    ax.grid(
        which="major",
        alpha=0.28,
    )
    ax.grid(
        which="minor",
        alpha=0.10,
        linestyle=":",
    )

    ax.legend(
        loc="lower left",
        frameon=False,
        fontsize=9,
    )

    # --------------------------------------------------------
    # 右图：abnormal正类
    # --------------------------------------------------------

    ax = axes[1]

    for model_key, config in MODELS.items():
        result = results[
            model_key
        ]["abnormal"]

        ax.plot(
            result["recall"][::-1],
            result["precision"][::-1],
            color=config["color"],
            linewidth=2.4,
            label=(
                f"{config['name']} "
                f"(PR-AUC={result['pr_auc']:.4f})"
            ),
        )

    ax.set_xlim(0.0, 1.002)
    ax.set_ylim(0.0, 1.002)

    ax.xaxis.set_major_locator(
        MultipleLocator(0.10)
    )
    ax.xaxis.set_minor_locator(
        MultipleLocator(0.025)
    )
    ax.yaxis.set_major_locator(
        MultipleLocator(0.10)
    )
    ax.yaxis.set_minor_locator(
        MultipleLocator(0.025)
    )

    ax.set_title(
        "(b) Abnormal-sample detection",
        fontsize=15,
        pad=12,
    )

    ax.set_xlabel(
        "Abnormal-class Recall",
        fontsize=12,
    )

    ax.set_ylabel(
        "Abnormal-class Precision",
        fontsize=12,
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

    ax.legend(
        loc="lower left",
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        "Precision–Recall analysis on the Hard open set",
        fontsize=17,
        y=1.01,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    # ========================================================
    # 5. 输出结果
    # ========================================================

    print("\n" + "=" * 90)
    print("Hard开放集双视角PR分析完成")
    print("=" * 90)

    print(
        summary_df[
            [
                "model",
                "normal_pr_auc",
                "abnormal_pr_auc",
                "normal_average_precision",
                "abnormal_average_precision",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print(f"\n图片：{OUTPUT_PNG}")
    print(f"PDF：{OUTPUT_PDF}")
    print(f"汇总：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()