# -*- coding: utf-8 -*-
"""
DOLG三种损失的Hard开放集阈值稳定性分析。

模型：
1. DOLG + ArcFace
2. DOLG + ArcFace + Center Loss
3. DOLG + ArcFace + SupCon

严格沿用原测试逻辑：
- normal = 1，abnormal = 0
- 每头牛分别计算embedding中心
- 欧氏距离作为异常分数
- threshold = mean + k * std
- score <= threshold 判定为normal
- 主指标按全部样本计算Micro Precision、Recall、F1

输出：
1. threshold_sweep_metrics.csv
2. threshold_model_summary.csv
3. threshold_sample_scores.csv
4. hard_threshold_f1_curve.png/pdf
5. hard_precision_recall_curve.png/pdf
"""

import gc
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
)


# ============================================================
# 1. 路径与参数
# ============================================================

PROJECT_ROOT = Path(r"D:\code_project\cow_recognition")

BASE_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "test"
    / "test_ablation.py"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "dolg_selected_loss_10ep"
)

SPLIT_JSON = Path(
    r"D:\datasets\cow_recognition8.3"
    r"\openset\openset_ori_8.2\split.json"
)

HARD_ORI_ROOT = Path(
    r"D:\datasets\cow_recognition8.3"
    r"\openset\openset_consensus_hard_11to1_v2_ori"
)

HARD_GT_ROOT = Path(
    r"D:\datasets\cow_recognition8.3"
    r"\openset\openset_consensus_hard_11to1_v2"
)

# 分析结果放入paper_materials
OUTPUT_DIR = (
    PROJECT_ROOT
    / "paper_materials"
    / "threshold_stability"
    / "dolg_hard"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_WORKERS = 0
SEED = 42

# 扫描mean + k*std中的k
K_VALUES = np.round(
    np.linspace(-1.0, 3.0, 161),
    4,
)

# 原测试使用的k
ORIGINAL_K = 1.0


MODELS = {
    "dolg_arcface": {
        "display_name": "DOLG+ArcFace",
        "checkpoint": (
            MODEL_ROOT
            / "dolg_arcface"
            / "checkpoints"
            / "epoch_10.pth"
        ),
        "color": "#7f7f7f",
    },
    "dolg_center": {
        "display_name": "DOLG+ArcFace+Center",
        "checkpoint": (
            MODEL_ROOT
            / "dolg_arcface_center_lam001"
            / "checkpoints"
            / "epoch_10.pth"
        ),
        "color": "#e67e22",
    },
    "dolg_supcon": {
        "display_name": "DOLG+ArcFace+SupCon",
        "checkpoint": (
            MODEL_ROOT
            / "dolg_arcface_supcon_lam01"
            / "checkpoints"
            / "epoch_10.pth"
        ),
        "color": "#2878b5",
    },
}


# ============================================================
# 2. 加载原测试模块
# ============================================================

def load_base_module():
    if not BASE_SCRIPT.is_file():
        raise FileNotFoundError(
            f"找不到基础测试脚本：{BASE_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "threshold_stability_base",
        BASE_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"无法加载基础测试脚本：{BASE_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


# ============================================================
# 3. 加载DOLG模型
# ============================================================

def build_dolg_model(checkpoint_path, base):
    from src.models.dolg_arcface_model import DOLGArcFaceCowModel

    checkpoint, state_dict = base.load_checkpoint(
        str(checkpoint_path)
    )

    arcface_weight = state_dict.get("arcface.weight")

    if arcface_weight is None:
        raise KeyError(
            f"{checkpoint_path}中没有arcface.weight"
        )

    num_classes = int(arcface_weight.shape[0])
    embedding_dim = int(arcface_weight.shape[1])

    run_config = (
        checkpoint.get("run_config", {})
        if isinstance(checkpoint, dict)
        else {}
    )

    model = DOLGArcFaceCowModel(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        local_dim=int(
            run_config.get("local_dim", 512)
        ),
        fusion_hidden_dim=int(
            run_config.get("fusion_hidden_dim", 512)
        ),
        s=float(
            run_config.get("arc_s", 30.0)
        ),
        m=float(
            run_config.get("arc_m", 0.25)
        ),
        pretrained=False,
        freeze_backbone=False,
        gem_p=float(
            run_config.get("gem_p", 3.0)
        ),
        gem_learn_p=not bool(
            run_config.get("fixed_gem_p", False)
        ),
        dilation_rates=tuple(
            run_config.get(
                "dilation_rates",
                (1, 2, 3),
            )
        ),
        dropout=float(
            run_config.get("dropout", 0.3)
        ),
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(DEVICE)
    model.eval()

    epoch = (
        checkpoint.get("epoch", "unknown")
        if isinstance(checkpoint, dict)
        else "unknown"
    )

    print(
        f"模型加载成功：{checkpoint_path.parent.parent.name}, "
        f"epoch={epoch}, classes={num_classes}, "
        f"embedding_dim={embedding_dim}"
    )

    return model


# ============================================================
# 4. 提取逐样本标准化距离
# ============================================================

def collect_model_scores(
    model_key,
    model_config,
    unseen_cows,
    transform,
    base,
):
    checkpoint_path = model_config["checkpoint"]

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"找不到checkpoint：{checkpoint_path}"
        )

    print("\n" + "=" * 80)
    print(
        f"开始提取：{model_config['display_name']}"
    )
    print("=" * 80)

    model = build_dolg_model(
        checkpoint_path,
        base,
    )

    rows = []
    skipped_cows = 0

    for cow_index, cow_id in enumerate(
        unseen_cows,
        start=1,
    ):
        ori_paths = base.list_ori_images_for_cow(
            ori_root=str(HARD_ORI_ROOT),
            cow_id=str(cow_id),
        )

        gt_dict = base.build_gt_dict_for_cow(
            gt_root=str(HARD_GT_ROOT),
            cow_id=str(cow_id),
        )

        if len(ori_paths) < 2 or not gt_dict:
            skipped_cows += 1
            continue

        embeddings, embedding_paths = (
            base.extract_embeddings_for_paths(
                model=model,
                image_paths=ori_paths,
                transform=transform,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
            )
        )

        if embeddings.shape[0] < 2:
            skipped_cows += 1
            continue

        # 与原测试完全相同的欧氏距离
        distances = base.calc_center_euclidean_scores(
            embeddings
        )

        score_mean = float(np.mean(distances))
        score_std = float(np.std(distances))

        # 极少数情况下所有距离完全相同
        if score_std <= 1e-12:
            print(
                f"[WARN] cow={cow_id}距离标准差接近0，跳过"
            )
            skipped_cows += 1
            continue

        # z_score对应原阈值中的k：
        # distance <= mean + k*std
        # 等价于 z_score <= k
        z_scores = (
            distances - score_mean
        ) / score_std

        matched_count = 0

        for image_path, distance, z_score in zip(
            embedding_paths,
            distances,
            z_scores,
        ):
            filename = Path(image_path).name

            if filename not in gt_dict:
                continue

            label = int(gt_dict[filename])

            rows.append({
                "model_key": model_key,
                "model": model_config[
                    "display_name"
                ],
                "cow_id": str(cow_id),
                "filename": filename,
                "label": label,
                "label_name": (
                    "normal"
                    if label == 1
                    else "abnormal"
                ),
                "distance": float(distance),
                "cow_score_mean": score_mean,
                "cow_score_std": score_std,
                "z_score": float(z_score),
            })

            matched_count += 1

        if cow_index % 20 == 0:
            print(
                f"[{cow_index}/{len(unseen_cows)}] "
                f"cow={cow_id}, matched={matched_count}"
            )

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError(
            f"{model_key}没有得到有效测试样本"
        )

    print(
        f"完成：{model_config['display_name']}, "
        f"cows={result['cow_id'].nunique()}, "
        f"samples={len(result)}, "
        f"skipped_cows={skipped_cows}"
    )

    return result


# ============================================================
# 5. Micro指标
# ============================================================

def calculate_micro_metrics(
    labels,
    predictions,
):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    )

    # normal = 1
    tp = int(
        ((labels == 1) & (predictions == 1)).sum()
    )
    tn = int(
        ((labels == 0) & (predictions == 0)).sum()
    )
    fp = int(
        ((labels == 0) & (predictions == 1)).sum()
    )
    fn = int(
        ((labels == 1) & (predictions == 0)).sum()
    )

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    accuracy = (
        (tp + tn)
        / max(len(labels), 1)
    )

    abnormal_false_retention = (
        fp / max(fp + tn, 1)
    )

    return {
        "accuracy_micro": accuracy,
        "precision_normal_micro": precision,
        "recall_normal_micro": recall,
        "f1_normal_micro": f1,
        "abnormal_false_retention_rate": (
            abnormal_false_retention
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# 6. 阈值扫描
# ============================================================

def run_threshold_sweep(all_scores):
    sweep_rows = []
    summary_rows = []
    pr_results = {}

    for model_key, config in MODELS.items():
        model_data = (
            all_scores[
                all_scores["model_key"] == model_key
            ]
            .copy()
        )

        labels = model_data[
            "label"
        ].to_numpy(dtype=np.int64)

        z_scores = model_data[
            "z_score"
        ].to_numpy(dtype=float)

        for k in K_VALUES:
            # 原规则：
            # distance <= mean + k*std -> normal
            predictions = (
                z_scores <= k
            ).astype(np.int64)

            metrics = calculate_micro_metrics(
                labels,
                predictions,
            )

            sweep_rows.append({
                "model_key": model_key,
                "model": config["display_name"],
                "std_scale_k": float(k),
                "num_cows": int(
                    model_data["cow_id"].nunique()
                ),
                "num_samples": int(len(model_data)),
                **metrics,
            })

        # PR曲线需要“越大越像normal”的分数
        normal_confidence = -z_scores

        precision_curve, recall_curve, _ = (
            precision_recall_curve(
                labels,
                normal_confidence,
                pos_label=1,
            )
        )

        # recall_curve默认通常是递减方向
        pr_auc = float(
            auc(
                recall_curve[::-1],
                precision_curve[::-1],
            )
        )

        average_precision = float(
            average_precision_score(
                labels,
                normal_confidence,
            )
        )

        pr_results[model_key] = {
            "precision": precision_curve,
            "recall": recall_curve,
            "pr_auc": pr_auc,
            "average_precision": average_precision,
        }

    sweep_df = pd.DataFrame(sweep_rows)

    for model_key, config in MODELS.items():
        model_sweep = (
            sweep_df[
                sweep_df["model_key"] == model_key
            ]
            .copy()
        )

        # F1最高的阈值
        best_index = (
            model_sweep["f1_normal_micro"].idxmax()
        )
        best_row = model_sweep.loc[best_index]

        # 原测试k=1.0对应结果
        original_index = (
            model_sweep["std_scale_k"]
            .sub(ORIGINAL_K)
            .abs()
            .idxmin()
        )
        original_row = model_sweep.loc[
            original_index
        ]

        pr_info = pr_results[model_key]

        summary_rows.append({
            "model_key": model_key,
            "model": config["display_name"],
            "num_cows": int(
                best_row["num_cows"]
            ),
            "num_samples": int(
                best_row["num_samples"]
            ),
            "best_k": float(
                best_row["std_scale_k"]
            ),
            "best_precision": float(
                best_row[
                    "precision_normal_micro"
                ]
            ),
            "best_recall": float(
                best_row[
                    "recall_normal_micro"
                ]
            ),
            "best_f1": float(
                best_row["f1_normal_micro"]
            ),
            "original_k": ORIGINAL_K,
            "original_precision": float(
                original_row[
                    "precision_normal_micro"
                ]
            ),
            "original_recall": float(
                original_row[
                    "recall_normal_micro"
                ]
            ),
            "original_f1": float(
                original_row["f1_normal_micro"]
            ),
            "pr_auc": pr_info["pr_auc"],
            "average_precision": (
                pr_info["average_precision"]
            ),
        })

    summary_df = pd.DataFrame(summary_rows)

    return sweep_df, summary_df, pr_results


# ============================================================
# 7. 绘图
# ============================================================

def configure_plot_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
    })


def plot_f1_curve(
    sweep_df,
    summary_df,
):
    configure_plot_style()

    fig, ax = plt.subplots(
        figsize=(9, 6.5)
    )

    for model_key, config in MODELS.items():
        data = sweep_df[
            sweep_df["model_key"] == model_key
        ]

        ax.plot(
            data["std_scale_k"],
            data["f1_normal_micro"],
            color=config["color"],
            linewidth=2.3,
            label=config["display_name"],
        )

        summary = summary_df[
            summary_df["model_key"] == model_key
        ].iloc[0]

        ax.scatter(
            summary["best_k"],
            summary["best_f1"],
            color=config["color"],
            s=55,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )

    ax.axvline(
        ORIGINAL_K,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="Original setting (k=1.0)",
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
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "hard_threshold_f1_curve.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        OUTPUT_DIR
        / "hard_threshold_f1_curve.pdf",
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def plot_pr_curve(pr_results):
    configure_plot_style()

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    for model_key, config in MODELS.items():
        result = pr_results[model_key]

        ax.plot(
            result["recall"][::-1],
            result["precision"][::-1],
            color=config["color"],
            linewidth=2.3,
            label=(
                f"{config['display_name']} "
                f"(PR-AUC={result['pr_auc']:.4f})"
            ),
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

    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)

    ax.legend(
        loc="lower left",
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "hard_precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        OUTPUT_DIR
        / "hard_precision_recall_curve.pdf",
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


# ============================================================
# 8. 主程序
# ============================================================

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    for required_path in (
        SPLIT_JSON,
        HARD_ORI_ROOT,
        HARD_GT_ROOT,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"路径不存在：{required_path}"
            )

    base = load_base_module()

    unseen_cows = base.load_split_ids(
        str(SPLIT_JSON),
        split_key="unseen_cows",
    )

    transform = base.build_eval_transforms(
        img_size=224
    )

    score_frames = []

    for model_key, model_config in MODELS.items():
        model_scores = collect_model_scores(
            model_key=model_key,
            model_config=model_config,
            unseen_cows=unseen_cows,
            transform=transform,
            base=base,
        )

        score_frames.append(model_scores)

    all_scores = pd.concat(
        score_frames,
        ignore_index=True,
    )

    # 检查三组模型是否使用相同测试样本
    sample_counts = (
        all_scores
        .groupby("model_key")
        .size()
    )

    print("\n各模型有效样本数：")
    print(sample_counts)

    if sample_counts.nunique() != 1:
        print(
            "[WARN] 三组模型有效样本数不完全一致，"
            "请检查是否存在图像读取失败。"
        )

    sweep_df, summary_df, pr_results = (
        run_threshold_sweep(all_scores)
    )

    # 保存所有中间结果，便于复查
    all_scores.to_csv(
        OUTPUT_DIR
        / "threshold_sample_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sweep_df.to_csv(
        OUTPUT_DIR
        / "threshold_sweep_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "threshold_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_f1_curve(
        sweep_df,
        summary_df,
    )

    plot_pr_curve(pr_results)

    print("\n" + "=" * 90)
    print("阈值稳定性分析完成")
    print("=" * 90)

    print(
        summary_df[
            [
                "model",
                "best_k",
                "best_f1",
                "original_f1",
                "pr_auc",
                "average_precision",
            ]
        ].to_string(index=False)
    )

    print(f"\n输出目录：{OUTPUT_DIR}")
    print("1. threshold_sample_scores.csv")
    print("2. threshold_sweep_metrics.csv")
    print("3. threshold_model_summary.csv")
    print("4. hard_threshold_f1_curve.png")
    print("5. hard_precision_recall_curve.png")


if __name__ == "__main__":
    main()