# -*- coding: utf-8 -*-
"""
DOLG-ArcFace 开放集测试脚本。

测试逻辑沿用现有 ArcFace 测试脚本：
1. 从 split.json 读取 unseen_cows；
2. 对每头 unseen 牛的 ori 图像提取 DOLG embedding；
3. 计算每张图到该牛 embedding 中心的欧氏距离；
4. 使用 threshold = mean + std_scale * std 判定 normal / abnormal；
5. GT 仅用于预测完成后的指标评估；
6. 保存每头牛指标 CSV 和两类错误图片。

建议放置位置：
    test_analysis/test_sup_exp003_dolg_arcface.py
或：
    scripts/test_sup_exp003_dolg_arcface.py
"""

import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# =========================================================
# 0. 基础配置
# =========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


# =========================================================
# 1. 项目路径与模型导入
# =========================================================
# 兼容脚本放在项目根目录、scripts 或 test_analysis 子目录的情况。
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_CANDIDATES = [
    SCRIPT_DIR,
    SCRIPT_DIR.parent,
    SCRIPT_DIR.parent.parent,
]

for candidate in PROJECT_ROOT_CANDIDATES:
    if (candidate / "src").exists():
        PROJECT_ROOT = candidate
        break
else:
    PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sup.models.dolg_arcface_model import DOLGArcFaceCowModel


# =========================================================
# 2. 图像变换
# =========================================================
def build_eval_transforms(img_size: int = 224):
    """必须与训练阶段的验证变换保持一致。"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# =========================================================
# 3. checkpoint 加载与 DOLG 配置推断
# =========================================================
def load_checkpoint(model_path: str):
    ckpt = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint 中未找到有效的模型 state_dict。")

    return ckpt, normalize_state_dict_keys(state_dict)


def normalize_state_dict_keys(state_dict: dict) -> dict:
    """兼容 DataParallel、DDP 和 torch.compile 产生的参数名前缀。"""
    normalized = {}

    prefixes = ("module.", "_orig_mod.")

    for key, value in state_dict.items():
        new_key = key
        changed = True

        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        normalized[new_key] = value

    return normalized


def infer_dolg_config(state_dict: dict) -> Dict[str, int]:
    """
    从 checkpoint 参数形状推断 DOLG 的关键维度。

    无法从 state_dict 推断 dilation 的具体数值，因此仍需与训练脚本保持一致；
    默认使用训练脚本中的 (1, 2, 3)。
    """
    required_keys = [
        "arcface.weight",
        "global_projection.0.weight",
        "embedding_head.0.weight",
    ]

    missing = [key for key in required_keys if key not in state_dict]
    if missing:
        raise KeyError(
            "checkpoint 不是当前 DOLGArcFaceCowModel 的权重，缺少参数："
            + ", ".join(missing)
        )

    arcface_weight = state_dict["arcface.weight"]
    global_projection_weight = state_dict["global_projection.0.weight"]
    embedding_head_weight = state_dict["embedding_head.0.weight"]

    if arcface_weight.ndim != 2:
        raise ValueError("arcface.weight 形状异常。")

    num_classes = int(arcface_weight.shape[0])
    embedding_dim = int(arcface_weight.shape[1])
    local_dim = int(global_projection_weight.shape[0])
    fusion_hidden_dim = int(embedding_head_weight.shape[0])

    expected_fusion_input = 2 * local_dim
    actual_fusion_input = int(embedding_head_weight.shape[1])

    if actual_fusion_input != expected_fusion_input:
        raise ValueError(
            "DOLG 融合层维度不匹配："
            f"embedding_head.0.weight 输入维度={actual_fusion_input}，"
            f"但根据 local_dim={local_dim} 应为 {expected_fusion_input}。"
        )

    return {
        "num_classes": num_classes,
        "embedding_dim": embedding_dim,
        "local_dim": local_dim,
        "fusion_hidden_dim": fusion_hidden_dim,
    }


def build_model_from_ckpt(
    model_path: str,
    device: str,
    dilation_rates: Sequence[int] = (1, 2, 3),
):
    ckpt, state_dict = load_checkpoint(model_path)
    config = infer_dolg_config(state_dict)

    epoch = ckpt.get("epoch", "unknown") if isinstance(ckpt, dict) else "unknown"

    print("=" * 80)
    print(f"[INFO] model_path        : {model_path}")
    print(f"[INFO] checkpoint epoch  : {epoch}")
    print(f"[INFO] num_classes       : {config['num_classes']}")
    print(f"[INFO] embedding_dim     : {config['embedding_dim']}")
    print(f"[INFO] local_dim         : {config['local_dim']}")
    print(f"[INFO] fusion_hidden_dim : {config['fusion_hidden_dim']}")
    print(f"[INFO] dilation_rates    : {tuple(dilation_rates)}")
    print("=" * 80)

    model = DOLGArcFaceCowModel(
        num_classes=config["num_classes"],
        embedding_dim=config["embedding_dim"],
        local_dim=config["local_dim"],
        fusion_hidden_dim=config["fusion_hidden_dim"],
        pretrained=False,
        freeze_backbone=False,
        gem_p=3.0,
        gem_learn_p=True,
        dilation_rates=tuple(dilation_rates),
        dropout=0.3,
    )

    # 测试阶段必须严格加载。出现不一致时直接报错，避免静默产生错误结果。
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()

    model_info = model.get_model_info()
    print(f"[INFO] loaded GeM p      : {model_info['gem_p']:.6f}")
    print("[INFO] checkpoint loaded strictly: OK")

    return model


# =========================================================
# 4. split.json 读取
# =========================================================
def load_split_ids(
    split_json_path: str,
    split_key: str = "unseen_cows",
) -> List[str]:
    with open(split_json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if split_key not in data:
        raise KeyError(f"{split_json_path} 中不存在键：{split_key}")

    return [str(value) for value in data[split_key]]


# =========================================================
# 5. 文件读取
# =========================================================
def list_images_in_dir(dir_path: Path) -> List[str]:
    if not dir_path.exists():
        return []

    return sorted(
        str(path)
        for path in dir_path.iterdir()
        if path.is_file() and is_image_file(path)
    )


def list_ori_images_for_cow(ori_root: str, cow_id: str) -> List[str]:
    return list_images_in_dir(Path(ori_root) / cow_id)


def build_gt_dict_for_cow(gt_root: str, cow_id: str) -> Dict[str, int]:
    """
    normal = 1
    abnormal = 0
    按文件名建立 GT 字典。
    """
    gt: Dict[str, int] = {}

    cow_dir = Path(gt_root) / cow_id
    normal_dir = cow_dir / "normal"
    abnormal_dir = cow_dir / "abnormal"

    if normal_dir.exists():
        for path in normal_dir.iterdir():
            if path.is_file() and is_image_file(path):
                gt[path.name] = 1

    if abnormal_dir.exists():
        for path in abnormal_dir.iterdir():
            if path.is_file() and is_image_file(path):
                gt[path.name] = 0

    return gt


# =========================================================
# 6. Dataset
# =========================================================
class SimpleImageDataset(Dataset):
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]

        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"读取图像失败：{path}") from exc

        return image, path


# =========================================================
# 7. DOLG embedding 提取
# =========================================================
@torch.inference_mode()
def extract_embeddings_for_paths(
    model,
    image_paths: List[str],
    transform,
    device: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[np.ndarray, List[str]]:
    dataset = SimpleImageDataset(
        image_paths=image_paths,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
    )

    all_embeddings = []
    all_paths: List[str] = []

    for images, paths in loader:
        images = images.to(device, non_blocking=True)

        # DOLGArcFaceCowModel 与原模型保持相同的 forward_emb 接口。
        embeddings = model.forward_emb(images)
        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        all_embeddings.append(embeddings.cpu().numpy())
        all_paths.extend(list(paths))

    if not all_embeddings:
        return np.zeros((0, 1), dtype=np.float32), []

    embeddings_np = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    return embeddings_np, all_paths


# =========================================================
# 8. 到当前牛 embedding 中心的欧氏距离
# =========================================================
def calc_center_euclidean_scores(embeddings: np.ndarray) -> np.ndarray:
    """
    对同一牛目录中的全部 embedding 求均值中心，
    再计算每张图到中心的欧氏距离。距离越大，越倾向于 abnormal。
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings 应为二维矩阵 [N,D]，实际为 {embeddings.shape}"
        )

    if embeddings.shape[0] < 2:
        return np.zeros((embeddings.shape[0],), dtype=np.float32)

    center = np.mean(embeddings, axis=0, keepdims=True)
    scores = np.linalg.norm(embeddings - center, axis=1)
    return scores.astype(np.float32)


def build_threshold_mean_std(
    scores: np.ndarray,
    std_scale: float = 1.0,
) -> Dict[str, float]:
    if scores.size == 0:
        raise ValueError("scores 为空，无法计算阈值。")

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    threshold = mean + std_scale * std

    return {
        "mean": mean,
        "std": std,
        "threshold": float(threshold),
    }


# =========================================================
# 9. 二分类指标：normal=1，abnormal=0
# =========================================================
def calc_metrics_binary(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)

    if y_true_np.size == 0:
        raise ValueError("没有有效 GT 样本，无法计算评估指标。")

    if y_true_np.shape != y_pred_np.shape:
        raise ValueError("y_true 与 y_pred 长度不一致。")

    tp = int(((y_true_np == 1) & (y_pred_np == 1)).sum())
    tn = int(((y_true_np == 0) & (y_pred_np == 0)).sum())
    fp = int(((y_true_np == 0) & (y_pred_np == 1)).sum())
    fn = int(((y_true_np == 1) & (y_pred_np == 0)).sum())

    accuracy = (tp + tn) / max(len(y_true_np), 1)
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    return {
        "accuracy": float(accuracy),
        "precision_normal": float(precision),
        "recall_normal": float(recall),
        "f1_normal": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# =========================================================
# 10. 保存错误图片
# =========================================================
def save_error_image_with_text(
    src_path: str,
    dst_path: str,
    row: dict,
) -> None:
    with Image.open(src_path) as source_image:
        image = source_image.convert("RGB")

    draw = ImageDraw.Draw(image)
    text = (
        f"score: {row['score']:.6f}\n"
        f"threshold: {row['threshold']:.6f}\n"
        f"pred: {row['pred']}\n"
        f"gt: {row['gt']}"
    )

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    # 加一个黑底，避免文字在亮色牛背上看不清。
    text_box = draw.multiline_textbbox((10, 10), text, font=font, spacing=4)
    draw.rectangle(text_box, fill=(0, 0, 0))
    draw.multiline_text(
        (10, 10),
        text,
        fill=(255, 0, 0),
        font=font,
        spacing=4,
    )

    destination = Path(dst_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def save_wrong_images_for_cow(
    error_root: str,
    cow_id: str,
    normal_to_abnormal: list,
    abnormal_to_normal: list,
) -> None:
    error_root_path = Path(error_root)

    normal_to_abnormal_dir = (
        error_root_path / str(cow_id) / "normal_to_abnormal"
    )
    abnormal_to_normal_dir = (
        error_root_path / str(cow_id) / "abnormal_to_normal"
    )

    normal_to_abnormal_dir.mkdir(parents=True, exist_ok=True)
    abnormal_to_normal_dir.mkdir(parents=True, exist_ok=True)

    for row in normal_to_abnormal:
        source = Path(row["path"])
        if source.exists():
            save_error_image_with_text(
                str(source),
                str(normal_to_abnormal_dir / source.name),
                row,
            )

    for row in abnormal_to_normal:
        source = Path(row["path"])
        if source.exists():
            save_error_image_with_text(
                str(source),
                str(abnormal_to_normal_dir / source.name),
                row,
            )


# =========================================================
# 11. CSV 保存
# =========================================================
def save_metrics_csv(
    rows: List[dict],
    overall_metrics: Dict[str, float],
    csv_path: str,
) -> None:
    if not rows:
        raise ValueError("没有任何有效牛只结果，无法写入 CSV。")

    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 最后一行追加 micro overall，方便一次查看总结果。
    overall_row = {
        "cow_id": "OVERALL_MICRO",
        "num_ori_images": "",
        "num_matched_gt": sum(int(row["num_matched_gt"]) for row in rows),
        "num_unmatched_gt": sum(int(row["num_unmatched_gt"]) for row in rows),
        "mean": "",
        "std": "",
        "threshold": "",
        "accuracy": overall_metrics["accuracy"],
        "precision_normal": overall_metrics["precision_normal"],
        "recall_normal": overall_metrics["recall_normal"],
        "f1_normal": overall_metrics["f1_normal"],
        "tp": overall_metrics["tp"],
        "tn": overall_metrics["tn"],
        "fp": overall_metrics["fp"],
        "fn": overall_metrics["fn"],
        "num_normal_to_abnormal": sum(
            int(row["num_normal_to_abnormal"]) for row in rows
        ),
        "num_abnormal_to_normal": sum(
            int(row["num_abnormal_to_normal"]) for row in rows
        ),
    }

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(overall_row)

    print(f"[INFO] CSV saved to: {output_path}")


# =========================================================
# 12. 主评估流程
# =========================================================
def evaluate_unseen_ori_with_gt_check(
    split_json_path: str,
    ori_root: str,
    gt_root: str,
    model_path: str,
    overall_csv: str,
    error_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
    seed: int = 42,
    std_scale: float = 1.0,
    dilation_rates: Sequence[int] = (1, 2, 3),
):
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    transform = build_eval_transforms(img_size=img_size)

    unseen_cows = load_split_ids(
        split_json_path,
        split_key="unseen_cows",
    )
    print(f"[INFO] unseen cows = {len(unseen_cows)}")

    model = build_model_from_ckpt(
        model_path=model_path,
        device=device,
        dilation_rates=dilation_rates,
    )

    all_y_true: List[int] = []
    all_y_pred: List[int] = []
    rows: List[dict] = []

    for cow_id in unseen_cows:
        print("\n" + "=" * 80)
        print(f"[INFO] Processing cow_id = {cow_id}")

        ori_paths = list_ori_images_for_cow(ori_root, cow_id)
        gt_dict = build_gt_dict_for_cow(gt_root, cow_id)

        if len(ori_paths) < 2:
            print(f"[WARN] ori 图片少于 2 张，跳过：{cow_id}")
            continue

        if not gt_dict:
            print(f"[WARN] GT 为空，跳过：{cow_id}")
            continue

        embeddings, embedding_paths = extract_embeddings_for_paths(
            model=model,
            image_paths=ori_paths,
            transform=transform,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        if embeddings.shape[0] < 2:
            print(f"[WARN] embedding 数量少于 2，跳过：{cow_id}")
            continue

        scores = calc_center_euclidean_scores(embeddings)
        threshold_info = build_threshold_mean_std(
            scores,
            std_scale=std_scale,
        )

        threshold = threshold_info["threshold"]
        mean = threshold_info["mean"]
        std = threshold_info["std"]

        y_true: List[int] = []
        y_pred: List[int] = []
        normal_to_abnormal = []
        abnormal_to_normal = []
        num_unmatched_gt = 0

        for path, score in zip(embedding_paths, scores):
            filename = Path(path).name
            prediction = 1 if float(score) <= threshold else 0

            if filename not in gt_dict:
                num_unmatched_gt += 1
                continue

            ground_truth = int(gt_dict[filename])

            detail = {
                "path": path,
                "filename": filename,
                "score": float(score),
                "threshold": float(threshold),
                "pred": int(prediction),
                "gt": int(ground_truth),
            }

            y_true.append(ground_truth)
            y_pred.append(prediction)

            if ground_truth == 1 and prediction == 0:
                normal_to_abnormal.append(detail)
            elif ground_truth == 0 and prediction == 1:
                abnormal_to_normal.append(detail)

        if not y_true:
            print(f"[WARN] 没有任何文件名能够与 GT 对齐，跳过：{cow_id}")
            continue

        metrics = calc_metrics_binary(y_true, y_pred)

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        print(f"[INFO] num_ori_images        = {len(ori_paths)}")
        print(f"[INFO] num_matched_gt       = {len(y_true)}")
        print(f"[INFO] num_unmatched_gt     = {num_unmatched_gt}")
        print(f"[INFO] score mean/std       = {mean:.6f} / {std:.6f}")
        print(f"[INFO] threshold            = {threshold:.6f}")
        print(f"[INFO] metrics              = {metrics}")
        print(f"[INFO] normal_to_abnormal   = {len(normal_to_abnormal)}")
        print(f"[INFO] abnormal_to_normal   = {len(abnormal_to_normal)}")

        save_wrong_images_for_cow(
            error_root=error_root,
            cow_id=cow_id,
            normal_to_abnormal=normal_to_abnormal,
            abnormal_to_normal=abnormal_to_normal,
        )

        rows.append({
            "cow_id": cow_id,
            "num_ori_images": len(ori_paths),
            "num_matched_gt": len(y_true),
            "num_unmatched_gt": num_unmatched_gt,
            "mean": mean,
            "std": std,
            "threshold": threshold,
            "accuracy": metrics["accuracy"],
            "precision_normal": metrics["precision_normal"],
            "recall_normal": metrics["recall_normal"],
            "f1_normal": metrics["f1_normal"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "num_normal_to_abnormal": len(normal_to_abnormal),
            "num_abnormal_to_normal": len(abnormal_to_normal),
        })

    if not all_y_true:
        raise RuntimeError(
            "没有任何有效评测样本，请检查 split.json、ori_root、gt_root 和文件名对应关系。"
        )

    overall_metrics = calc_metrics_binary(all_y_true, all_y_pred)

    print("\n" + "#" * 80)
    print("[INFO] OVERALL MICRO METRICS")
    print(overall_metrics)
    print("#" * 80)

    save_metrics_csv(
        rows=rows,
        overall_metrics=overall_metrics,
        csv_path=overall_csv,
    )

    return overall_metrics


# =========================================================
# 13. main：只需要修改这里的路径
# =========================================================
def main() -> None:
    split_json_path = (
        r"D:\code_project\triplet_code\data\dataset_OBB\split.json"
    )

    ori_root = (
        r"D:\code_project\triplet_code\data\dataset_obb_ori"
    )

    gt_root = (
        r"D:\datasets\openset_CowBack\openset_quality_P99.5"
    )

    # 改成训练完成后的 DOLG checkpoint。
    model_path = (
        r"D:\code_project\triplet_code\logs\exp003_7.25_arcface_DOLG\checkpoints\latest.pth"
    )

    overall_csv = (
        r"D:\code_project\triplet_code\logs\exp003_7.25_arcface_DOLG\dolg_unseen_eval_metrics_open995.csv"
    )

    error_root = (
        r"D:\code_project\triplet_code\logs\exp003_7.25_arcface_DOLG\dolg_unseen_error_images"
    )

    evaluate_unseen_ori_with_gt_check(
        split_json_path=split_json_path,
        ori_root=ori_root,
        gt_root=gt_root,
        model_path=model_path,
        overall_csv=overall_csv,
        error_root=error_root,
        batch_size=64,
        num_workers=4,
        img_size=224,
        seed=42,
        std_scale=1.0,
        # 必须与训练脚本中的 dilation_rates 一致。
        dilation_rates=(1, 2, 3),
    )


if __name__ == "__main__":
    main()
