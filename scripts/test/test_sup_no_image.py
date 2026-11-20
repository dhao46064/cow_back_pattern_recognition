# -*- coding: utf-8 -*-
r"""
仅计算并保存指标：
1. 不保存错误图片
2. 不保存逐图片明细
3. 不保存 JSON
4. 保存每头牛指标 CSV
5. 保存总体 micro / macro 指标 CSV

建议保存：
D:\code_project\triplet_code\test_analysis\unseen_ori_metrics_only.py"""


import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List
CURRENT_FILE = Path(__file__).resolve()

# parents[0] = scripts\train
# parents[1] = scripts
# parents[2] = cow_recognition
PROJECT_ROOT = CURRENT_FILE.parents[2]

SRC_ROOT = PROJECT_ROOT / "src"

if not SRC_ROOT.exists():
    raise FileNotFoundError(
        f"没有找到src目录：{SRC_ROOT}\n"
        f"当前训练脚本：{CURRENT_FILE}\n"
        f"计算得到的项目根目录：{PROJECT_ROOT}"
    )

# 必须使用insert(0)，保证优先加载当前项目中的src
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

print("=" * 80)
print(f"训练脚本：{CURRENT_FILE}")
print(f"项目根目录：{PROJECT_ROOT}")
print(f"src目录：{SRC_ROOT}")
print("=" * 80)

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# =========================================================
# 0. 项目路径
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.arcface_model import ArcFaceCowModel


IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


# =========================================================
# 1. 图像变换
# =========================================================
def build_eval_transforms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# =========================================================
# 2. checkpoint 与模型
# =========================================================
def load_checkpoint(model_path: str):
    try:
        ckpt = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(
            model_path,
            map_location="cpu",
        )

    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]

    return ckpt


def infer_arcface_config(state_dict: dict):
    if "arcface.weight" not in state_dict:
        raise KeyError(
            "checkpoint 中没有 arcface.weight，"
            "无法推断 num_classes / embedding_dim"
        )

    weight = state_dict["arcface.weight"]
    return int(weight.shape[0]), int(weight.shape[1])


def build_model_from_ckpt(
    model_path: str,
    device: str,
):
    state_dict = load_checkpoint(model_path)

    num_classes, embedding_dim = infer_arcface_config(
        state_dict
    )

    print("=" * 80)
    print(f"[INFO] model_path  : {model_path}")
    print(f"[INFO] num_classes : {num_classes}")
    print(f"[INFO] emb_dim     : {embedding_dim}")
    print("=" * 80)

    model = ArcFaceCowModel(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained=False,
        freeze_backbone=False,
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(f"[INFO] missing keys   : {len(missing)}")
    print(f"[INFO] unexpected keys: {len(unexpected)}")

    if missing:
        print(f"[WARN] first missing   : {missing[:10]}")

    if unexpected:
        print(f"[WARN] first unexpected: {unexpected[:10]}")

    model = model.to(device)
    model.eval()

    return model


# =========================================================
# 3. split 与文件读取
# =========================================================
def load_split_ids(
    split_json_path: str,
    split_key: str = "unseen_cows",
) -> List[str]:
    with open(
        split_json_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if split_key not in data:
        raise KeyError(
            f"{split_json_path} 中不存在键：{split_key}"
        )

    return [str(cow_id) for cow_id in data[split_key]]


def list_images_in_dir(dir_path: Path) -> List[str]:
    if not dir_path.exists():
        return []

    return sorted(
        str(path)
        for path in dir_path.iterdir()
        if path.is_file() and is_image_file(path)
    )


def list_ori_images_for_cow(
    ori_root: str,
    cow_id: str,
) -> List[str]:
    return list_images_in_dir(
        Path(ori_root) / cow_id
    )


def build_gt_dict_for_cow(
    gt_root: str,
    cow_id: str,
) -> Dict[str, int]:
    """
    normal = 1
    abnormal = 0
    """
    gt_dict: Dict[str, int] = {}

    cow_dir = Path(gt_root) / cow_id
    normal_dir = cow_dir / "normal"
    abnormal_dir = cow_dir / "abnormal"

    if normal_dir.exists():
        for path in normal_dir.iterdir():
            if path.is_file() and is_image_file(path):
                gt_dict[path.name] = 1

    if abnormal_dir.exists():
        for path in abnormal_dir.iterdir():
            if path.is_file() and is_image_file(path):
                gt_dict[path.name] = 0

    return gt_dict


# =========================================================
# 4. Dataset 与 embedding
# =========================================================
class SimpleImageDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        transform=None,
    ):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]

        with Image.open(path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, path


@torch.no_grad()
def extract_embeddings_for_paths(
    model,
    image_paths: List[str],
    transform,
    device: str,
    batch_size: int = 64,
    num_workers: int = 0,
):
    dataset = SimpleImageDataset(
        image_paths=image_paths,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    all_embeddings = []
    all_paths = []

    for images, paths in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        embeddings = model.forward_emb(images)

        embeddings = embeddings / (
            embeddings.norm(
                dim=1,
                keepdim=True,
            )
            + 1e-12
        )

        all_embeddings.append(
            embeddings.cpu().numpy()
        )
        all_paths.extend(list(paths))

    if not all_embeddings:
        return (
            np.zeros((0, 1), dtype=np.float32),
            [],
        )

    embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    ).astype(np.float32)

    return embeddings, all_paths


# =========================================================
# 5. 分数、阈值与指标
# =========================================================
def calc_center_euclidean_scores(
    embeddings: np.ndarray,
) -> np.ndarray:
    if embeddings.shape[0] < 2:
        return np.zeros(
            embeddings.shape[0],
            dtype=np.float32,
        )

    center = np.mean(
        embeddings,
        axis=0,
        keepdims=True,
    )

    scores = np.linalg.norm(
        embeddings - center,
        axis=1,
    )

    return scores.astype(np.float32)


def build_threshold_mean_std(
    scores: np.ndarray,
    std_scale: float = 1.0,
) -> Dict[str, float]:
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    threshold = mean + std_scale * std

    return {
        "mean": mean,
        "std": std,
        "threshold": float(threshold),
    }


def calc_metrics_binary(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    y_true_array = np.asarray(
        y_true,
        dtype=np.int64,
    )
    y_pred_array = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    tp = int(
        (
            (y_true_array == 1)
            & (y_pred_array == 1)
        ).sum()
    )
    tn = int(
        (
            (y_true_array == 0)
            & (y_pred_array == 0)
        ).sum()
    )
    fp = int(
        (
            (y_true_array == 0)
            & (y_pred_array == 1)
        ).sum()
    )
    fn = int(
        (
            (y_true_array == 1)
            & (y_pred_array == 0)
        ).sum()
    )

    accuracy = (tp + tn) / max(len(y_true_array), 1)
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

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


def safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


# =========================================================
# 6. 保存 CSV
# =========================================================
def save_csv(
    rows: List[dict],
    csv_path: str,
):
    if not rows:
        raise ValueError(
            f"没有可保存的数据：{csv_path}"
        )

    output_path = Path(csv_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] csv saved to: {output_path}")


# =========================================================
# 7. 主评估流程
# =========================================================
def evaluate_unseen_metrics_only(
    split_json_path: str,
    ori_root: str,
    gt_root: str,
    model_path: str,
    per_cow_csv: str,
    overall_csv: str,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    std_scale: float = 1.0,
):
    set_seed(seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[INFO] device = {device}")

    transform = build_eval_transforms(
        img_size=224
    )

    unseen_cows = load_split_ids(
        split_json_path,
        split_key="unseen_cows",
    )

    print(f"[INFO] unseen cows = {len(unseen_cows)}")

    model = build_model_from_ckpt(
        model_path=model_path,
        device=device,
    )

    all_y_true: List[int] = []
    all_y_pred: List[int] = []
    per_cow_rows: List[dict] = []

    for cow_id in unseen_cows:
        print("\n" + "=" * 80)
        print(f"[INFO] Processing cow_id = {cow_id}")

        ori_paths = list_ori_images_for_cow(
            ori_root=ori_root,
            cow_id=cow_id,
        )

        gt_dict = build_gt_dict_for_cow(
            gt_root=gt_root,
            cow_id=cow_id,
        )

        if len(ori_paths) < 2:
            print(
                f"[WARN] ori 图片少于 2 张，跳过：{cow_id}"
            )
            continue

        if not gt_dict:
            print(
                f"[WARN] GT 为空，跳过：{cow_id}"
            )
            continue

        embeddings, embedding_paths = (
            extract_embeddings_for_paths(
                model=model,
                image_paths=ori_paths,
                transform=transform,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        )

        if embeddings.shape[0] < 2:
            print(
                f"[WARN] embedding 少于 2 个，跳过：{cow_id}"
            )
            continue

        scores = calc_center_euclidean_scores(
            embeddings
        )

        threshold_info = build_threshold_mean_std(
            scores=scores,
            std_scale=std_scale,
        )

        threshold = threshold_info["threshold"]

        y_true: List[int] = []
        y_pred: List[int] = []

        num_unmatched_gt = 0
        normal_to_abnormal = 0
        abnormal_to_normal = 0

        for path, score in zip(
            embedding_paths,
            scores,
        ):
            filename = Path(path).name

            pred = int(
                float(score) <= threshold
            )

            if filename not in gt_dict:
                num_unmatched_gt += 1
                continue

            gt = int(gt_dict[filename])

            y_true.append(gt)
            y_pred.append(pred)

            if gt == 1 and pred == 0:
                normal_to_abnormal += 1
            elif gt == 0 and pred == 1:
                abnormal_to_normal += 1

        if not y_true:
            print(
                f"[WARN] 无可匹配 GT 样本，跳过：{cow_id}"
            )
            continue

        metrics = calc_metrics_binary(
            y_true=y_true,
            y_pred=y_pred,
        )

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        per_cow_rows.append({
            "cow_id": cow_id,
            "num_ori_images": len(ori_paths),
            "num_matched_gt": len(y_true),
            "num_unmatched_gt": num_unmatched_gt,
            "score_mean": threshold_info["mean"],
            "score_std": threshold_info["std"],
            "threshold": threshold,
            "accuracy": metrics["accuracy"],
            "precision_normal": metrics["precision_normal"],
            "recall_normal": metrics["recall_normal"],
            "f1_normal": metrics["f1_normal"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "num_normal_to_abnormal": normal_to_abnormal,
            "num_abnormal_to_normal": abnormal_to_normal,
        })

        print(
            f"[INFO] matched/unmatched = "
            f"{len(y_true)}/{num_unmatched_gt}"
        )
        print(
            f"[INFO] mean/std          = "
            f"{threshold_info['mean']:.6f}/"
            f"{threshold_info['std']:.6f}"
        )
        print(
            f"[INFO] threshold         = "
            f"{threshold:.6f}"
        )
        print(f"[INFO] metrics           = {metrics}")
        print(
            f"[INFO] normal→abnormal   = "
            f"{normal_to_abnormal}"
        )
        print(
            f"[INFO] abnormal→normal   = "
            f"{abnormal_to_normal}"
        )

    if not all_y_true:
        raise ValueError(
            "没有任何有效评测样本，请检查路径和文件名。"
        )

    overall_micro = calc_metrics_binary(
        y_true=all_y_true,
        y_pred=all_y_pred,
    )

    macro_accuracy = safe_mean([
        row["accuracy"]
        for row in per_cow_rows
    ])
    macro_precision = safe_mean([
        row["precision_normal"]
        for row in per_cow_rows
    ])
    macro_recall = safe_mean([
        row["recall_normal"]
        for row in per_cow_rows
    ])
    macro_f1 = safe_mean([
        row["f1_normal"]
        for row in per_cow_rows
    ])

    overall_rows = [{
        "model_path": model_path,
        "std_scale": std_scale,
        "num_cows": len(per_cow_rows),
        "num_samples": len(all_y_true),
        "accuracy_micro": overall_micro["accuracy"],
        "precision_micro_normal": overall_micro[
            "precision_normal"
        ],
        "recall_micro_normal": overall_micro[
            "recall_normal"
        ],
        "f1_micro_normal": overall_micro[
            "f1_normal"
        ],
        "accuracy_macro": macro_accuracy,
        "precision_macro_normal": macro_precision,
        "recall_macro_normal": macro_recall,
        "f1_macro_normal": macro_f1,
        "tp": overall_micro["tp"],
        "tn": overall_micro["tn"],
        "fp": overall_micro["fp"],
        "fn": overall_micro["fn"],
    }]

    save_csv(
        rows=per_cow_rows,
        csv_path=per_cow_csv,
    )

    save_csv(
        rows=overall_rows,
        csv_path=overall_csv,
    )

    print("\n" + "#" * 80)
    print("[OVERALL MICRO]")
    print(overall_micro)
    print("[OVERALL MACRO]")
    print({
        "accuracy_macro": macro_accuracy,
        "precision_macro_normal": macro_precision,
        "recall_macro_normal": macro_recall,
        "f1_macro_normal": macro_f1,
        "num_cows": len(per_cow_rows),
    })
    print("#" * 80)


# =========================================================
# 8. main：只改这里
# =========================================================
def main():
    model_name = "exp_ablation_ResNet50"

    split_json_path = (
        r"D:\datasets\cow_recognition8.3"
        r"\openset\openset_ori_8.2\split.json"

    )

    ori_root = (
        r"D:\code_project\triplet_code"
        r"\data\dataset_obb_ori"
    )

    gt_root = (
        r"D:\datasets\cow_recognition8.3\openset\openset_ori_8.2"
    )

    model_path = (
        r"D:\code_project\cow_recognition\outputs"
        rf"\{model_name}\checkpoints\epoch_10.pth"
    )

    per_cow_csv = (
        rf"D:\code_project\cow_recognition\outputs\{model_name}"
        rf"\unseen_ori_per_cow_metrics.csv"
    )

    overall_csv = (
        rf"D:\code_project\cow_recognition\outputs\{model_name}"
        rf"\unseen_ori_overall_metrics.csv"
    )

    evaluate_unseen_metrics_only(
        split_json_path=split_json_path,
        ori_root=ori_root,
        gt_root=gt_root,
        model_path=model_path,
        per_cow_csv=per_cow_csv,
        overall_csv=overall_csv,
        batch_size=64,
        num_workers=0,
        seed=42,
        std_scale=1.0,
    )


if __name__ == "__main__":
    main()
