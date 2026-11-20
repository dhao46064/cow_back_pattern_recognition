# -*- coding: utf-8 -*-
r"""
Backbone 消融实验开放集测试。

严格沿用原始 test_sup_no_image.py 的评估逻辑：

1. 从 split.json 读取 unseen_cows；
2. 某头牛在 ori_root 中不存在：直接跳过；
3. 某头牛在 gt_root 中不存在：直接跳过；
4. normal / abnormal 文件夹不存在：不报错；
5. normal = 1，abnormal = 0；
6. 只评估文件名能够与 GT 匹配的图片；
7. embedding 进行 L2 归一化；
8. 每头牛单独计算 embedding 均值中心；
9. 使用欧氏距离作为异常分数；
10. threshold = mean + std_scale * std；
11. 保存每个模型的每头牛指标；
12. 保存所有模型的总体 Micro / cow-level Macro 指标。

只修改了模型加载部分，以支持：
- ResNet18
- ResNet50
- DenseNet121
- EfficientNet-B0
- Swin-T
"""

import csv
import json
import random
import sys

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# =========================================================
# 0. 项目路径
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

# 当前文件：
# cow_recognition/scripts/test/test_ablation.py
#
# parents[0] = scripts/test
# parents[1] = scripts
# parents[2] = cow_recognition
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if not SRC_ROOT.exists():
    raise FileNotFoundError(
        f"没有找到 src 目录：{SRC_ROOT}\n"
        f"当前测试脚本：{CURRENT_FILE}\n"
        f"计算得到的项目根目录：{PROJECT_ROOT}"
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 90)
print(f"[SCRIPT]       {CURRENT_FILE}")
print(f"[PROJECT_ROOT] {PROJECT_ROOT}")
print(f"[SRC_ROOT]     {SRC_ROOT}")
print("=" * 90)


from src.models.arcface_model_ablation import ArcFaceCowModel


# =========================================================
# 1. 模型配置：只在这里修改 checkpoint
# =========================================================

MODELS = {
    "ResNet18_epoch10": {
        "path": (
            r"D:\code_project\cow_recognition\outputs"
            r"\exp_ablation_resnet18\checkpoints\epoch_10.pth"
        ),
        "backbone": "resnet18",
    },

    "ResNet50_epoch10": {
        "path": (
            r"D:\code_project\cow_recognition\outputs"
            r"\exp_ablation_ResNet50\checkpoints\epoch_10.pth"
        ),
        "backbone": "resnet50",
    },

    "DenseNet121_epoch10": {
        "path": (
            r"D:\code_project\cow_recognition\outputs"
            r"\exp_ablation_densenet121\checkpoints\epoch_10.pth"
        ),
        "backbone": "densenet121",
    },

    "EfficientNetB0_epoch10": {
        "path": (
            r"D:\code_project\cow_recognition\outputs"
            r"\exp_ablation_EfficientNet_B0\checkpoints\epoch_10.pth"
        ),
        "backbone": "efficientnet_b0",
    },

    "SwinT_epoch10": {
        "path": (
            r"D:\code_project\cow_recognition\outputs"
            r"\exp_ablation_Swin_T\checkpoints\epoch_10.pth"
        ),
        "backbone": "swin_t",
    },

    # 训练完成 HRNet 后再启用：
    #
    # "HRNetW32_epoch10": {
    #     "path": (
    #         r"D:\code_project\cow_recognition\outputs"
    #         r"\exp_ablation_HRNet_W32\checkpoints\epoch_10.pth"
    #     ),
    #     "backbone": "hrnet_w32",
    # },
}


# =========================================================
# 2. 数据路径
# =========================================================

SPLIT_JSON = (
    r"D:\datasets\cow_recognition8.3"
    r"\openset\openset_ori_8.2\split.json"
)

ORI_ROOT = (
    r"D:\code_project\triplet_code"
    r"\data\dataset_obb_ori"
)

GT_ROOT = (
    r"D:\datasets\cow_recognition8.3"
    r"\openset\openset_ori_8.2"
)
# GT_ROOT = Path(
#     r"D:\datasets\cow_recognition8.3"
#     r"\openset\openset_consensus_hard_11to1_v2"
# )

# # 新生成的混合数据集，结构类似 dataset_obb_ori
# ORI_ROOT = Path(
#     r"D:\datasets\cow_recognition8.3"
#     r"\openset\openset_consensus_hard_11to1_v2_mixed"
# )


OUTPUT_DIR = Path(
    r"D:\code_project\cow_recognition"
    r"\outputs\ablation_openset_evaluation2"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "ablation_overall_micro_macro_metrics.csv"
)


# =========================================================
# 3. 测试参数
# =========================================================

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 0
SEED = 42
STD_SCALE = 1.0

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

IMG_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# =========================================================
# 4. 随机种子
# =========================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


# =========================================================
# 5. 图像变换
# =========================================================

def build_eval_transforms(
    img_size: int = 224,
):
    return transforms.Compose([
        transforms.Resize(
            (img_size, img_size)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])


# =========================================================
# 6. checkpoint 加载
# =========================================================

def load_checkpoint(
    model_path: str,
) -> Tuple[dict, dict]:
    checkpoint_path = Path(model_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint 不存在：{checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):
        state_dict = checkpoint["model"]

    elif (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
        ]

    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"checkpoint 中没有有效 state_dict："
            f"{checkpoint_path}"
        )

    return checkpoint, state_dict


def infer_arcface_config(
    state_dict: dict,
) -> Tuple[int, int]:
    if "arcface.weight" not in state_dict:
        raise KeyError(
            "checkpoint 中没有 arcface.weight，"
            "无法推断 num_classes 和 embedding_dim。"
        )

    weight = state_dict[
        "arcface.weight"
    ]

    num_classes = int(
        weight.shape[0]
    )

    embedding_dim = int(
        weight.shape[1]
    )

    return num_classes, embedding_dim


def build_model_from_ckpt(
    model_path: str,
    backbone: str,
    device: str,
):
    checkpoint, state_dict = load_checkpoint(
        model_path=model_path
    )

    num_classes, embedding_dim = (
        infer_arcface_config(
            state_dict
        )
    )

    checkpoint_epoch = (
        checkpoint.get(
            "epoch",
            "unknown",
        )
        if isinstance(checkpoint, dict)
        else "unknown"
    )

    print("=" * 90)
    print(f"[MODEL PATH]       {model_path}")
    print(f"[BACKBONE]         {backbone}")
    print(f"[CHECKPOINT EPOCH] {checkpoint_epoch}")
    print(f"[NUM CLASSES]      {num_classes}")
    print(f"[EMBEDDING DIM]    {embedding_dim}")
    print("=" * 90)

    model = ArcFaceCowModel(
        num_classes=num_classes,
        backbone=backbone,
        embedding_dim=embedding_dim,
        pretrained=False,
    )

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    print(
        f"[LOAD] missing keys    = "
        f"{len(missing_keys)}"
    )
    print(
        f"[LOAD] unexpected keys = "
        f"{len(unexpected_keys)}"
    )

    if missing_keys:
        print(
            f"[WARN] first missing keys: "
            f"{missing_keys[:10]}"
        )

    if unexpected_keys:
        print(
            f"[WARN] first unexpected keys: "
            f"{unexpected_keys[:10]}"
        )

    # 避免模型结构没有正确加载，却继续产生错误指标。
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "checkpoint 与当前模型结构不完全匹配。\n"
            f"backbone={backbone}\n"
            f"missing={len(missing_keys)}\n"
            f"unexpected={len(unexpected_keys)}\n"
            "请检查训练和测试是否使用同一个 "
            "arcface_model_ablation.py。"
        )

    model = model.to(device)
    model.eval()

    return model, checkpoint_epoch


# =========================================================
# 7. split.json 与文件读取
# =========================================================

def load_split_ids(
    split_json_path: str,
    split_key: str = "unseen_cows",
) -> List[str]:
    path = Path(split_json_path)

    if not path.exists():
        raise FileNotFoundError(
            f"split.json 不存在：{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if split_key not in data:
        raise KeyError(
            f"{path} 中不存在键：{split_key}"
        )

    return [
        str(cow_id)
        for cow_id in data[split_key]
    ]


def list_images_in_dir(
    dir_path: Path,
) -> List[str]:
    """
    严格沿用原始代码。

    目录不存在：
        返回空列表，不报错。

    目录存在：
        只读取合法图片文件。
    """
    if not dir_path.exists():
        return []

    return sorted(
        str(path)
        for path in dir_path.iterdir()
        if (
            path.is_file()
            and is_image_file(path)
        )
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
    严格沿用原始代码：

    normal = 1
    abnormal = 0

    cow 目录不存在：
        返回空字典。

    normal 文件夹不存在：
        不读取 normal。

    abnormal 文件夹不存在：
        不读取 abnormal。
    """
    gt_dict: Dict[str, int] = {}

    cow_dir = (
        Path(gt_root)
        / cow_id
    )

    normal_dir = (
        cow_dir
        / "normal"
    )

    abnormal_dir = (
        cow_dir
        / "abnormal"
    )

    if normal_dir.exists():
        for path in normal_dir.iterdir():
            if (
                path.is_file()
                and is_image_file(path)
            ):
                gt_dict[path.name] = 1

    if abnormal_dir.exists():
        for path in abnormal_dir.iterdir():
            if (
                path.is_file()
                and is_image_file(path)
            ):
                gt_dict[path.name] = 0

    return gt_dict


# =========================================================
# 8. Dataset
# =========================================================

class SimpleImageDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        transform=None,
    ):
        # 必须使用 str，避免 DataLoader 无法处理 WindowsPath。
        self.image_paths = [
            str(path)
            for path in image_paths
        ]

        self.transform = transform

    def __len__(self) -> int:
        return len(
            self.image_paths
        )

    def __getitem__(
        self,
        index: int,
    ):
        path = self.image_paths[index]

        try:
            with Image.open(path) as image:
                image = image.convert(
                    "RGB"
                )

                if self.transform is not None:
                    image = self.transform(
                        image
                    )

        except Exception as exc:
            raise RuntimeError(
                f"图像读取失败：{path}"
            ) from exc

        return image, path


# =========================================================
# 9. embedding 提取
# =========================================================

@torch.inference_mode()
def extract_embeddings_for_paths(
    model,
    image_paths: List[str],
    transform,
    device: str,
    batch_size: int = 64,
    num_workers: int = 0,
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
        pin_memory=(
            device.startswith("cuda")
        ),
        persistent_workers=(
            num_workers > 0
        ),
    )

    all_embeddings = []
    all_paths: List[str] = []

    for images, paths in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        embeddings = (
            model.forward_emb(images)
        )

        # 严格沿用原始 L2 归一化。
        embeddings = (
            embeddings
            / (
                embeddings.norm(
                    dim=1,
                    keepdim=True,
                )
                + 1e-12
            )
        )

        all_embeddings.append(
            embeddings
            .cpu()
            .numpy()
        )

        all_paths.extend(
            list(paths)
        )

    if not all_embeddings:
        return (
            np.zeros(
                (0, 1),
                dtype=np.float32,
            ),
            [],
        )

    embeddings_np = np.concatenate(
        all_embeddings,
        axis=0,
    ).astype(np.float32)

    return (
        embeddings_np,
        all_paths,
    )


# =========================================================
# 10. 分数与阈值
# =========================================================

def calc_center_euclidean_scores(
    embeddings: np.ndarray,
) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings 必须为二维矩阵 "
            f"[N,D]，当前形状：{embeddings.shape}"
        )

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

    return scores.astype(
        np.float32
    )


def build_threshold_mean_std(
    scores: np.ndarray,
    std_scale: float = 1.0,
) -> Dict[str, float]:
    if scores.size == 0:
        raise ValueError(
            "scores 为空，无法计算阈值。"
        )

    mean = float(
        np.mean(scores)
    )

    std = float(
        np.std(scores)
    )

    threshold = (
        mean
        + std_scale * std
    )

    return {
        "mean": mean,
        "std": std,
        "threshold": float(
            threshold
        ),
    }


# =========================================================
# 11. 二分类指标
# =========================================================

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

    if y_true_array.size == 0:
        raise ValueError(
            "y_true 为空，无法计算指标。"
        )

    if (
        y_true_array.shape
        != y_pred_array.shape
    ):
        raise ValueError(
            "y_true 与 y_pred 长度不一致。"
        )

    # normal = 1
    # abnormal = 0

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

    accuracy = (
        (tp + tn)
        / max(
            len(y_true_array),
            1,
        )
    )

    precision = (
        tp
        / max(
            tp + fp,
            1e-12,
        )
    )

    recall = (
        tp
        / max(
            tp + fn,
            1e-12,
        )
    )

    f1 = (
        2.0
        * precision
        * recall
        / max(
            precision + recall,
            1e-12,
        )
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "precision_normal": float(
            precision
        ),
        "recall_normal": float(
            recall
        ),
        "f1_normal": float(
            f1
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def safe_mean(
    values: List[float],
) -> float:
    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


# =========================================================
# 12. CSV 保存
# =========================================================

def save_csv(
    rows: List[dict],
    csv_path: Path,
) -> None:
    if not rows:
        raise ValueError(
            f"没有可保存的数据：{csv_path}"
        )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[INFO] CSV saved to: "
        f"{csv_path}"
    )


# =========================================================
# 13. 单模型评估
# =========================================================

def evaluate_one_model(
    model_name: str,
    backbone: str,
    model_path: str,
    unseen_cows: List[str],
    transform,
    device: str,
    batch_size: int,
    num_workers: int,
    std_scale: float,
) -> Tuple[List[dict], dict]:
    model, checkpoint_epoch = (
        build_model_from_ckpt(
            model_path=model_path,
            backbone=backbone,
            device=device,
        )
    )

    all_y_true: List[int] = []
    all_y_pred: List[int] = []

    per_cow_rows: List[dict] = []

    skipped_ori_missing = 0
    skipped_gt_missing = 0
    skipped_no_match = 0

    for cow_id in unseen_cows:
        print("\n" + "=" * 80)
        print(
            f"[INFO] Processing cow_id = "
            f"{cow_id}"
        )

        # 原始逻辑：
        # 目录不存在时返回空列表，不会执行 iterdir。
        ori_paths = list_ori_images_for_cow(
            ori_root=ORI_ROOT,
            cow_id=cow_id,
        )

        # 原始逻辑：
        # cow/normal 或 cow/abnormal 不存在时，
        # 返回空 GT，不会报错。
        gt_dict = build_gt_dict_for_cow(
            gt_root=GT_ROOT,
            cow_id=cow_id,
        )

        if len(ori_paths) < 2:
            skipped_ori_missing += 1

            print(
                f"[WARN] ori 目录不存在或图片少于 2 张，"
                f"跳过：{cow_id}"
            )
            continue

        if not gt_dict:
            skipped_gt_missing += 1

            print(
                f"[WARN] GT 牛目录不存在或 GT 为空，"
                f"跳过：{cow_id}"
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
            skipped_ori_missing += 1

            print(
                f"[WARN] embedding 少于 2 个，"
                f"跳过：{cow_id}"
            )
            continue

        scores = (
            calc_center_euclidean_scores(
                embeddings
            )
        )

        threshold_info = (
            build_threshold_mean_std(
                scores=scores,
                std_scale=std_scale,
            )
        )

        threshold = threshold_info[
            "threshold"
        ]

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

            prediction = int(
                float(score)
                <= threshold
            )

            # 严格沿用原始逻辑：
            # 未匹配 GT 的文件直接跳过，
            # 不能自动当成 abnormal。
            if filename not in gt_dict:
                num_unmatched_gt += 1
                continue

            ground_truth = int(
                gt_dict[filename]
            )

            y_true.append(
                ground_truth
            )

            y_pred.append(
                prediction
            )

            if (
                ground_truth == 1
                and prediction == 0
            ):
                normal_to_abnormal += 1

            elif (
                ground_truth == 0
                and prediction == 1
            ):
                abnormal_to_normal += 1

        if not y_true:
            skipped_no_match += 1

            print(
                f"[WARN] 没有任何文件名能够与 GT 对齐，"
                f"跳过：{cow_id}"
            )
            continue

        metrics = calc_metrics_binary(
            y_true=y_true,
            y_pred=y_pred,
        )

        all_y_true.extend(
            y_true
        )

        all_y_pred.extend(
            y_pred
        )

        per_cow_rows.append({
            "model": model_name,
            "backbone": backbone,
            "checkpoint_epoch": (
                checkpoint_epoch
            ),
            "cow_id": cow_id,
            "num_ori_images": len(
                ori_paths
            ),
            "num_matched_gt": len(
                y_true
            ),
            "num_unmatched_gt": (
                num_unmatched_gt
            ),
            "score_mean": (
                threshold_info["mean"]
            ),
            "score_std": (
                threshold_info["std"]
            ),
            "threshold": threshold,
            "accuracy": (
                metrics["accuracy"]
            ),
            "precision_normal": (
                metrics[
                    "precision_normal"
                ]
            ),
            "recall_normal": (
                metrics[
                    "recall_normal"
                ]
            ),
            "f1_normal": (
                metrics["f1_normal"]
            ),
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "num_normal_to_abnormal": (
                normal_to_abnormal
            ),
            "num_abnormal_to_normal": (
                abnormal_to_normal
            ),
        })

        print(
            f"[INFO] matched/unmatched = "
            f"{len(y_true)}/"
            f"{num_unmatched_gt}"
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

        print(
            f"[INFO] metrics           = "
            f"{metrics}"
        )

        print(
            f"[INFO] normal→abnormal   = "
            f"{normal_to_abnormal}"
        )

        print(
            f"[INFO] abnormal→normal   = "
            f"{abnormal_to_normal}"
        )

    if not all_y_true:
        raise RuntimeError(
            f"模型 {model_name} 没有任何有效评测样本。\n"
            "请检查 split.json、ori_root、gt_root "
            "以及文件名对应关系。"
        )

    # -----------------------------------------------------
    # Micro：所有有效图片合并计算
    # -----------------------------------------------------

    micro_metrics = calc_metrics_binary(
        y_true=all_y_true,
        y_pred=all_y_pred,
    )

    # -----------------------------------------------------
    # Macro：每头牛先算指标，再对牛等权平均
    # -----------------------------------------------------

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

    overall_row = {
        "model": model_name,
        "backbone": backbone,
        "model_path": model_path,
        "checkpoint_epoch": (
            checkpoint_epoch
        ),
        "std_scale": std_scale,
        "num_cows": len(
            per_cow_rows
        ),
        "num_samples": len(
            all_y_true
        ),

        "accuracy_micro": (
            micro_metrics["accuracy"]
        ),
        "precision_micro_normal": (
            micro_metrics[
                "precision_normal"
            ]
        ),
        "recall_micro_normal": (
            micro_metrics[
                "recall_normal"
            ]
        ),
        "f1_micro_normal": (
            micro_metrics[
                "f1_normal"
            ]
        ),

        "accuracy_macro": (
            macro_accuracy
        ),
        "precision_macro_normal": (
            macro_precision
        ),
        "recall_macro_normal": (
            macro_recall
        ),
        "f1_macro_normal": (
            macro_f1
        ),

        "tp": micro_metrics["tp"],
        "tn": micro_metrics["tn"],
        "fp": micro_metrics["fp"],
        "fn": micro_metrics["fn"],

        "skipped_ori_missing_or_lt2": (
            skipped_ori_missing
        ),
        "skipped_gt_missing_or_empty": (
            skipped_gt_missing
        ),
        "skipped_no_matched_gt": (
            skipped_no_match
        ),
    }

    print("\n" + "#" * 90)
    print(
        f"[MODEL RESULT] {model_name}"
    )
    print("[OVERALL MICRO]")
    print(micro_metrics)
    print("[OVERALL MACRO]")
    print({
        "accuracy_macro": (
            macro_accuracy
        ),
        "precision_macro_normal": (
            macro_precision
        ),
        "recall_macro_normal": (
            macro_recall
        ),
        "f1_macro_normal": (
            macro_f1
        ),
        "num_cows": len(
            per_cow_rows
        ),
    })
    print("[SKIPPED]")
    print({
        "ori_missing_or_lt2": (
            skipped_ori_missing
        ),
        "gt_missing_or_empty": (
            skipped_gt_missing
        ),
        "no_matched_gt": (
            skipped_no_match
        ),
    })
    print("#" * 90)

    # 及时释放当前模型显存。
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        per_cow_rows,
        overall_row,
    )


# =========================================================
# 14. 主函数
# =========================================================

def main() -> None:
    set_seed(SEED)

    split_path = Path(
        SPLIT_JSON
    )

    ori_root_path = Path(
        ORI_ROOT
    )

    gt_root_path = Path(
        GT_ROOT
    )

    if not split_path.exists():
        raise FileNotFoundError(
            f"split.json 不存在："
            f"{split_path}"
        )

    if not ori_root_path.exists():
        raise FileNotFoundError(
            f"ori_root 不存在："
            f"{ori_root_path}"
        )

    if not gt_root_path.exists():
        raise FileNotFoundError(
            f"gt_root 不存在："
            f"{gt_root_path}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    transform = build_eval_transforms(
        img_size=IMG_SIZE
    )

    unseen_cows = load_split_ids(
        split_json_path=SPLIT_JSON,
        split_key="unseen_cows",
    )

    print("=" * 90)
    print(f"[DEVICE]       {DEVICE}")
    print(f"[UNSEEN COWS]  {len(unseen_cows)}")
    print(f"[SPLIT JSON]   {SPLIT_JSON}")
    print(f"[ORI ROOT]     {ORI_ROOT}")
    print(f"[GT ROOT]      {GT_ROOT}")
    print(f"[OUTPUT DIR]   {OUTPUT_DIR}")
    print(f"[BATCH SIZE]   {BATCH_SIZE}")
    print(f"[STD SCALE]    {STD_SCALE}")
    print("=" * 90)

    overall_rows: List[dict] = []

    for model_name, config in MODELS.items():
        model_path = str(
            config["path"]
        )

        backbone = str(
            config["backbone"]
        )

        print("\n" + "=" * 100)
        print(
            f"[START MODEL] {model_name}"
        )
        print(
            f"[BACKBONE]    {backbone}"
        )
        print(
            f"[CHECKPOINT]  {model_path}"
        )
        print("=" * 100)

        # checkpoint 不存在时清晰提示并跳过，
        # 不影响已经配置正确的其他模型。
        if not Path(model_path).exists():
            print(
                f"[ERROR] checkpoint 不存在，"
                f"跳过模型：{model_path}"
            )
            continue

        per_cow_rows, overall_row = (
            evaluate_one_model(
                model_name=model_name,
                backbone=backbone,
                model_path=model_path,
                unseen_cows=unseen_cows,
                transform=transform,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                std_scale=STD_SCALE,
            )
        )

        per_cow_csv = (
            OUTPUT_DIR
            / (
                f"{model_name}"
                "_per_cow_metrics.csv"
            )
        )

        save_csv(
            rows=per_cow_rows,
            csv_path=per_cow_csv,
        )

        overall_rows.append(
            overall_row
        )

    if not overall_rows:
        raise RuntimeError(
            "没有任何模型成功完成评估，"
            "请检查 MODELS 中的 checkpoint 路径。"
        )

    save_csv(
        rows=overall_rows,
        csv_path=SUMMARY_CSV,
    )

    print("\n" + "=" * 100)
    print("[ALL MODELS FINISHED]")
    print(
        f"[SUMMARY CSV] {SUMMARY_CSV}"
    )
    print("=" * 100)

    for row in overall_rows:
        print({
            "model": row["model"],
            "accuracy_micro": (
                row["accuracy_micro"]
            ),
            "precision_micro_normal": (
                row[
                    "precision_micro_normal"
                ]
            ),
            "recall_micro_normal": (
                row[
                    "recall_micro_normal"
                ]
            ),
            "f1_micro_normal": (
                row["f1_micro_normal"]
            ),
            "accuracy_macro": (
                row["accuracy_macro"]
            ),
            "precision_macro_normal": (
                row[
                    "precision_macro_normal"
                ]
            ),
            "recall_macro_normal": (
                row[
                    "recall_macro_normal"
                ]
            ),
            "f1_macro_normal": (
                row["f1_macro_normal"]
            ),
            "num_cows": row["num_cows"],
            "num_samples": (
                row["num_samples"]
            ),
        })


if __name__ == "__main__":
    main()