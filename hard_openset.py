# -*- coding: utf-8 -*-

"""
多模型共识困难开放集构建脚本 V2

=========================================================
一、目标牛
=========================================================

目标牛来自：

    D:\datasets\cow_recognition8.3
    \openset\openset_ori_8.2

每个目标牛使用它的全部 normal 建立特征中心。


=========================================================
二、全局 abnormal 候选池
=========================================================

候选池由三部分组成：

1. train_dataset 每个编号 normal 随机抽取 3 张；
2. train_dataset 每个编号全部 abnormal；
3. openset_ori_8.2 每个编号 normal 随机抽取 3 张。

不使用 openset_ori_8.2 原有 abnormal。


=========================================================
三、同编号过滤
=========================================================

针对目标牛 target_cow_id：

1. 如果候选的已知身份编号等于 target_cow_id，删除；
2. 如果候选原文件名中包含 target_cow_id，删除。

避免将目标牛自身图片作为 abnormal。


=========================================================
四、多模型共识
=========================================================

使用模型：

- ResNet18
- ResNet50
- DenseNet121
- EfficientNet-B0
- Swin-T

每个模型：

1. 使用目标牛全部 normal embedding 建立中心；
2. 计算候选图片到该中心的余弦距离；
3. 对候选按距离从小到大排名；
4. 距离越小，候选越像目标牛，越困难。

最终按照：

1. 模型投票数量；
2. 多模型归一化排名得分；
3. 多模型平均排名；
4. 多模型平均距离；

选取共识困难 abnormal。


=========================================================
五、最终比例
=========================================================

每头牛：

    abnormal数量 ≈ normal数量 / 11

最终维持：

    normal : abnormal ≈ 11 : 1


=========================================================
六、输出
=========================================================

OUTPUT_ROOT/
├── cow_id/
│   ├── normal/
│   └── abnormal/
├── labels.csv
├── selected_abnormal_manifest.csv
├── dataset_summary.csv
├── candidate_pool_manifest.csv
├── candidate_sources_before_dedup.csv
├── config.json
├── split.json
└── score_audit/
"""

import csv
import gc
import hashlib
import json
import random
import re
import shutil
import sys

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tqdm import tqdm


# =========================================================
# 0. 自动寻找项目根目录
# =========================================================

CURRENT_FILE = Path(__file__).resolve()


def find_project_root(
    current_file: Path,
) -> Path:
    """
    从当前脚本位置逐级向上查找包含 src 的目录。
    """

    search_dirs = [
        current_file.parent,
        *current_file.parents,
    ]

    for directory in search_dirs:
        if (directory / "src").is_dir():
            return directory

    searched = "\n".join(
        str(directory)
        for directory in search_dirs
    )

    raise FileNotFoundError(
        "没有找到包含 src 文件夹的项目根目录。\n"
        f"当前脚本：{current_file}\n"
        f"已经搜索：\n{searched}"
    )


PROJECT_ROOT = find_project_root(
    CURRENT_FILE
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


print("=" * 100)
print(f"[SCRIPT]       {CURRENT_FILE}")
print(f"[PROJECT_ROOT] {PROJECT_ROOT}")
print(f"[SRC_ROOT]     {PROJECT_ROOT / 'src'}")
print("=" * 100)


from src.models.arcface_model_ablation import (
    ArcFaceCowModel,
)


# =========================================================
# 1. 输入与输出路径
# =========================================================

TRAIN_ROOT = Path(
    r"D:\datasets\cow_recognition8.3"
    r"\train_dataset"
)

OPENSET_ROOT = Path(
    r"D:\datasets\cow_recognition8.3"
    r"\openset\openset_ori_8.2"
)

OUTPUT_ROOT = Path(
    r"D:\datasets\cow_recognition8.3"
    r"\openset\openset_consensus_hard_11to1_v2"
)


# =========================================================
# 2. 模型配置
# =========================================================

MODELS = {
    "ResNet18": {
        "path": (
            r"D:\code_project\cow_recognition\outputs"
            r"\exp_ablation_resnet18"
            r"\checkpoints\epoch_10.pth"
        ),
        "backbone": "resnet18",
    },

    "ResNet50": {
        "path": (
            r"D:\code_project\cow_recognition\outputs"
            r"\exp_ablation_ResNet50"
            r"\checkpoints\epoch_10.pth"
        ),
        "backbone": "resnet50",
    },

    "DenseNet121": {
        "path": (
            r"D:\code_project\cow_recognition\outputs"
            r"\exp_ablation_densenet121"
            r"\checkpoints\epoch_10.pth"
        ),
        "backbone": "densenet121",
    },

    "EfficientNetB0": {
        "path": (
            r"D:\code_project\cow_recognition\outputs"
            r"\exp_ablation_EfficientNet_B0"
            r"\checkpoints\epoch_10.pth"
        ),
        "backbone": "efficientnet_b0",
    },

    "SwinT": {
        "path": (
            r"D:\code_project\cow_recognition\outputs"
            r"\exp_ablation_Swin_T"
            r"\checkpoints\epoch_10.pth"
        ),
        "backbone": "swin_t",
    },
}


# =========================================================
# 3. 参数
# =========================================================

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 0

SEED = 42

# 训练集每个编号抽取的 normal 数量
TRAIN_NORMAL_SAMPLE_PER_COW = 3

# 开放集每个编号抽取的 normal 数量
OPENSET_NORMAL_SAMPLE_PER_COW = 3

# 最终比例 normal : abnormal ≈ 11 : 1
NORMAL_ABNORMAL_RATIO = 11.0

# 目标牛至少需要多少张 normal
MIN_TARGET_NORMAL = 2

# 每个模型至少取前多少名进行投票
MODEL_TOP_N_MIN = 100

# 每个模型投票区间至少为最终Top-K的多少倍
MODEL_TOP_N_FACTOR = 5

# 每头牛保存前多少个候选用于人工检查
SCORE_AUDIT_TOP_N = 200

# 是否按文件内容SHA1去重
DEDUPLICATE_BY_SHA1 = True

# 运行前是否删除旧输出
CLEAR_OUTPUT_ROOT = True

# 文件名匹配模式：
#
# boundary：
# 目标编号前后不能继续连接数字。
#
# substring：
# 只要文件名中出现目标编号就过滤。
SAME_ID_MATCH_MODE = "boundary"


DEVICE = torch.device(
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
# 4. 基础函数
# =========================================================

def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clear_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_image_file(
    path: Path,
) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in IMG_EXTS
    )


def list_images(
    folder: Path,
) -> List[Path]:
    """
    文件夹不存在时返回空列表。
    """

    if not folder.exists():
        return []

    return sorted(
        [
            path
            for path in folder.iterdir()
            if is_image_file(path)
        ],
        key=lambda path: path.name,
    )


def discover_cow_dirs(
    root: Path,
) -> Dict[str, Path]:
    """
    发现数据集下的牛编号一级目录。
    """

    result: Dict[str, Path] = {}

    if not root.exists():
        return result

    for path in root.iterdir():
        if not path.is_dir():
            continue

        normal_dir = path / "normal"
        abnormal_dir = path / "abnormal"

        if normal_dir.exists() or abnormal_dir.exists():
            result[str(path.name)] = path

    return dict(
        sorted(
            result.items(),
            key=lambda item: item[0],
        )
    )


def safe_column_name(
    name: str,
) -> str:
    value = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        name,
    )

    return value.strip("_")


def sha1_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    sha1 = hashlib.sha1()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            sha1.update(chunk)

    return sha1.hexdigest()


def stable_path_hash(
    path: Path,
) -> str:
    return hashlib.sha1(
        str(path.resolve())
        .encode("utf-8")
    ).hexdigest()


def conventional_round(
    value: float,
) -> int:
    """
    使用常规四舍五入，不使用Python的银行家舍入。
    """

    return int(
        np.floor(value + 0.5)
    )


def calculate_abnormal_count(
    normal_count: int,
) -> int:
    return max(
        1,
        conventional_round(
            normal_count
            / NORMAL_ABNORMAL_RATIO
        ),
    )


# =========================================================
# 5. CSV保存
# =========================================================

def save_csv(
    rows: List[dict],
    save_path: Path,
    fieldnames: Optional[List[str]] = None,
) -> None:
    if not rows:
        print(
            f"[WARN] 无数据，未保存：{save_path}"
        )
        return

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if fieldnames is None:
        fieldnames = list(
            rows[0].keys()
        )

    with save_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[SAVE] {save_path}"
    )


# =========================================================
# 6. 文件名编号匹配
# =========================================================

def filename_matches_cow_id(
    filename: str,
    cow_id: str,
) -> bool:
    """
    检查文件名中是否包含指定牛编号。
    """

    stem = Path(filename).stem

    target = str(cow_id)

    if SAME_ID_MATCH_MODE == "substring":
        return target in stem

    if SAME_ID_MATCH_MODE == "boundary":
        pattern = (
            rf"(?<!\d)"
            rf"{re.escape(target)}"
            rf"(?!\d)"
        )

        return (
            re.search(
                pattern,
                stem,
            )
            is not None
        )

    raise ValueError(
        "SAME_ID_MATCH_MODE只能为"
        "'boundary'或'substring'"
    )


def find_known_ids_in_filename(
    filename: str,
    known_cow_ids: Set[str],
) -> Set[str]:
    """
    从文件名中寻找已知的牛编号。

    只匹配训练集或开放集中真实存在的编号，
    避免将日期、时间等数字误当作牛编号。
    """

    matched_ids: Set[str] = set()

    for cow_id in known_cow_ids:
        if filename_matches_cow_id(
            filename=filename,
            cow_id=cow_id,
        ):
            matched_ids.add(
                cow_id
            )

    return matched_ids


# =========================================================
# 7. 读取目标牛全部normal
# =========================================================

def collect_target_normals(
    openset_cow_dirs: Dict[str, Path],
) -> Dict[str, List[dict]]:
    """
    目标牛来自开放集。

    每头目标牛使用全部normal建立中心，
    最终也复制全部normal。
    """

    result: Dict[
        str,
        List[dict]
    ] = {}

    for cow_id, cow_dir in tqdm(
        openset_cow_dirs.items(),
        desc="读取目标牛全部normal",
    ):
        images = list_images(
            cow_dir / "normal"
        )

        if len(images) < MIN_TARGET_NORMAL:
            print(
                f"[WARN] 目标牛 {cow_id} normal只有"
                f"{len(images)}张，跳过"
            )
            continue

        result[cow_id] = []

        for image_path in images:
            result[cow_id].append({
                "cow_id": cow_id,
                "path": str(image_path),
                "filename": image_path.name,
                "suffix": image_path.suffix.lower(),
            })

    return result


# =========================================================
# 8. 候选池记录
# =========================================================

def create_candidate_record(
    path: Path,
    source_type: str,
    source_parent_cow_id: str,
    identity_cow_ids: Set[str],
) -> dict:
    return {
        "path": str(path),
        "filename": path.name,
        "suffix": path.suffix.lower(),

        "source_type": source_type,

        # 原来位于哪个牛编号目录下
        "source_parent_cow_id": str(
            source_parent_cow_id
        ),

        # 能够确定的图片真实身份编号
        "identity_cow_ids": set(
            str(x)
            for x in identity_cow_ids
        ),
    }


# =========================================================
# 9. 每编号normal随机抽3张
# =========================================================

def sample_normal_candidates(
    cow_dirs: Dict[str, Path],
    source_type: str,
    sample_per_cow: int,
) -> Tuple[List[dict], List[dict]]:
    """
    从每个编号的normal中固定随机抽取若干张。

    source_type：
        train_normal
        openset_normal
    """

    records: List[dict] = []
    sample_manifest: List[dict] = []

    for cow_id, cow_dir in tqdm(
        cow_dirs.items(),
        desc=f"抽取{source_type}",
    ):
        normal_images = list_images(
            cow_dir / "normal"
        )

        if not normal_images:
            continue

        sample_count = min(
            sample_per_cow,
            len(normal_images),
        )

        # 每个编号使用独立固定随机种子。
        rng = random.Random(
            f"{SEED}|{source_type}|{cow_id}"
        )

        if len(normal_images) > sample_count:
            selected_images = rng.sample(
                normal_images,
                sample_count,
            )
        else:
            selected_images = list(
                normal_images
            )

        selected_images = sorted(
            selected_images,
            key=lambda path: path.name,
        )

        for index, image_path in enumerate(
            selected_images,
            start=1,
        ):
            records.append(
                create_candidate_record(
                    path=image_path,
                    source_type=source_type,
                    source_parent_cow_id=cow_id,
                    identity_cow_ids={
                        str(cow_id)
                    },
                )
            )

            sample_manifest.append({
                "source_type": source_type,
                "source_cow_id": cow_id,
                "selected_index": index,
                "available_normal_count": len(
                    normal_images
                ),
                "selected_normal_count": sample_count,
                "filename": image_path.name,
                "path": str(image_path),
            })

    return records, sample_manifest


# =========================================================
# 10. 训练集全部abnormal
# =========================================================

def collect_train_abnormal_candidates(
    train_cow_dirs: Dict[str, Path],
    known_cow_ids: Set[str],
) -> Tuple[List[dict], List[dict]]:
    """
    收集训练集全部abnormal。

    abnormal所在父目录不代表图片真实身份，
    因此尝试从文件名中匹配已知牛编号。
    """

    records: List[dict] = []
    manifest_rows: List[dict] = []

    for parent_cow_id, cow_dir in tqdm(
        train_cow_dirs.items(),
        desc="收集训练集全部abnormal",
    ):
        abnormal_images = list_images(
            cow_dir / "abnormal"
        )

        for image_path in abnormal_images:
            matched_ids = (
                find_known_ids_in_filename(
                    filename=image_path.name,
                    known_cow_ids=known_cow_ids,
                )
            )

            records.append(
                create_candidate_record(
                    path=image_path,
                    source_type="train_abnormal",
                    source_parent_cow_id=(
                        parent_cow_id
                    ),
                    identity_cow_ids=matched_ids,
                )
            )

            manifest_rows.append({
                "source_type": "train_abnormal",
                "source_parent_cow_id": (
                    parent_cow_id
                ),
                "matched_identity_cow_ids": (
                    ";".join(
                        sorted(matched_ids)
                    )
                ),
                "filename": image_path.name,
                "path": str(image_path),
            })

    return records, manifest_rows


# =========================================================
# 11. 候选池SHA1去重
# =========================================================

def deduplicate_candidates(
    raw_records: List[dict],
) -> List[dict]:
    """
    根据文件内容SHA1去重。

    同一图片出现在多个来源时：
    - 只保留一个实际路径；
    - 合并其所有来源；
    - 合并其所有已知身份编号。
    """

    unique_records: Dict[
        str,
        dict
    ] = {}

    failed_count = 0

    for record in tqdm(
        raw_records,
        desc="候选池SHA1去重",
    ):
        path = Path(
            record["path"]
        )

        try:
            if DEDUPLICATE_BY_SHA1:
                file_hash = sha1_file(
                    path
                )
            else:
                file_hash = stable_path_hash(
                    path
                )

        except Exception as exc:
            failed_count += 1

            print(
                f"[WARN] 文件读取失败，跳过："
                f"{path}\n{exc}"
            )
            continue

        if file_hash not in unique_records:
            unique_records[file_hash] = {
                "sha1": file_hash,

                # 实际用于后续推理与复制的代表路径
                "path": record["path"],
                "filename": record["filename"],
                "suffix": record["suffix"],

                "source_types": {
                    record["source_type"]
                },

                "source_parent_cow_ids": {
                    record[
                        "source_parent_cow_id"
                    ]
                },

                "identity_cow_ids": set(
                    record["identity_cow_ids"]
                ),

                "source_paths": {
                    record["path"]
                },

                "source_filenames": {
                    record["filename"]
                },

                "duplicate_count": 1,
            }

        else:
            saved = unique_records[
                file_hash
            ]

            saved["source_types"].add(
                record["source_type"]
            )

            saved[
                "source_parent_cow_ids"
            ].add(
                record[
                    "source_parent_cow_id"
                ]
            )

            saved[
                "identity_cow_ids"
            ].update(
                record["identity_cow_ids"]
            )

            saved["source_paths"].add(
                record["path"]
            )

            saved[
                "source_filenames"
            ].add(
                record["filename"]
            )

            saved["duplicate_count"] += 1

    result = list(
        unique_records.values()
    )

    result.sort(
        key=lambda row: row["sha1"]
    )

    for index, record in enumerate(
        result
    ):
        record["candidate_index"] = index

    print("=" * 80)
    print(
        f"[POOL] 去重前候选数量："
        f"{len(raw_records)}"
    )
    print(
        f"[POOL] 去重后候选数量："
        f"{len(result)}"
    )
    print(
        f"[POOL] 重复数量："
        f"{len(raw_records) - len(result)}"
    )
    print(
        f"[POOL] 读取失败："
        f"{failed_count}"
    )
    print("=" * 80)

    return result


def candidate_to_manifest_row(
    record: dict,
) -> dict:
    return {
        "candidate_index": (
            record["candidate_index"]
        ),
        "sha1": record["sha1"],
        "representative_path": (
            record["path"]
        ),
        "representative_filename": (
            record["filename"]
        ),
        "source_types": ";".join(
            sorted(
                record["source_types"]
            )
        ),
        "source_parent_cow_ids": ";".join(
            sorted(
                record[
                    "source_parent_cow_ids"
                ]
            )
        ),
        "identity_cow_ids": ";".join(
            sorted(
                record[
                    "identity_cow_ids"
                ]
            )
        ),
        "source_paths": ";".join(
            sorted(
                record["source_paths"]
            )
        ),
        "source_filenames": ";".join(
            sorted(
                record[
                    "source_filenames"
                ]
            )
        ),
        "duplicate_count": (
            record["duplicate_count"]
        ),
    }


# =========================================================
# 12. 图像预处理
# =========================================================

def build_eval_transform():
    return transforms.Compose([
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE)
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


class ImagePathDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        transform,
    ):
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

                image = self.transform(
                    image
                )

        except Exception as exc:
            raise RuntimeError(
                f"图片读取失败：{path}"
            ) from exc

        return image, path


# =========================================================
# 13. checkpoint加载
# =========================================================

def normalize_state_dict_keys(
    state_dict: Dict[
        str,
        torch.Tensor
    ],
) -> Dict[
    str,
    torch.Tensor
]:
    result = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith(
            "module."
        ):
            new_key = new_key[
                len("module.") :
            ]

        if new_key.startswith(
            "_orig_mod."
        ):
            new_key = new_key[
                len("_orig_mod.") :
            ]

        result[new_key] = value

    return result


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[dict, dict]:
    path = Path(
        checkpoint_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint不存在：{path}"
        )

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):
        state_dict = checkpoint[
            "model"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "model_state_dict"
        in checkpoint
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

    if not isinstance(
        state_dict,
        dict,
    ):
        raise TypeError(
            f"没有找到有效state_dict：{path}"
        )

    state_dict = normalize_state_dict_keys(
        state_dict
    )

    return checkpoint, state_dict


def build_model(
    checkpoint_path: str,
    backbone: str,
):
    checkpoint, state_dict = (
        load_checkpoint(
            checkpoint_path
        )
    )

    if "arcface.weight" not in state_dict:
        raise KeyError(
            "checkpoint中没有arcface.weight"
        )

    arcface_weight = state_dict[
        "arcface.weight"
    ]

    num_classes = int(
        arcface_weight.shape[0]
    )

    embedding_dim = int(
        arcface_weight.shape[1]
    )

    epoch = (
        checkpoint.get(
            "epoch",
            "unknown",
        )
        if isinstance(checkpoint, dict)
        else "unknown"
    )

    print(
        f"[MODEL] backbone={backbone}"
    )
    print(
        f"[MODEL] epoch={epoch}"
    )
    print(
        f"[MODEL] classes={num_classes}"
    )
    print(
        f"[MODEL] embedding_dim="
        f"{embedding_dim}"
    )

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
        f"[LOAD] missing="
        f"{len(missing_keys)}"
    )
    print(
        f"[LOAD] unexpected="
        f"{len(unexpected_keys)}"
    )

    if missing_keys:
        print(
            f"[LOAD] missing keys："
            f"{missing_keys[:10]}"
        )

    if unexpected_keys:
        print(
            f"[LOAD] unexpected keys："
            f"{unexpected_keys[:10]}"
        )

    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "checkpoint与当前模型结构不一致。\n"
            f"backbone={backbone}\n"
            f"missing={len(missing_keys)}\n"
            f"unexpected={len(unexpected_keys)}"
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    return (
        model,
        epoch,
        embedding_dim,
        num_classes,
    )


# =========================================================
# 14. embedding提取
# =========================================================

@torch.inference_mode()
def extract_embeddings(
    model,
    image_paths: List[str],
    transform,
    description: str,
) -> np.ndarray:
    if not image_paths:
        raise ValueError(
            f"{description}图片路径为空"
        )

    dataset = ImagePathDataset(
        image_paths=image_paths,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    all_embeddings = []

    for images, _ in tqdm(
        loader,
        desc=description,
        leave=False,
    ):
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        embeddings = model.forward_emb(
            images
        )

        # 严格沿用pair_dis中的L2归一化。
        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        all_embeddings.append(
            embeddings
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    if not all_embeddings:
        raise RuntimeError(
            f"{description}没有成功提取embedding"
        )

    result = np.concatenate(
        all_embeddings,
        axis=0,
    )

    del loader
    del dataset

    clear_memory()

    return result


# =========================================================
# 15. 所有模型特征提取
# =========================================================

def extract_features_for_all_models(
    target_normals_by_cow: Dict[
        str,
        List[dict]
    ],
    candidate_records: List[dict],
    transform,
):
    """
    对每个模型：

    1. 提取所有目标牛全部normal embedding；
    2. 为每头目标牛建立prototype；
    3. 提取全部候选池embedding。

    每张图片每个模型只推理一次。
    """

    all_target_normal_paths: List[str] = []

    normal_slices: Dict[
        str,
        Tuple[int, int]
    ] = {}

    cursor = 0

    for cow_id in sorted(
        target_normals_by_cow.keys()
    ):
        records = target_normals_by_cow[
            cow_id
        ]

        start = cursor

        for record in records:
            all_target_normal_paths.append(
                record["path"]
            )
            cursor += 1

        normal_slices[cow_id] = (
            start,
            cursor,
        )

    candidate_paths = [
        record["path"]
        for record in candidate_records
    ]

    model_features = {}
    model_metadata = {}

    for model_name, config in MODELS.items():
        print("\n" + "=" * 100)
        print(
            f"[START MODEL] {model_name}"
        )
        print(
            f"[BACKBONE]    "
            f"{config['backbone']}"
        )
        print(
            f"[TARGET NORMAL IMAGES] "
            f"{len(all_target_normal_paths)}"
        )
        print(
            f"[CANDIDATE IMAGES]     "
            f"{len(candidate_paths)}"
        )
        print("=" * 100)

        model, epoch, embedding_dim, num_classes = (
            build_model(
                checkpoint_path=(
                    config["path"]
                ),
                backbone=(
                    config["backbone"]
                ),
            )
        )

        normal_embeddings = (
            extract_embeddings(
                model=model,
                image_paths=(
                    all_target_normal_paths
                ),
                transform=transform,
                description=(
                    f"{model_name} 提取目标normal"
                ),
            )
        )

        candidate_embeddings = (
            extract_embeddings(
                model=model,
                image_paths=candidate_paths,
                transform=transform,
                description=(
                    f"{model_name} 提取候选池"
                ),
            )
        )

        prototypes = {}

        for cow_id, (
            start,
            end,
        ) in normal_slices.items():
            cow_embeddings = (
                normal_embeddings[start:end]
            )

            # 使用该目标牛全部normal embedding求平均。
            prototype = np.mean(
                cow_embeddings,
                axis=0,
            )

            # 求平均后再次进行L2归一化。
            prototype = (
                prototype
                / (
                    np.linalg.norm(
                        prototype
                    )
                    + 1e-12
                )
            ).astype(np.float32)

            prototypes[cow_id] = (
                prototype
            )

        model_features[model_name] = {
            "prototypes": prototypes,
            "candidate_embeddings": (
                candidate_embeddings
            ),
        }

        model_metadata[model_name] = {
            "backbone": config["backbone"],
            "checkpoint": config["path"],
            "epoch": epoch,
            "embedding_dim": embedding_dim,
            "num_classes": num_classes,
        }

        del model
        del normal_embeddings

        clear_memory()

    return (
        model_features,
        model_metadata,
    )


# =========================================================
# 16. 判断候选是否与目标牛同编号
# =========================================================

def candidate_matches_target(
    candidate: dict,
    target_cow_id: str,
) -> bool:
    """
    满足以下任一条件就删除：

    1. 候选已有明确身份编号，并包含目标牛编号；
    2. 候选任意原始文件名中包含目标牛编号。
    """

    target = str(
        target_cow_id
    )

    if target in candidate[
        "identity_cow_ids"
    ]:
        return True

    for filename in candidate[
        "source_filenames"
    ]:
        if filename_matches_cow_id(
            filename=filename,
            cow_id=target,
        ):
            return True

    return False


# =========================================================
# 17. 多模型共识排序
# =========================================================

def calculate_consensus_ranking(
    target_cow_id: str,
    requested_top_k: int,
    candidate_records: List[dict],
    model_features: dict,
):
    eligible_global_indices: List[int] = []

    filtered_same_id_count = 0

    for global_index, candidate in enumerate(
        candidate_records
    ):
        if candidate_matches_target(
            candidate=candidate,
            target_cow_id=target_cow_id,
        ):
            filtered_same_id_count += 1
            continue

        eligible_global_indices.append(
            global_index
        )

    if not eligible_global_indices:
        return None

    eligible_indices_np = np.asarray(
        eligible_global_indices,
        dtype=np.int64,
    )

    candidate_count = len(
        eligible_global_indices
    )

    model_top_n = min(
        candidate_count,
        max(
            MODEL_TOP_N_MIN,
            requested_top_k
            * MODEL_TOP_N_FACTOR,
        ),
    )

    vote_count = np.zeros(
        candidate_count,
        dtype=np.int32,
    )

    rank_score_sum = np.zeros(
        candidate_count,
        dtype=np.float64,
    )

    rank_sum = np.zeros(
        candidate_count,
        dtype=np.float64,
    )

    distance_sum = np.zeros(
        candidate_count,
        dtype=np.float64,
    )

    per_model_results = {}

    for model_name, features in (
        model_features.items()
    ):
        prototype = features[
            "prototypes"
        ][target_cow_id]

        candidate_embeddings = (
            features[
                "candidate_embeddings"
            ][eligible_indices_np]
        )

        # embedding和prototype均已经L2归一化。
        similarities = (
            candidate_embeddings
            @ prototype
        )

        # 距离越小，候选越像目标牛。
        distances = (
            1.0 - similarities
        ).astype(np.float32)

        order = np.argsort(
            distances,
            kind="stable",
        )

        ranks = np.empty(
            candidate_count,
            dtype=np.int32,
        )

        ranks[order] = np.arange(
            1,
            candidate_count + 1,
            dtype=np.int32,
        )

        if candidate_count == 1:
            rank_scores = np.ones(
                1,
                dtype=np.float32,
            )
        else:
            rank_scores = (
                1.0
                - (
                    ranks.astype(
                        np.float32
                    )
                    - 1.0
                )
                / (
                    candidate_count
                    - 1.0
                )
            )

        votes = (
            ranks <= model_top_n
        ).astype(np.int32)

        vote_count += votes
        rank_score_sum += rank_scores
        rank_sum += ranks
        distance_sum += distances

        per_model_results[
            model_name
        ] = {
            "distance": distances,
            "rank": ranks,
            "rank_score": rank_scores,
            "vote": votes,
        }

    model_count = len(
        model_features
    )

    consensus_score = (
        rank_score_sum
        / model_count
    )

    mean_rank = (
        rank_sum
        / model_count
    )

    mean_distance = (
        distance_sum
        / model_count
    )

    # 排序优先级：
    #
    # 1. 模型投票数越多越靠前；
    # 2. 共识排名得分越高越靠前；
    # 3. 平均排名越小越靠前；
    # 4. 平均距离越小越靠前。
    final_order = sorted(
        range(candidate_count),
        key=lambda local_index: (
            -int(
                vote_count[local_index]
            ),
            -float(
                consensus_score[
                    local_index
                ]
            ),
            float(
                mean_rank[local_index]
            ),
            float(
                mean_distance[
                    local_index
                ]
            ),
            candidate_records[
                eligible_global_indices[
                    local_index
                ]
            ]["sha1"],
        ),
    )

    return {
        "eligible_global_indices": (
            eligible_global_indices
        ),
        "filtered_same_id_count": (
            filtered_same_id_count
        ),
        "candidate_count": (
            candidate_count
        ),
        "model_top_n": model_top_n,
        "vote_count": vote_count,
        "consensus_score": (
            consensus_score
        ),
        "mean_rank": mean_rank,
        "mean_distance": mean_distance,
        "per_model": (
            per_model_results
        ),
        "final_order": final_order,
    }


# =========================================================
# 18. 输出文件命名
# =========================================================

def build_normal_output_name(
    target_cow_id: str,
    index: int,
    suffix: str,
) -> str:
    return (
        f"{target_cow_id}"
        f"__normal"
        f"__{index:05d}"
        f"{suffix.lower()}"
    )


def build_abnormal_output_name(
    target_cow_id: str,
    rank: int,
    candidate: dict,
) -> str:
    source_type_text = safe_column_name(
        "_".join(
            sorted(
                candidate["source_types"]
            )
        )
    )

    return (
        f"{target_cow_id}"
        f"__abnormal"
        f"__rank_{rank:04d}"
        f"__{source_type_text}"
        f"__{candidate['sha1'][:12]}"
        f"{candidate['suffix'].lower()}"
    )


# =========================================================
# 19. 构建单头牛数据
# =========================================================

def build_one_target_cow(
    target_cow_id: str,
    normal_records: List[dict],
    candidate_records: List[dict],
    model_features: dict,
):
    normal_count = len(
        normal_records
    )

    requested_top_k = (
        calculate_abnormal_count(
            normal_count
        )
    )

    ranking_result = (
        calculate_consensus_ranking(
            target_cow_id=(
                target_cow_id
            ),
            requested_top_k=(
                requested_top_k
            ),
            candidate_records=(
                candidate_records
            ),
            model_features=(
                model_features
            ),
        )
    )

    if ranking_result is None:
        return {
            "status": (
                "no_eligible_candidate"
            ),
            "cow_id": target_cow_id,
            "normal_count": (
                normal_count
            ),
        }

    selected_count = min(
        requested_top_k,
        ranking_result[
            "candidate_count"
        ],
    )

    selected_local_indices = (
        ranking_result[
            "final_order"
        ][:selected_count]
    )

    selected_local_set = set(
        selected_local_indices
    )

    cow_output_dir = (
        OUTPUT_ROOT
        / target_cow_id
    )

    normal_output_dir = (
        cow_output_dir
        / "normal"
    )

    abnormal_output_dir = (
        cow_output_dir
        / "abnormal"
    )

    normal_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    abnormal_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_rows: List[dict] = []
    selected_rows: List[dict] = []

    # -----------------------------------------------------
    # 复制目标牛全部normal
    # -----------------------------------------------------

    for index, record in enumerate(
        normal_records,
        start=1,
    ):
        source_path = Path(
            record["path"]
        )

        new_filename = (
            build_normal_output_name(
                target_cow_id=(
                    target_cow_id
                ),
                index=index,
                suffix=record["suffix"],
            )
        )

        destination_path = (
            normal_output_dir
            / new_filename
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        row = {
            "target_cow_id": (
                target_cow_id
            ),
            "label": "normal",
            "source_types": (
                "openset_target_normal"
            ),
            "source_parent_cow_ids": (
                target_cow_id
            ),
            "identity_cow_ids": (
                target_cow_id
            ),
            "original_filename": (
                record["filename"]
            ),
            "original_path": (
                str(source_path)
            ),
            "new_filename": (
                new_filename
            ),
            "new_path": (
                str(destination_path)
            ),
            "selected_rank": "",
            "vote_count": "",
            "consensus_score": "",
            "mean_rank": "",
            "mean_distance": "",
            "sha1": "",
            "duplicate_count": 1,
        }

        for model_name in MODELS:
            column_name = (
                safe_column_name(
                    model_name
                )
            )

            row[
                f"{column_name}_distance"
            ] = ""

            row[
                f"{column_name}_rank"
            ] = ""

        label_rows.append(
            row
        )

    # -----------------------------------------------------
    # 复制多模型共识困难abnormal
    # -----------------------------------------------------

    for selected_rank, local_index in enumerate(
        selected_local_indices,
        start=1,
    ):
        global_index = (
            ranking_result[
                "eligible_global_indices"
            ][local_index]
        )

        candidate = candidate_records[
            global_index
        ]

        source_path = Path(
            candidate["path"]
        )

        new_filename = (
            build_abnormal_output_name(
                target_cow_id=(
                    target_cow_id
                ),
                rank=selected_rank,
                candidate=candidate,
            )
        )

        destination_path = (
            abnormal_output_dir
            / new_filename
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        row = {
            "target_cow_id": (
                target_cow_id
            ),
            "label": "abnormal",

            "source_types": ";".join(
                sorted(
                    candidate[
                        "source_types"
                    ]
                )
            ),

            "source_parent_cow_ids": (
                ";".join(
                    sorted(
                        candidate[
                            "source_parent_cow_ids"
                        ]
                    )
                )
            ),

            "identity_cow_ids": (
                ";".join(
                    sorted(
                        candidate[
                            "identity_cow_ids"
                        ]
                    )
                )
            ),

            "original_filename": (
                candidate["filename"]
            ),

            "original_path": (
                candidate["path"]
            ),

            "new_filename": (
                new_filename
            ),

            "new_path": (
                str(destination_path)
            ),

            "selected_rank": (
                selected_rank
            ),

            "vote_count": int(
                ranking_result[
                    "vote_count"
                ][local_index]
            ),

            "consensus_score": float(
                ranking_result[
                    "consensus_score"
                ][local_index]
            ),

            "mean_rank": float(
                ranking_result[
                    "mean_rank"
                ][local_index]
            ),

            "mean_distance": float(
                ranking_result[
                    "mean_distance"
                ][local_index]
            ),

            "sha1": candidate["sha1"],

            "duplicate_count": (
                candidate[
                    "duplicate_count"
                ]
            ),
        }

        for model_name in MODELS:
            column_name = (
                safe_column_name(
                    model_name
                )
            )

            model_result = (
                ranking_result[
                    "per_model"
                ][model_name]
            )

            row[
                f"{column_name}_distance"
            ] = float(
                model_result[
                    "distance"
                ][local_index]
            )

            row[
                f"{column_name}_rank"
            ] = int(
                model_result[
                    "rank"
                ][local_index]
            )

        label_rows.append(
            row
        )

        selected_rows.append(
            row
        )

    # -----------------------------------------------------
    # 保存该牛候选前200审计表
    # -----------------------------------------------------

    audit_rows: List[dict] = []

    audit_limit = min(
        SCORE_AUDIT_TOP_N,
        ranking_result[
            "candidate_count"
        ],
    )

    for final_position, local_index in enumerate(
        ranking_result[
            "final_order"
        ][:audit_limit],
        start=1,
    ):
        global_index = (
            ranking_result[
                "eligible_global_indices"
            ][local_index]
        )

        candidate = candidate_records[
            global_index
        ]

        audit_row = {
            "target_cow_id": (
                target_cow_id
            ),
            "final_position": (
                final_position
            ),
            "selected": int(
                local_index
                in selected_local_set
            ),
            "source_types": ";".join(
                sorted(
                    candidate[
                        "source_types"
                    ]
                )
            ),
            "source_parent_cow_ids": (
                ";".join(
                    sorted(
                        candidate[
                            "source_parent_cow_ids"
                        ]
                    )
                )
            ),
            "identity_cow_ids": ";".join(
                sorted(
                    candidate[
                        "identity_cow_ids"
                    ]
                )
            ),
            "filename": (
                candidate["filename"]
            ),
            "path": candidate["path"],
            "sha1": candidate["sha1"],
            "duplicate_count": (
                candidate[
                    "duplicate_count"
                ]
            ),
            "vote_count": int(
                ranking_result[
                    "vote_count"
                ][local_index]
            ),
            "consensus_score": float(
                ranking_result[
                    "consensus_score"
                ][local_index]
            ),
            "mean_rank": float(
                ranking_result[
                    "mean_rank"
                ][local_index]
            ),
            "mean_distance": float(
                ranking_result[
                    "mean_distance"
                ][local_index]
            ),
        }

        for model_name in MODELS:
            column_name = (
                safe_column_name(
                    model_name
                )
            )

            model_result = (
                ranking_result[
                    "per_model"
                ][model_name]
            )

            audit_row[
                f"{column_name}_distance"
            ] = float(
                model_result[
                    "distance"
                ][local_index]
            )

            audit_row[
                f"{column_name}_rank"
            ] = int(
                model_result[
                    "rank"
                ][local_index]
            )

            audit_row[
                f"{column_name}_rank_score"
            ] = float(
                model_result[
                    "rank_score"
                ][local_index]
            )

            audit_row[
                f"{column_name}_vote"
            ] = int(
                model_result[
                    "vote"
                ][local_index]
            )

        audit_rows.append(
            audit_row
        )

    audit_path = (
        OUTPUT_ROOT
        / "score_audit"
        / (
            f"{target_cow_id}"
            f"_candidate_top"
            f"{audit_limit}.csv"
        )
    )

    save_csv(
        audit_rows,
        audit_path,
    )

    actual_ratio = (
        normal_count
        / max(selected_count, 1)
    )

    summary_row = {
        "cow_id": target_cow_id,
        "status": "success",

        "normal_count": normal_count,

        "requested_abnormal_count": (
            requested_top_k
        ),

        "selected_abnormal_count": (
            selected_count
        ),

        "actual_normal_abnormal_ratio": (
            float(actual_ratio)
        ),

        "total_candidate_pool": len(
            candidate_records
        ),

        "filtered_same_id_count": (
            ranking_result[
                "filtered_same_id_count"
            ]
        ),

        "eligible_candidate_count": (
            ranking_result[
                "candidate_count"
            ]
        ),

        "model_top_n": (
            ranking_result[
                "model_top_n"
            ]
        ),
    }

    return {
        "status": "success",
        "cow_id": target_cow_id,
        "labels": label_rows,
        "selected": selected_rows,
        "summary": summary_row,
    }


# =========================================================
# 20. 保存过滤后的split.json
# =========================================================

def save_filtered_split(
    built_cow_ids: List[str],
) -> None:
    source_split_path = (
        OPENSET_ROOT
        / "split.json"
    )

    if not source_split_path.exists():
        print(
            "[WARN] 开放集不存在split.json，"
            "不生成新的split.json"
        )
        return

    with source_split_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        split_data = json.load(
            file
        )

    built_set = set(
        str(cow_id)
        for cow_id in built_cow_ids
    )

    new_split = {}

    for key, value in split_data.items():
        if (
            isinstance(value, list)
            and key.endswith("_cows")
        ):
            new_split[key] = [
                str(cow_id)
                for cow_id in value
                if str(cow_id) in built_set
            ]
        else:
            new_split[key] = value

    output_split_path = (
        OUTPUT_ROOT
        / "split.json"
    )

    with output_split_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            new_split,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[SAVE] {output_split_path}"
    )


# =========================================================
# 21. main
# =========================================================

def main() -> None:
    set_seed(SEED)
    clear_memory()

    if not TRAIN_ROOT.exists():
        raise FileNotFoundError(
            f"训练集不存在：{TRAIN_ROOT}"
        )

    if not OPENSET_ROOT.exists():
        raise FileNotFoundError(
            f"开放集不存在：{OPENSET_ROOT}"
        )

    if (
        OUTPUT_ROOT.resolve()
        == TRAIN_ROOT.resolve()
        or OUTPUT_ROOT.resolve()
        == OPENSET_ROOT.resolve()
    ):
        raise RuntimeError(
            "OUTPUT_ROOT不能和输入数据集相同，"
            "否则会删除原始数据。"
        )

    # 检查所有checkpoint。
    for model_name, config in MODELS.items():
        checkpoint_path = Path(
            config["path"]
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"{model_name} checkpoint不存在："
                f"{checkpoint_path}"
            )

    if CLEAR_OUTPUT_ROOT and OUTPUT_ROOT.exists():
        print(
            f"[CLEAR] 删除旧输出："
            f"{OUTPUT_ROOT}"
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(f"[DEVICE]       {DEVICE}")
    print(f"[TRAIN ROOT]   {TRAIN_ROOT}")
    print(f"[OPENSET ROOT] {OPENSET_ROOT}")
    print(f"[OUTPUT ROOT]  {OUTPUT_ROOT}")
    print(
        f"[RATIO]        normal:abnormal "
        f"≈ {NORMAL_ABNORMAL_RATIO}:1"
    )
    print(
        f"[TRAIN NORMAL] 每编号抽"
        f"{TRAIN_NORMAL_SAMPLE_PER_COW}张"
    )
    print(
        f"[OPEN NORMAL]  每编号抽"
        f"{OPENSET_NORMAL_SAMPLE_PER_COW}张"
    )
    print(
        "[OPEN ABNORMAL] 不使用"
    )
    print("=" * 100)

    # -----------------------------------------------------
    # 发现训练集和开放集牛编号
    # -----------------------------------------------------

    train_cow_dirs = discover_cow_dirs(
        TRAIN_ROOT
    )

    openset_cow_dirs = discover_cow_dirs(
        OPENSET_ROOT
    )

    if not train_cow_dirs:
        raise RuntimeError(
            f"训练集中没有发现牛编号目录："
            f"{TRAIN_ROOT}"
        )

    if not openset_cow_dirs:
        raise RuntimeError(
            f"开放集中没有发现牛编号目录："
            f"{OPENSET_ROOT}"
        )

    print(
        f"[DATA] 训练集编号数量："
        f"{len(train_cow_dirs)}"
    )

    print(
        f"[DATA] 开放集编号数量："
        f"{len(openset_cow_dirs)}"
    )

    known_cow_ids = (
        set(train_cow_dirs.keys())
        | set(openset_cow_dirs.keys())
    )

    # -----------------------------------------------------
    # 目标牛：开放集每头牛全部normal
    # -----------------------------------------------------

    target_normals_by_cow = (
        collect_target_normals(
            openset_cow_dirs
        )
    )

    if not target_normals_by_cow:
        raise RuntimeError(
            "没有有效目标牛。"
        )

    total_target_normal = sum(
        len(records)
        for records
        in target_normals_by_cow.values()
    )

    print(
        f"[TARGET] 有效目标牛："
        f"{len(target_normals_by_cow)}"
    )

    print(
        f"[TARGET] 全部normal数量："
        f"{total_target_normal}"
    )

    # -----------------------------------------------------
    # 候选池第一部分：
    # train_dataset 每编号normal抽3张
    # -----------------------------------------------------

    train_normal_records, (
        train_normal_manifest
    ) = sample_normal_candidates(
        cow_dirs=train_cow_dirs,
        source_type="train_normal",
        sample_per_cow=(
            TRAIN_NORMAL_SAMPLE_PER_COW
        ),
    )

    # -----------------------------------------------------
    # 候选池第二部分：
    # train_dataset 全部abnormal
    # -----------------------------------------------------

    train_abnormal_records, (
        train_abnormal_manifest
    ) = collect_train_abnormal_candidates(
        train_cow_dirs=train_cow_dirs,
        known_cow_ids=known_cow_ids,
    )

    # -----------------------------------------------------
    # 候选池第三部分：
    # openset每编号normal抽3张
    # -----------------------------------------------------

    openset_normal_records, (
        openset_normal_manifest
    ) = sample_normal_candidates(
        cow_dirs=openset_cow_dirs,
        source_type="openset_normal",
        sample_per_cow=(
            OPENSET_NORMAL_SAMPLE_PER_COW
        ),
    )

    print("=" * 80)
    print(
        f"[POOL SOURCE] train normal："
        f"{len(train_normal_records)}"
    )
    print(
        f"[POOL SOURCE] train abnormal："
        f"{len(train_abnormal_records)}"
    )
    print(
        f"[POOL SOURCE] openset normal："
        f"{len(openset_normal_records)}"
    )
    print(
        "[POOL SOURCE] openset abnormal：0"
    )
    print("=" * 80)

    raw_candidate_records = [
        *train_normal_records,
        *train_abnormal_records,
        *openset_normal_records,
    ]

    if not raw_candidate_records:
        raise RuntimeError(
            "候选池为空。"
        )

    # -----------------------------------------------------
    # 保存去重前来源清单
    # -----------------------------------------------------

    source_manifest_rows = [
        *train_normal_manifest,
        *train_abnormal_manifest,
        *openset_normal_manifest,
    ]

    save_csv(
        source_manifest_rows,
        OUTPUT_ROOT
        / "candidate_sources_before_dedup.csv",
    )

    # -----------------------------------------------------
    # SHA1去重
    # -----------------------------------------------------

    candidate_records = (
        deduplicate_candidates(
            raw_candidate_records
        )
    )

    if not candidate_records:
        raise RuntimeError(
            "候选池去重后为空。"
        )

    candidate_manifest_rows = [
        candidate_to_manifest_row(
            record
        )
        for record in candidate_records
    ]

    save_csv(
        candidate_manifest_rows,
        OUTPUT_ROOT
        / "candidate_pool_manifest.csv",
    )

    # -----------------------------------------------------
    # 五个模型提取目标normal和候选池特征
    # -----------------------------------------------------

    transform = build_eval_transform()

    model_features, model_metadata = (
        extract_features_for_all_models(
            target_normals_by_cow=(
                target_normals_by_cow
            ),
            candidate_records=(
                candidate_records
            ),
            transform=transform,
        )
    )

    # -----------------------------------------------------
    # 逐头目标牛构建困难开放集
    # -----------------------------------------------------

    all_label_rows: List[dict] = []
    all_selected_rows: List[dict] = []
    all_summary_rows: List[dict] = []

    built_cow_ids: List[str] = []

    target_cow_ids = sorted(
        target_normals_by_cow.keys()
    )

    for cow_index, cow_id in enumerate(
        target_cow_ids,
        start=1,
    ):
        print("\n" + "=" * 100)
        print(
            f"[BUILD] {cow_index}/"
            f"{len(target_cow_ids)} "
            f"cow_id={cow_id}"
        )

        result = build_one_target_cow(
            target_cow_id=cow_id,
            normal_records=(
                target_normals_by_cow[
                    cow_id
                ]
            ),
            candidate_records=(
                candidate_records
            ),
            model_features=(
                model_features
            ),
        )

        if result["status"] != "success":
            print(
                f"[WARN] {cow_id}构建失败："
                f"{result['status']}"
            )

            all_summary_rows.append({
                "cow_id": cow_id,
                "status": result["status"],
                "normal_count": (
                    result["normal_count"]
                ),
                "requested_abnormal_count": "",
                "selected_abnormal_count": 0,
                "actual_normal_abnormal_ratio": "",
                "total_candidate_pool": (
                    len(candidate_records)
                ),
                "filtered_same_id_count": "",
                "eligible_candidate_count": 0,
                "model_top_n": "",
            })

            continue

        built_cow_ids.append(
            cow_id
        )

        all_label_rows.extend(
            result["labels"]
        )

        all_selected_rows.extend(
            result["selected"]
        )

        all_summary_rows.append(
            result["summary"]
        )

        summary = result["summary"]

        print(
            f"[DONE] {cow_id}: "
            f"normal={summary['normal_count']}, "
            f"abnormal="
            f"{summary['selected_abnormal_count']}, "
            f"ratio="
            f"{summary['actual_normal_abnormal_ratio']:.3f}:1, "
            f"same_id_filtered="
            f"{summary['filtered_same_id_count']}"
        )

        # 每完成一头牛就保存summary，避免中断丢失。
        save_csv(
            all_summary_rows,
            OUTPUT_ROOT
            / "dataset_summary.csv",
        )

    if not built_cow_ids:
        raise RuntimeError(
            "没有任何目标牛构建成功。"
        )

    # -----------------------------------------------------
    # 最终CSV字段
    # -----------------------------------------------------

    model_columns: List[str] = []

    for model_name in MODELS:
        column_name = safe_column_name(
            model_name
        )

        model_columns.extend([
            f"{column_name}_distance",
            f"{column_name}_rank",
        ])

    label_fieldnames = [
        "target_cow_id",
        "label",
        "source_types",
        "source_parent_cow_ids",
        "identity_cow_ids",
        "original_filename",
        "original_path",
        "new_filename",
        "new_path",
        "selected_rank",
        "vote_count",
        "consensus_score",
        "mean_rank",
        "mean_distance",
        "sha1",
        "duplicate_count",
        *model_columns,
    ]

    save_csv(
        all_label_rows,
        OUTPUT_ROOT / "labels.csv",
        fieldnames=label_fieldnames,
    )

    save_csv(
        all_selected_rows,
        OUTPUT_ROOT
        / "selected_abnormal_manifest.csv",
        fieldnames=label_fieldnames,
    )

    save_csv(
        all_summary_rows,
        OUTPUT_ROOT
        / "dataset_summary.csv",
    )

    save_filtered_split(
        built_cow_ids
    )

    # -----------------------------------------------------
    # 保存配置
    # -----------------------------------------------------

    source_type_counts = {
        "train_normal_before_dedup": (
            len(train_normal_records)
        ),
        "train_abnormal_before_dedup": (
            len(train_abnormal_records)
        ),
        "openset_normal_before_dedup": (
            len(openset_normal_records)
        ),
        "openset_abnormal": 0,
        "total_before_dedup": (
            len(raw_candidate_records)
        ),
        "total_after_dedup": (
            len(candidate_records)
        ),
    }

    config_data = {
        "train_root": str(
            TRAIN_ROOT
        ),
        "openset_root": str(
            OPENSET_ROOT
        ),
        "output_root": str(
            OUTPUT_ROOT
        ),

        "device": str(DEVICE),
        "seed": SEED,

        "image_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,

        "train_normal_sample_per_cow": (
            TRAIN_NORMAL_SAMPLE_PER_COW
        ),

        "openset_normal_sample_per_cow": (
            OPENSET_NORMAL_SAMPLE_PER_COW
        ),

        "use_all_train_abnormal": True,

        "use_openset_abnormal": False,

        "use_all_target_normal_for_center": (
            True
        ),

        "normal_abnormal_ratio": (
            NORMAL_ABNORMAL_RATIO
        ),

        "same_id_match_mode": (
            SAME_ID_MATCH_MODE
        ),

        "deduplicate_by_sha1": (
            DEDUPLICATE_BY_SHA1
        ),

        "distance": "cosine_distance",

        "consensus_sort": [
            "vote_count_desc",
            "consensus_score_desc",
            "mean_rank_asc",
            "mean_distance_asc",
        ],

        "model_top_n_min": (
            MODEL_TOP_N_MIN
        ),

        "model_top_n_factor": (
            MODEL_TOP_N_FACTOR
        ),

        "source_type_counts": (
            source_type_counts
        ),

        "num_target_cows": (
            len(target_normals_by_cow)
        ),

        "num_built_cows": (
            len(built_cow_ids)
        ),

        "num_target_normal_images": (
            total_target_normal
        ),

        "models": model_metadata,
    }

    config_path = (
        OUTPUT_ROOT
        / "config.json"
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[SAVE] {config_path}"
    )

    print("\n" + "=" * 100)
    print("[ALL FINISHED]")
    print(
        f"[OUTPUT ROOT] "
        f"{OUTPUT_ROOT}"
    )
    print(
        f"[BUILT COWS] "
        f"{len(built_cow_ids)}"
    )
    print(
        f"[TOTAL LABELS] "
        f"{len(all_label_rows)}"
    )
    print(
        f"[SELECTED ABNORMAL] "
        f"{len(all_selected_rows)}"
    )
    print(
        f"[CANDIDATE POOL] "
        f"{len(candidate_records)}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()