# -*- coding: utf-8 -*-

import json
import random
import shutil
import hashlib

from pathlib import Path
from typing import Dict, List, Tuple


# =========================================================
# 1. 路径配置
# =========================================================

# 原始数据集
SOURCE_ROOT = Path(
    r"D:\code_project\triplet_code\data\dataset_OBB"
)

# 原始划分文件
SOURCE_SPLIT_JSON = (
    SOURCE_ROOT / "split.json"
)

# 新数据集输出位置
TARGET_ROOT = Path(
    r"D:\datasets\cow_recognition8.3\train_dataset"
)

# 新划分文件
TARGET_SPLIT_JSON = (
    TARGET_ROOT / "split.json"
)


# =========================================================
# 2. 划分参数
# =========================================================

TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

RANDOM_SEED = 42

# 是否删除已经存在的新数据集目录
# True：先删除TARGET_ROOT，再重新复制，结果最干净
# False：保留已有内容，但会覆盖同名牛编号文件夹
CLEAR_TARGET_ROOT = True


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
# 3. 基础函数
# =========================================================

def is_image_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in IMG_EXTS
    )


def load_original_split(
    split_json_path: Path,
) -> Dict:
    """
    读取原始split.json。
    seen_cows和unseen_cows完全按照原文件使用，
    不重新随机划分牛编号。
    """

    if not split_json_path.exists():
        raise FileNotFoundError(
            f"找不到原始划分文件：{split_json_path}"
        )

    with open(
        split_json_path,
        "r",
        encoding="utf-8",
    ) as file:
        split_data = json.load(file)

    required_keys = {
        "seen_cows",
        "unseen_cows",
    }

    missing_keys = (
        required_keys
        -
        set(split_data.keys())
    )

    if missing_keys:
        raise KeyError(
            f"原始split.json缺少字段：{missing_keys}"
        )

    return split_data


def stable_cow_seed(
    cow_id: str,
    base_seed: int,
) -> int:
    """
    为每头牛生成稳定且可复现的随机种子。

    不使用Python内置hash()，
    因为hash()在不同运行中可能发生变化。
    """

    text = f"{base_seed}_{cow_id}"

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    offset = int(
        digest[:8],
        16,
    )

    return (
        base_seed
        +
        offset
    ) % (2 ** 32)


def list_normal_images(
    dataset_root: Path,
    cow_id: str,
) -> List[Path]:
    """
    读取：
    dataset_root/cow_id/normal
    下面的所有图片。
    """

    normal_dir = (
        dataset_root
        /
        cow_id
        /
        "normal"
    )

    if not normal_dir.exists():
        return []

    return sorted([
        path
        for path in normal_dir.iterdir()
        if is_image_file(path)
    ])


def split_train_val(
    image_paths: List[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    """
    对单头牛内部图片按照train:val进行划分。
    """

    if not image_paths:
        return [], []

    if abs(
        train_ratio
        +
        val_ratio
        -
        1.0
    ) > 1e-8:
        raise ValueError(
            "train_ratio + val_ratio 必须等于1。"
        )

    image_paths = sorted(
        image_paths
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        image_paths
    )

    image_count = len(
        image_paths
    )

    # 只有1张图时，只能放入训练集
    if image_count == 1:
        return image_paths, []

    # 按20%计算验证集数量
    val_count = int(
        round(
            image_count
            *
            val_ratio
        )
    )

    # 至少保证train和val各有1张
    val_count = max(
        1,
        val_count,
    )

    val_count = min(
        val_count,
        image_count - 1,
    )

    val_paths = image_paths[
        :val_count
    ]

    train_paths = image_paths[
        val_count:
    ]

    return (
        train_paths,
        val_paths,
    )


def wrap_samples(
    image_paths: List[Path],
    cow_id: str,
) -> List[Dict]:
    """
    转换成训练代码需要的格式：
    {
        "path": "...",
        "cow_id": "..."
    }
    """

    return [
        {
            "path": str(path.resolve()),
            "cow_id": str(cow_id),
        }
        for path in image_paths
    ]


# =========================================================
# 4. 复制seen牛只
# =========================================================

def prepare_target_root(
    target_root: Path,
    clear_target_root: bool,
):
    """
    准备输出目录。
    """

    if (
        target_root.exists()
        and clear_target_root
    ):
        print(
            f"[INFO] 删除已有目标目录：{target_root}"
        )

        shutil.rmtree(
            target_root
        )

    target_root.mkdir(
        parents=True,
        exist_ok=True,
    )


def copy_seen_cows(
    source_root: Path,
    target_root: Path,
    seen_cows: List[str],
):
    """
    将seen牛编号的整个文件夹复制到新数据集。

    会保留原始目录结构，例如：
    cow_id/
        normal/
        abnormal/
    """

    copied_count = 0
    missing_cows = []

    print("\n" + "=" * 80)
    print("开始复制seen牛只")
    print("=" * 80)

    for index, cow_id in enumerate(
        seen_cows,
        start=1,
    ):
        source_cow_dir = (
            source_root
            /
            str(cow_id)
        )

        target_cow_dir = (
            target_root
            /
            str(cow_id)
        )

        if not source_cow_dir.exists():
            print(
                f"[WARNING] 原始牛编号目录不存在："
                f"{source_cow_dir}"
            )

            missing_cows.append(
                str(cow_id)
            )

            continue

        # 避免目标中残留旧文件
        if target_cow_dir.exists():
            shutil.rmtree(
                target_cow_dir
            )

        shutil.copytree(
            source_cow_dir,
            target_cow_dir,
        )

        copied_count += 1

        print(
            f"[{index}/{len(seen_cows)}] "
            f"已复制牛编号：{cow_id}"
        )

    print("-" * 80)
    print(f"seen牛数量：{len(seen_cows)}")
    print(f"成功复制：{copied_count}")
    print(f"缺失牛只：{len(missing_cows)}")

    if missing_cows:
        print(
            "缺失牛编号："
            +
            ", ".join(missing_cows)
        )

    return missing_cows


# =========================================================
# 5. 重新划分seen内部train/val
# =========================================================

def build_new_split(
    target_root: Path,
    original_split: Dict,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict:
    """
    保留原始seen_cows和unseen_cows，
    仅重新生成seen_train和seen_val。
    """

    # 完全保留原始顺序
    seen_cows = [
        str(cow_id)
        for cow_id in original_split[
            "seen_cows"
        ]
    ]

    unseen_cows = [
        str(cow_id)
        for cow_id in original_split[
            "unseen_cows"
        ]
    ]

    seen_train_samples = []
    seen_val_samples = []

    per_cow_statistics = []

    print("\n" + "=" * 80)
    print("开始按照train:val=8:2重新划分seen图片")
    print("=" * 80)

    for index, cow_id in enumerate(
        seen_cows,
        start=1,
    ):
        image_paths = list_normal_images(
            dataset_root=target_root,
            cow_id=cow_id,
        )

        cow_seed = stable_cow_seed(
            cow_id=cow_id,
            base_seed=seed,
        )

        train_paths, val_paths = (
            split_train_val(
                image_paths=image_paths,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=cow_seed,
            )
        )

        seen_train_samples.extend(
            wrap_samples(
                train_paths,
                cow_id,
            )
        )

        seen_val_samples.extend(
            wrap_samples(
                val_paths,
                cow_id,
            )
        )

        per_cow_statistics.append({
            "cow_id": cow_id,
            "total": len(image_paths),
            "train": len(train_paths),
            "val": len(val_paths),
        })

        print(
            f"[{index}/{len(seen_cows)}] "
            f"牛编号={cow_id} | "
            f"总数={len(image_paths)} | "
            f"train={len(train_paths)} | "
            f"val={len(val_paths)}"
        )

    new_split = {
        # 牛编号划分完全不变
        "seen_cows": seen_cows,
        "unseen_cows": unseen_cows,

        # 新的8:2图片划分
        "seen_train": seen_train_samples,
        "seen_val": seen_val_samples,

        # 保留字段，防止旧代码访问seen_test时报错
        # 本次不再使用seen_test
        "seen_test": [],

        "meta": {
            "seed": seed,
            "cow_split_source": str(
                SOURCE_SPLIT_JSON.resolve()
            ),
            "source_root": str(
                SOURCE_ROOT.resolve()
            ),
            "target_root": str(
                target_root.resolve()
            ),
            "seen_cow_count": len(
                seen_cows
            ),
            "unseen_cow_count": len(
                unseen_cows
            ),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": 0.0,
            "seen_train_count": len(
                seen_train_samples
            ),
            "seen_val_count": len(
                seen_val_samples
            ),
            "per_cow_statistics": (
                per_cow_statistics
            ),
        },
    }

    return new_split


def save_split(
    split_data: Dict,
    save_path: Path,
):
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        save_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            split_data,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print(
        f"\n[INFO] 新split.json已保存：{save_path}"
    )


# =========================================================
# 6. 校验结果
# =========================================================

def validate_split(
    split_data: Dict,
    target_root: Path,
):
    """
    检查：
    1. train和val是否有路径重复；
    2. 文件是否真实存在；
    3. target中是否只包含seen牛；
    4. seen和unseen是否存在交集。
    """

    train_samples = split_data[
        "seen_train"
    ]

    val_samples = split_data[
        "seen_val"
    ]

    train_paths = {
        sample["path"]
        for sample in train_samples
    }

    val_paths = {
        sample["path"]
        for sample in val_samples
    }

    overlap = (
        train_paths
        &
        val_paths
    )

    missing_paths = []

    for path_string in (
        train_paths
        |
        val_paths
    ):
        if not Path(path_string).exists():
            missing_paths.append(
                path_string
            )

    seen_cows = set(
        split_data["seen_cows"]
    )

    unseen_cows = set(
        split_data["unseen_cows"]
    )

    cow_overlap = (
        seen_cows
        &
        unseen_cows
    )

    target_cows = {
        path.name
        for path in target_root.iterdir()
        if path.is_dir()
    }

    unexpected_target_cows = (
        target_cows
        -
        seen_cows
    )

    print("\n" + "=" * 80)
    print("数据集校验")
    print("=" * 80)

    print(
        f"seen牛数量：{len(seen_cows)}"
    )

    print(
        f"unseen牛数量：{len(unseen_cows)}"
    )

    print(
        f"seen_train图片：{len(train_samples)}"
    )

    print(
        f"seen_val图片：{len(val_samples)}"
    )

    print(
        f"train/val重复图片：{len(overlap)}"
    )

    print(
        f"不存在的图片路径：{len(missing_paths)}"
    )

    print(
        f"seen/unseen牛编号交集：{len(cow_overlap)}"
    )

    print(
        f"目标目录意外牛编号："
        f"{len(unexpected_target_cows)}"
    )

    if overlap:
        raise RuntimeError(
            "seen_train和seen_val存在重复图片。"
        )

    if missing_paths:
        raise FileNotFoundError(
            f"有{len(missing_paths)}个图片路径不存在。"
        )

    if cow_overlap:
        raise RuntimeError(
            "seen_cows和unseen_cows存在交集。"
        )

    if unexpected_target_cows:
        print(
            "[WARNING] 目标目录存在非seen牛编号："
            +
            ", ".join(
                sorted(
                    unexpected_target_cows
                )
            )
        )

    print(
        "\n[INFO] 数据集和split.json校验通过。"
    )


# =========================================================
# 7. 读取接口
# =========================================================

def load_split(
    root_dir: str,
    save_name: str = "split.json",
) -> Dict:
    """
    注意：不再使用写死的split.json路径。

    root_dir指向哪个数据集，
    就读取哪个数据集下的split.json。
    """

    split_path = (
        Path(root_dir)
        /
        save_name
    )

    if not split_path.exists():
        raise FileNotFoundError(
            f"找不到划分文件：{split_path}"
        )

    with open(
        split_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_cow_ids(
    root_dir: str,
    split: str = "seen_cows",
    save_name: str = "split.json",
) -> List[str]:
    data = load_split(
        root_dir,
        save_name,
    )

    if split not in data:
        raise KeyError(
            f"split.json不存在字段：{split}"
        )

    return data[split]


def get_samples(
    root_dir: str,
    split: str = "seen_train",
    save_name: str = "split.json",
) -> List[Dict]:
    data = load_split(
        root_dir,
        save_name,
    )

    if split not in data:
        raise KeyError(
            f"split.json不存在字段：{split}"
        )

    return data[split]


def get_paths_and_labels(
    root_dir: str,
    split: str = "seen_train",
    save_name: str = "split.json",
) -> Tuple[List[str], List[str]]:
    samples = get_samples(
        root_dir,
        split,
        save_name,
    )

    paths = [
        sample["path"]
        for sample in samples
    ]

    labels = [
        sample["cow_id"]
        for sample in samples
    ]

    return paths, labels


# =========================================================
# 8. 主程序
# =========================================================

def main():
    print("=" * 80)
    print("构建新的seen牛训练数据集")
    print("=" * 80)

    print(
        f"原始数据集：{SOURCE_ROOT}"
    )

    print(
        f"原始split：{SOURCE_SPLIT_JSON}"
    )

    print(
        f"目标数据集：{TARGET_ROOT}"
    )

    print(
        f"新split：{TARGET_SPLIT_JSON}"
    )

    # 1. 读取原始split.json
    original_split = load_original_split(
        SOURCE_SPLIT_JSON
    )

    seen_cows = [
        str(cow_id)
        for cow_id in original_split[
            "seen_cows"
        ]
    ]

    unseen_cows = [
        str(cow_id)
        for cow_id in original_split[
            "unseen_cows"
        ]
    ]

    print(
        f"\n原始seen牛数量：{len(seen_cows)}"
    )

    print(
        f"原始unseen牛数量：{len(unseen_cows)}"
    )

    # 2. 准备新数据集目录
    prepare_target_root(
        target_root=TARGET_ROOT,
        clear_target_root=CLEAR_TARGET_ROOT,
    )

    # 3. 复制所有seen牛文件夹
    copy_seen_cows(
        source_root=SOURCE_ROOT,
        target_root=TARGET_ROOT,
        seen_cows=seen_cows,
    )

    # 4. 对seen正常图片重新进行8:2划分
    new_split = build_new_split(
        target_root=TARGET_ROOT,
        original_split=original_split,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED,
    )

    # 5. 保存新split.json
    save_split(
        split_data=new_split,
        save_path=TARGET_SPLIT_JSON,
    )

    # 6. 校验
    validate_split(
        split_data=new_split,
        target_root=TARGET_ROOT,
    )

    print("\n" + "=" * 80)
    print("处理完成")
    print("=" * 80)

    print(
        "seen/unseen牛编号划分：保持原split.json不变"
    )

    print(
        "seen内部图片划分：train:val=8:2"
    )

    print(
        f"新数据集：{TARGET_ROOT}"
    )

    print(
        f"新split.json：{TARGET_SPLIT_JSON}"
    )


if __name__ == "__main__":
    main()