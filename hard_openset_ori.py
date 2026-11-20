# -*- coding: utf-8 -*-

"""
将每个牛编号下的 normal 和 abnormal 混合到同一个目录。

输入结构：

SOURCE_ROOT/
├── 000071/
│   ├── normal/
│   │   ├── xxx.jpg
│   │   └── ...
│   └── abnormal/
│       ├── yyy.jpg
│       └── ...
├── 001801/
│   ├── normal/
│   └── abnormal/
└── split.json


输出结构：

OUTPUT_ROOT/
├── 000071/
│   ├── xxx.jpg
│   ├── yyy.jpg
│   └── ...
├── 001801/
│   └── ...
├── split.json
├── merge_manifest.csv
└── merge_summary.csv


注意：

1. normal 和 abnormal 图片复制后混合在同一个 cow_id 文件夹；
2. 图片原文件名完全不变；
3. 如果 normal 和 abnormal 中出现同名文件，程序直接报错；
4. 不修改原始数据集；
5. split.json 会复制到输出根目录；
6. 输出 manifest，记录每张图原来的标签。
"""

import csv
import shutil

from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


# =========================================================
# 1. 路径配置
# =========================================================

# 刚生成的、仍包含 normal/abnormal 子文件夹的数据集
SOURCE_ROOT = Path(
    r"D:\datasets\cow_recognition8.3"
    r"\openset\openset_consensus_hard_11to1_v2"
)

# 新生成的混合数据集，结构类似 dataset_obb_ori
OUTPUT_ROOT = Path(
    r"D:\datasets\cow_recognition8.3"
    r"\openset\openset_consensus_hard_11to1_v2_mixed"
)


# =========================================================
# 2. 参数配置
# =========================================================

# True：
# 每次运行前删除旧输出目录，然后重新生成。
CLEAR_OUTPUT_ROOT = True


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
    只识别包含 normal 或 abnormal 子目录的一级目录。
    """

    result: Dict[str, Path] = {}

    for path in root.iterdir():

        if not path.is_dir():
            continue

        normal_dir = path / "normal"
        abnormal_dir = path / "abnormal"

        if (
            normal_dir.is_dir()
            or abnormal_dir.is_dir()
        ):
            result[path.name] = path

    return dict(
        sorted(
            result.items(),
            key=lambda item: item[0],
        )
    )


def save_csv(
    rows: List[dict],
    save_path: Path,
) -> None:
    if not rows:
        print(
            f"[WARN] 没有数据，未保存：{save_path}"
        )
        return

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with save_path.open(
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
        f"[SAVE] {save_path}"
    )


# =========================================================
# 4. 检查文件名冲突
# =========================================================

def check_filename_conflicts(
    cow_id: str,
    normal_images: List[Path],
    abnormal_images: List[Path],
) -> None:
    """
    Windows文件系统通常不区分大小写，因此使用casefold检查。

    如果normal与abnormal出现同名文件，不能直接覆盖。
    因为后续GT匹配需要保留原文件名。
    """

    normal_name_map = {
        path.name.casefold(): path
        for path in normal_images
    }

    abnormal_name_map = {
        path.name.casefold(): path
        for path in abnormal_images
    }

    duplicated_names = (
        set(normal_name_map.keys())
        & set(abnormal_name_map.keys())
    )

    if not duplicated_names:
        return

    conflict_lines = []

    for filename_key in sorted(
        duplicated_names
    ):
        normal_path = normal_name_map[
            filename_key
        ]

        abnormal_path = abnormal_name_map[
            filename_key
        ]

        conflict_lines.append(
            "\n".join([
                f"文件名：{normal_path.name}",
                f"normal：{normal_path}",
                f"abnormal：{abnormal_path}",
            ])
        )

    conflict_text = "\n\n".join(
        conflict_lines
    )

    raise RuntimeError(
        f"牛编号 {cow_id} 的 normal 和 abnormal "
        f"存在同名文件，不能直接混合：\n\n"
        f"{conflict_text}\n\n"
        "程序已停止，避免覆盖图片以及破坏GT文件名对应关系。"
    )


# =========================================================
# 5. 复制单个编号
# =========================================================

def merge_one_cow(
    cow_id: str,
    source_cow_dir: Path,
) -> dict:
    normal_dir = (
        source_cow_dir
        / "normal"
    )

    abnormal_dir = (
        source_cow_dir
        / "abnormal"
    )

    normal_images = list_images(
        normal_dir
    )

    abnormal_images = list_images(
        abnormal_dir
    )

    check_filename_conflicts(
        cow_id=cow_id,
        normal_images=normal_images,
        abnormal_images=abnormal_images,
    )

    output_cow_dir = (
        OUTPUT_ROOT
        / cow_id
    )

    output_cow_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []

    # -----------------------------------------------------
    # 复制 normal
    # -----------------------------------------------------

    for source_path in normal_images:

        destination_path = (
            output_cow_dir
            / source_path.name
        )

        if destination_path.exists():
            raise FileExistsError(
                f"输出文件已经存在：{destination_path}"
            )

        shutil.copy2(
            source_path,
            destination_path,
        )

        manifest_rows.append({
            "cow_id": cow_id,
            "original_label": "normal",
            "filename": source_path.name,
            "source_path": str(
                source_path
            ),
            "destination_path": str(
                destination_path
            ),
        })

    # -----------------------------------------------------
    # 复制 abnormal
    # -----------------------------------------------------

    for source_path in abnormal_images:

        destination_path = (
            output_cow_dir
            / source_path.name
        )

        if destination_path.exists():
            raise FileExistsError(
                f"输出文件已经存在：{destination_path}"
            )

        shutil.copy2(
            source_path,
            destination_path,
        )

        manifest_rows.append({
            "cow_id": cow_id,
            "original_label": "abnormal",
            "filename": source_path.name,
            "source_path": str(
                source_path
            ),
            "destination_path": str(
                destination_path
            ),
        })

    summary_row = {
        "cow_id": cow_id,
        "normal_count": len(
            normal_images
        ),
        "abnormal_count": len(
            abnormal_images
        ),
        "mixed_total_count": (
            len(normal_images)
            + len(abnormal_images)
        ),
        "normal_abnormal_ratio": (
            (
                len(normal_images)
                / len(abnormal_images)
            )
            if abnormal_images
            else ""
        ),
        "output_dir": str(
            output_cow_dir
        ),
    }

    return {
        "manifest_rows": (
            manifest_rows
        ),
        "summary_row": (
            summary_row
        ),
    }


# =========================================================
# 6. 复制数据集根目录文件
# =========================================================

def copy_root_metadata() -> None:
    """
    split.json需要复制，方便后续测试读取unseen_cows。

    其他CSV和config也一并复制，但添加source_前缀，
    避免与本脚本生成的manifest冲突。
    """

    files_to_copy = {
        "split.json": "split.json",
        "config.json": "source_config.json",
        "labels.csv": "source_labels.csv",
        "dataset_summary.csv": (
            "source_dataset_summary.csv"
        ),
        "selected_abnormal_manifest.csv": (
            "source_selected_abnormal_manifest.csv"
        ),
        "candidate_pool_manifest.csv": (
            "source_candidate_pool_manifest.csv"
        ),
        "candidate_sources_before_dedup.csv": (
            "source_candidate_sources_before_dedup.csv"
        ),
    }

    for source_name, output_name in (
        files_to_copy.items()
    ):
        source_path = (
            SOURCE_ROOT
            / source_name
        )

        if not source_path.exists():
            print(
                f"[WARN] 根目录文件不存在，跳过："
                f"{source_path}"
            )
            continue

        destination_path = (
            OUTPUT_ROOT
            / output_name
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        print(
            f"[COPY] {source_path.name} "
            f"→ {destination_path.name}"
        )


# =========================================================
# 7. 主函数
# =========================================================

def main() -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"输入数据集不存在：{SOURCE_ROOT}"
        )

    if (
        SOURCE_ROOT.resolve()
        == OUTPUT_ROOT.resolve()
    ):
        raise RuntimeError(
            "SOURCE_ROOT和OUTPUT_ROOT不能相同，"
            "否则会破坏原始数据。"
        )

    if CLEAR_OUTPUT_ROOT and OUTPUT_ROOT.exists():

        print(
            f"[CLEAR] 删除旧输出目录："
            f"{OUTPUT_ROOT}"
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    cow_dirs = discover_cow_dirs(
        SOURCE_ROOT
    )

    if not cow_dirs:
        raise RuntimeError(
            f"没有发现包含normal或abnormal的"
            f"牛编号目录：{SOURCE_ROOT}"
        )

    print("=" * 100)
    print(f"[SOURCE ROOT] {SOURCE_ROOT}")
    print(f"[OUTPUT ROOT] {OUTPUT_ROOT}")
    print(f"[COW COUNT]   {len(cow_dirs)}")
    print("[MODE]        normal与abnormal混合")
    print("[FILENAME]    保持原文件名不变")
    print("=" * 100)

    all_manifest_rows = []
    all_summary_rows = []

    for index, (
        cow_id,
        source_cow_dir,
    ) in enumerate(
        tqdm(
            cow_dirs.items(),
            desc="混合各牛编号图片",
        ),
        start=1,
    ):
        result = merge_one_cow(
            cow_id=cow_id,
            source_cow_dir=(
                source_cow_dir
            ),
        )

        all_manifest_rows.extend(
            result[
                "manifest_rows"
            ]
        )

        all_summary_rows.append(
            result[
                "summary_row"
            ]
        )

        summary = result[
            "summary_row"
        ]

        print(
            f"[{index}/{len(cow_dirs)}] "
            f"{cow_id}: "
            f"normal={summary['normal_count']}, "
            f"abnormal={summary['abnormal_count']}, "
            f"total={summary['mixed_total_count']}"
        )

    # 复制split.json及原始构建记录
    copy_root_metadata()

    # 保存本次混合记录
    save_csv(
        all_manifest_rows,
        OUTPUT_ROOT
        / "merge_manifest.csv",
    )

    save_csv(
        all_summary_rows,
        OUTPUT_ROOT
        / "merge_summary.csv",
    )

    total_normal = sum(
        row["normal_count"]
        for row in all_summary_rows
    )

    total_abnormal = sum(
        row["abnormal_count"]
        for row in all_summary_rows
    )

    total_images = (
        total_normal
        + total_abnormal
    )

    print("\n" + "=" * 100)
    print("[ALL FINISHED]")
    print(f"[OUTPUT ROOT]    {OUTPUT_ROOT}")
    print(f"[COW COUNT]      {len(all_summary_rows)}")
    print(f"[NORMAL COUNT]   {total_normal}")
    print(f"[ABNORMAL COUNT] {total_abnormal}")
    print(f"[TOTAL IMAGES]   {total_images}")
    print("=" * 100)


if __name__ == "__main__":
    main()