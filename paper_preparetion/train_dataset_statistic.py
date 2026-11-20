# -*- coding: utf-8 -*-

import csv
import json
from collections import Counter
from pathlib import Path


DATASET_ROOT = Path(
    r"D:\datasets\cow_recognition8.3\train_dataset"
)

SPLIT_JSON = DATASET_ROOT / "split.json"

OUTPUT_CSV = Path(
    r"D:\code_project\cow_recognition\outputs\train_dataset_summary.csv"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}


def count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0

    return sum(
        1
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():

    with open(SPLIT_JSON, "r", encoding="utf-8") as file:
        split_data = json.load(file)

    cow_ids = [
        str(cow_id)
        for cow_id in split_data["seen_cows"]
    ]

    train_counter = Counter(
        str(sample["cow_id"])
        for sample in split_data["seen_train"]
    )

    val_counter = Counter(
        str(sample["cow_id"])
        for sample in split_data["seen_val"]
    )

    total_normal = 0
    total_abnormal = 0
    total_train = 0
    total_val = 0

    # 保存每头牛各自的normal/abnormal比值
    per_cow_ratios = []

    consistency_results = []

    for cow_id in cow_ids:

        cow_dir = DATASET_ROOT / cow_id

        normal_count = count_images(
            cow_dir / "normal"
        )

        abnormal_count = count_images(
            cow_dir / "abnormal"
        )

        train_count = train_counter[cow_id]
        val_count = val_counter[cow_id]

        total_normal += normal_count
        total_abnormal += abnormal_count
        total_train += train_count
        total_val += val_count

        # 每头牛分别计算normal/abnormal
        if abnormal_count > 0:
            per_cow_ratios.append(
                normal_count / abnormal_count
            )

        consistency_results.append(
            normal_count
            ==
            train_count + val_count
        )

    cow_count = len(cow_ids)

    # 逐牛计算normal/abnormal后再平均
    macro_normal_abnormal_ratio = (
        sum(per_cow_ratios)
        /
        len(per_cow_ratios)
    )

    summary = {
        "cow_count": cow_count,

        "normal_total": total_normal,
        "abnormal_total": total_abnormal,
        "all_images_total": (
            total_normal + total_abnormal
        ),

        "train_total": total_train,
        "val_total": total_val,
        "train_val_total": (
            total_train + total_val
        ),

        "average_normal_per_cow": round(
            total_normal / cow_count, 4
        ),

        "average_abnormal_per_cow": round(
            total_abnormal / cow_count, 4
        ),

        "average_all_images_per_cow": round(
            (total_normal + total_abnormal)
            / cow_count,
            4,
        ),

        "average_train_per_cow": round(
            total_train / cow_count, 4
        ),

        "average_val_per_cow": round(
            total_val / cow_count, 4
        ),

        # 最重要的结果：每头牛分别计算后再平均
        "mean_per_cow_normal_abnormal_ratio": round(
            macro_normal_abnormal_ratio,
            4,
        ),

        "ratio_valid_cow_count": len(
            per_cow_ratios
        ),

        "normal_equals_train_plus_val": all(
            consistency_results
        ),
    }

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # CSV中只写一行汇总结果
    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary.keys()),
        )

        writer.writeheader()
        writer.writerow(summary)

    print("=" * 60)
    print("数据集汇总统计完成")
    print("=" * 60)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nCSV保存位置：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()