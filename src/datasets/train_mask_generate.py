## 对训练集生成掩码 便于后续数据增强
# -*- coding: utf-8 -*-

import csv
import json

import cv2
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


# =========================================================
# 1. 路径配置
# =========================================================

# YOLO-seg模型
MODEL_PATH = Path(
    r"D:\code_project\cow_recognition\model\best.pt"
)

# 新训练数据集
INPUT_ROOT_DIR = Path(
    r"D:\datasets\cow_recognition8.3\train_dataset"
)

# 新训练数据集的划分文件
SPLIT_JSON_PATH = (
    INPUT_ROOT_DIR / "split.json"
)

# 浅色区域mask输出目录
OUTPUT_MASK_ROOT_DIR = Path(
    r"D:\datasets\cow_recognition8.3"
    r"\train_dataset_light_masks"
)

# 失败记录
FAIL_CSV_PATH = (
    OUTPUT_MASK_ROOT_DIR
    /
    "failed_masks.csv"
)

# mask统计信息
STATS_CSV_PATH = (
    OUTPUT_MASK_ROOT_DIR
    /
    "mask_statistics.csv"
)


# =========================================================
# 2. YOLO推理参数
# =========================================================

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.6
IMG_SIZE = 640

DEVICE = (
    "0"
    if torch.cuda.is_available()
    else "cpu"
)

USE_HALF = torch.cuda.is_available()


# =========================================================
# 3. 浅色区域提取参数
# =========================================================

# 仅处理split.json中的这些部分
# 训练增强只作用于seen_train，因此默认只生成seen_train的mask
SPLIT_KEYS = [
    "seen_train"
]

# HSV中V通道的最低亮度阈值
# 最终阈值取：
# max(Otsu自动阈值, MIN_LIGHT_V)
MIN_LIGHT_V = 80

# 牛背分割mask最少像素数
MIN_COW_MASK_AREA = 500

# 浅色区域最少像素数
MIN_LIGHT_MASK_AREA = 100

# 删除浅色mask中的小连通区域
MIN_COMPONENT_AREA = 50

# 形态学核大小，必须是奇数
MORPH_KERNEL_SIZE = 5

# 是否覆盖已经存在的mask
OVERWRITE_EXISTING = False


# =========================================================
# 4. 图片格式
# =========================================================

IMG_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# =========================================================
# 5. 中文路径读写
# =========================================================

def read_image_chinese_path(
    image_path: Path,
):
    """
    支持中文路径的OpenCV图片读取。
    """

    try:
        image_data = np.fromfile(
            str(image_path),
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            image_data,
            cv2.IMREAD_COLOR,
        )

        return image

    except Exception:
        return None


def save_mask_chinese_path(
    save_path: Path,
    mask: np.ndarray,
) -> bool:
    """
    以PNG格式保存单通道二值mask。
    白色255：需要增强的浅色区域
    黑色0：不增强区域
    """

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        success, encoded = cv2.imencode(
            ".png",
            mask,
        )

        if not success:
            return False

        encoded.tofile(
            str(save_path)
        )

        return True

    except Exception:
        return False


# =========================================================
# 6. 读取split.json
# =========================================================

def load_training_samples(
    split_json_path: Path,
    split_keys,
):
    """
    从split.json读取指定数据划分。

    默认只读取seen_train。
    """

    if not split_json_path.exists():
        raise FileNotFoundError(
            f"找不到split.json：{split_json_path}"
        )

    with open(
        split_json_path,
        "r",
        encoding="utf-8",
    ) as file:
        split_data = json.load(file)

    samples = []

    for split_key in split_keys:

        if split_key not in split_data:
            raise KeyError(
                f"split.json中不存在字段：{split_key}"
            )

        for sample in split_data[split_key]:

            if "path" not in sample:
                continue

            if "cow_id" not in sample:
                continue

            samples.append({
                "path": sample["path"],
                "cow_id": str(sample["cow_id"]),
                "split": split_key,
            })

    # 根据路径去重
    unique_samples = {}

    for sample in samples:
        unique_samples[
            str(sample["path"])
        ] = sample

    samples = list(
        unique_samples.values()
    )

    return samples


# =========================================================
# 7. 路径处理
# =========================================================

def resolve_image_path(
    sample,
    input_root: Path,
) -> Path:
    """
    优先使用split.json中的绝对路径。

    如果绝对路径失效，则尝试：
    INPUT_ROOT/cow_id/normal/文件名
    """

    original_path = Path(
        sample["path"]
    )

    if original_path.exists():
        return original_path

    fallback_path = (
        input_root
        /
        sample["cow_id"]
        /
        "normal"
        /
        original_path.name
    )

    if fallback_path.exists():
        return fallback_path

    return original_path


def build_output_mask_path(
    image_path: Path,
    cow_id: str,
    input_root: Path,
    output_root: Path,
) -> Path:
    """
    根据原图路径生成对应mask路径。

    正常情况下：
    input_root/cow_id/normal/a.jpg
    转换为：
    output_root/cow_id/normal/a.png
    """

    try:
        relative_path = (
            image_path.resolve()
            .relative_to(
                input_root.resolve()
            )
        )

    except ValueError:
        # 如果原图不在input_root内部，
        # 使用cow_id/normal/文件名作为回退结构
        relative_path = (
            Path(cow_id)
            /
            "normal"
            /
            image_path.name
        )

    mask_relative_path = (
        relative_path.with_suffix(
            ".png"
        )
    )

    return (
        output_root
        /
        mask_relative_path
    )


# =========================================================
# 8. 删除小连通区域
# =========================================================

def remove_small_components(
    binary_mask: np.ndarray,
    min_area: int,
) -> np.ndarray:
    """
    删除面积小于min_area的白色连通区域。
    """

    binary_mask = (
        binary_mask > 0
    ).astype(np.uint8)

    number_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
    )

    cleaned_mask = np.zeros_like(
        binary_mask,
        dtype=np.uint8,
    )

    for label_index in range(
        1,
        number_labels,
    ):
        component_area = stats[
            label_index,
            cv2.CC_STAT_AREA,
        ]

        if component_area >= min_area:
            cleaned_mask[
                labels == label_index
            ] = 255

    return cleaned_mask


# =========================================================
# 9. 牛背浅色区域提取
# =========================================================

def extract_light_region_mask(
    image: np.ndarray,
    cow_mask: np.ndarray,
):
    """
    在YOLO牛背mask内部提取浅色非黑区域。

    方法：
    1. 转换到HSV颜色空间；
    2. 读取V亮度通道；
    3. 在牛背mask内部计算Otsu自动阈值；
    4. 排除黑色花纹；
    5. 进行形态学去噪；
    6. 去除小连通区域。

    返回：
    light_mask
    brightness_threshold
    cow_mask_area
    light_mask_area
    light_ratio
    """

    cow_mask_binary = (
        cow_mask > 0
    ).astype(np.uint8)

    cow_mask_area = int(
        cow_mask_binary.sum()
    )

    if cow_mask_area < MIN_COW_MASK_AREA:
        return None, None, cow_mask_area, 0, 0.0

    hsv_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    value_channel = hsv_image[
        :,
        :,
        2,
    ]

    # 只使用牛背分割区域内部的亮度值
    valid_values = value_channel[
        cow_mask_binary > 0
    ]

    if valid_values.size == 0:
        return None, None, cow_mask_area, 0, 0.0

    valid_values_for_otsu = (
        valid_values
        .astype(np.uint8)
        .reshape(-1, 1)
    )

    # Otsu自动区分牛背浅色区域与黑色花纹
    otsu_threshold, _ = cv2.threshold(
        valid_values_for_otsu,
        0,
        255,
        cv2.THRESH_BINARY
        +
        cv2.THRESH_OTSU,
    )

    brightness_threshold = max(
        int(round(otsu_threshold)),
        MIN_LIGHT_V,
    )

    # 防止异常阈值超过有效范围
    brightness_threshold = min(
        brightness_threshold,
        250,
    )

    light_mask = (
        (
            cow_mask_binary > 0
        )
        &
        (
            value_channel
            >=
            brightness_threshold
        )
    ).astype(np.uint8) * 255

    # 形态学处理
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE,
        ),
    )

    # 删除小噪声
    light_mask = cv2.morphologyEx(
        light_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    # 填充浅色区域内部的小缺口
    light_mask = cv2.morphologyEx(
        light_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # 确保浅色mask绝不超出牛背分割区域
    light_mask[
        cow_mask_binary == 0
    ] = 0

    light_mask = remove_small_components(
        binary_mask=light_mask,
        min_area=MIN_COMPONENT_AREA,
    )

    light_mask_area = int(
        np.count_nonzero(
            light_mask
        )
    )

    light_ratio = (
        light_mask_area
        /
        cow_mask_area
        if cow_mask_area > 0
        else 0.0
    )

    if light_mask_area < MIN_LIGHT_MASK_AREA:
        return (
            None,
            brightness_threshold,
            cow_mask_area,
            light_mask_area,
            light_ratio,
        )

    return (
        light_mask,
        brightness_threshold,
        cow_mask_area,
        light_mask_area,
        light_ratio,
    )


# =========================================================
# 10. 单张图片处理
# =========================================================

def process_single_image(
    model,
    image_path: Path,
    output_mask_path: Path,
):
    """
    单张图片：
    YOLO牛背分割
    +
    浅色非黑区域mask提取
    """

    image = read_image_chinese_path(
        image_path
    )

    if image is None:
        return {
            "success": False,
            "reason": "image_read_failed",
        }

    try:
        results = model(
            image,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False,
            retina_masks=True,
            half=USE_HALF,
        )

    except Exception as exception:
        return {
            "success": False,
            "reason": (
                f"yolo_inference_failed: "
                f"{exception}"
            ),
        }

    if not results:
        return {
            "success": False,
            "reason": "empty_yolo_result",
        }

    result = results[0]

    if result.masks is None:
        return {
            "success": False,
            "reason": "no_segmentation_mask",
        }

    masks = result.masks.data

    if masks is None or len(masks) == 0:
        return {
            "success": False,
            "reason": "empty_segmentation_masks",
        }

    # 选择面积最大的YOLO分割mask
    mask_areas = []

    for mask_tensor in masks:
        area = float(
            mask_tensor
            .detach()
            .float()
            .sum()
            .cpu()
            .item()
        )

        mask_areas.append(
            area
        )

    best_mask_index = int(
        np.argmax(mask_areas)
    )

    best_mask = (
        masks[best_mask_index]
        .detach()
        .cpu()
        .numpy()
    )

    best_mask = (
        best_mask > 0.5
    ).astype(np.uint8) * 255

    # 调整到原图尺寸
    cow_mask_resized = cv2.resize(
        best_mask,
        (
            image.shape[1],
            image.shape[0],
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    (
        light_mask,
        brightness_threshold,
        cow_mask_area,
        light_mask_area,
        light_ratio,
    ) = extract_light_region_mask(
        image=image,
        cow_mask=cow_mask_resized,
    )

    if light_mask is None:
        return {
            "success": False,
            "reason": "light_region_too_small",
            "threshold": brightness_threshold,
            "cow_mask_area": cow_mask_area,
            "light_mask_area": light_mask_area,
            "light_ratio": light_ratio,
        }

    save_success = save_mask_chinese_path(
        save_path=output_mask_path,
        mask=light_mask,
    )

    if not save_success:
        return {
            "success": False,
            "reason": "mask_save_failed",
            "threshold": brightness_threshold,
            "cow_mask_area": cow_mask_area,
            "light_mask_area": light_mask_area,
            "light_ratio": light_ratio,
        }

    return {
        "success": True,
        "reason": "success",
        "threshold": brightness_threshold,
        "cow_mask_area": cow_mask_area,
        "light_mask_area": light_mask_area,
        "light_ratio": light_ratio,
    }


# =========================================================
# 11. CSV记录
# =========================================================

def write_fail_record(
    writer,
    sample,
    image_path: Path,
    output_mask_path: Path,
    result,
):
    writer.writerow([
        sample["cow_id"],
        sample["split"],
        image_path.name,
        str(image_path),
        str(output_mask_path),
        result.get(
            "reason",
            "unknown",
        ),
    ])


def write_statistics_record(
    writer,
    sample,
    image_path: Path,
    output_mask_path: Path,
    result,
):
    writer.writerow([
        sample["cow_id"],
        sample["split"],
        image_path.name,
        str(image_path),
        str(output_mask_path),
        result.get(
            "threshold",
            "",
        ),
        result.get(
            "cow_mask_area",
            "",
        ),
        result.get(
            "light_mask_area",
            "",
        ),
        result.get(
            "light_ratio",
            "",
        ),
    ])


# =========================================================
# 12. 主程序
# =========================================================

def main():

    print("=" * 80)
    print("生成训练集浅色区域掩码")
    print("=" * 80)

    print(
        f"YOLO模型：{MODEL_PATH}"
    )

    print(
        f"训练数据集：{INPUT_ROOT_DIR}"
    )

    print(
        f"split.json：{SPLIT_JSON_PATH}"
    )

    print(
        f"mask输出目录：{OUTPUT_MASK_ROOT_DIR}"
    )

    print(
        f"处理划分：{SPLIT_KEYS}"
    )

    print(
        f"推理设备：{DEVICE}"
    )

    print(
        f"最低亮度阈值：{MIN_LIGHT_V}"
    )

    print("=" * 80)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO模型不存在：{MODEL_PATH}"
        )

    if not INPUT_ROOT_DIR.exists():
        raise FileNotFoundError(
            f"训练数据集不存在：{INPUT_ROOT_DIR}"
        )

    if not SPLIT_JSON_PATH.exists():
        raise FileNotFoundError(
            f"split.json不存在：{SPLIT_JSON_PATH}"
        )

    OUTPUT_MASK_ROOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FAIL_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 读取训练样本
    samples = load_training_samples(
        split_json_path=SPLIT_JSON_PATH,
        split_keys=SPLIT_KEYS,
    )

    print(
        f"\n读取到训练图片数量：{len(samples)}"
    )

    if len(samples) == 0:
        print(
            "[WARNING] split.json中没有读取到训练图片。"
        )
        return

    # 加载YOLO模型
    model = YOLO(
        str(MODEL_PATH)
    )

    total_count = len(samples)
    success_count = 0
    fail_count = 0
    skip_count = 0

    cow_success_counter = {}
    cow_fail_counter = {}

    with open(
        FAIL_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as fail_file, open(
        STATS_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stats_file:

        fail_writer = csv.writer(
            fail_file
        )

        stats_writer = csv.writer(
            stats_file
        )

        fail_writer.writerow([
            "cow_id",
            "split",
            "image_name",
            "image_path",
            "mask_path",
            "failure_reason",
        ])

        stats_writer.writerow([
            "cow_id",
            "split",
            "image_name",
            "image_path",
            "mask_path",
            "brightness_threshold",
            "cow_mask_area",
            "light_mask_area",
            "light_ratio",
        ])

        for sample in tqdm(
            samples,
            desc="生成浅色区域mask",
        ):
            cow_id = sample["cow_id"]

            image_path = resolve_image_path(
                sample=sample,
                input_root=INPUT_ROOT_DIR,
            )

            output_mask_path = build_output_mask_path(
                image_path=image_path,
                cow_id=cow_id,
                input_root=INPUT_ROOT_DIR,
                output_root=OUTPUT_MASK_ROOT_DIR,
            )

            if not image_path.exists():

                result = {
                    "success": False,
                    "reason": "image_not_found",
                }

                write_fail_record(
                    writer=fail_writer,
                    sample=sample,
                    image_path=image_path,
                    output_mask_path=output_mask_path,
                    result=result,
                )

                fail_count += 1

                cow_fail_counter[cow_id] = (
                    cow_fail_counter.get(
                        cow_id,
                        0,
                    )
                    +
                    1
                )

                continue

            if (
                output_mask_path.exists()
                and
                not OVERWRITE_EXISTING
            ):
                skip_count += 1
                continue

            result = process_single_image(
                model=model,
                image_path=image_path,
                output_mask_path=output_mask_path,
            )

            if result["success"]:

                success_count += 1

                cow_success_counter[cow_id] = (
                    cow_success_counter.get(
                        cow_id,
                        0,
                    )
                    +
                    1
                )

                write_statistics_record(
                    writer=stats_writer,
                    sample=sample,
                    image_path=image_path,
                    output_mask_path=output_mask_path,
                    result=result,
                )

            else:

                fail_count += 1

                cow_fail_counter[cow_id] = (
                    cow_fail_counter.get(
                        cow_id,
                        0,
                    )
                    +
                    1
                )

                write_fail_record(
                    writer=fail_writer,
                    sample=sample,
                    image_path=image_path,
                    output_mask_path=output_mask_path,
                    result=result,
                )

    print("\n" + "=" * 80)
    print("训练集浅色区域mask生成完成")
    print("=" * 80)

    print(
        f"总训练图片：{total_count}"
    )

    print(
        f"成功生成：{success_count}"
    )

    print(
        f"处理失败：{fail_count}"
    )

    print(
        f"跳过已有：{skip_count}"
    )

    print(
        f"mask目录：{OUTPUT_MASK_ROOT_DIR}"
    )

    print(
        f"失败记录：{FAIL_CSV_PATH}"
    )

    print(
        f"统计文件：{STATS_CSV_PATH}"
    )

    print("\n各牛编号处理结果：")

    all_cow_ids = sorted(
        set(cow_success_counter.keys())
        |
        set(cow_fail_counter.keys())
    )

    for cow_id in all_cow_ids:

        cow_success = cow_success_counter.get(
            cow_id,
            0,
        )

        cow_fail = cow_fail_counter.get(
            cow_id,
            0,
        )

        print(
            f"牛编号={cow_id} | "
            f"成功={cow_success} | "
            f"失败={cow_fail}"
        )

    print("=" * 80)


# =========================================================
# 13. 启动入口
# =========================================================

if __name__ == "__main__":
    main()