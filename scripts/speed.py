# -*- coding: utf-8 -*-
r"""
Backbone 消融实验推理速度测试。

测试内容：
1. 调用 model.forward_emb()，与开放集测试保持一致；
2. 不包含磁盘读取、PIL解码和图像预处理；
3. 输入尺寸统一为 3×224×224；
4. 精度统一为 FP32；
5. GPU测试使用 CUDA Event 准确计时；
6. 正式计时前进行多轮 warm-up；
7. 输出：
    - 单个 batch 平均耗时；
    - 单张图片平均耗时；
    - 中位数延迟；
    - P95延迟；
    - FPS / throughput；
    - 模型参数量；
    - checkpoint大小；
    - CUDA峰值显存；
8. 分别测试 batch_size=1 和 batch_size=32；
9. 所有模型结果保存到同一个 CSV。

注意：
论文中建议主要报告：
    batch_size=1 的 latency_per_image_ms 和 fps

batch_size=32 主要反映批量处理吞吐量。
"""

import csv
import gc
import random
import sys
import time

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import torch


# =========================================================
# 0. 自动寻找项目根目录
# =========================================================

CURRENT_FILE = Path(__file__).resolve()


def find_project_root(
    current_file: Path,
) -> Path:
    """
    从当前脚本开始逐级向上查找包含 src 的目录。

    脚本放在以下任意位置都能运行：

    scripts/test/
    scripts/test_analysis/
    scripts/test_analysis/speed/
    """

    search_dirs = [
        current_file.parent,
        *current_file.parents,
    ]

    for directory in search_dirs:
        if (
            directory / "src"
        ).is_dir():
            return directory

    searched_dirs = "\n".join(
        str(directory)
        for directory in search_dirs
    )

    raise FileNotFoundError(
        "没有找到包含 src 文件夹的项目根目录。\n"
        f"当前脚本：{current_file}\n"
        f"已经搜索：\n{searched_dirs}"
    )


PROJECT_ROOT = find_project_root(
    CURRENT_FILE
)

SRC_ROOT = (
    PROJECT_ROOT
    / "src"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


print("=" * 100)
print(f"[SCRIPT]       {CURRENT_FILE}")
print(f"[PROJECT_ROOT] {PROJECT_ROOT}")
print(f"[SRC_ROOT]     {SRC_ROOT}")
print("=" * 100)


from src.models.arcface_model_ablation import (
    ArcFaceCowModel,
)


# =========================================================
# 1. 模型配置
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

    # HRNet训练完成后取消注释。
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
# 2. 输出路径
# =========================================================

OUTPUT_DIR = Path(
    r"D:\code_project\cow_recognition"
    r"\outputs\ablation_inference_speed"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "inference_speed_summary.csv"
)


# =========================================================
# 3. 速度测试参数
# =========================================================

IMG_SIZE = 224

# batch=1用于单张图片延迟。
# batch=32用于批量吞吐量。
BATCH_SIZES = [
    1,
    32,
]

# 预热次数。
# GPU第一次运行通常包含CUDA初始化和算法搜索，
# 不能直接计入正式速度。
WARMUP_STEPS = 50

# 正式计时次数。
MEASURE_STEPS = 200

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# 4. 基础配置
# =========================================================

def set_seed(
    seed: int = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clear_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# 推理速度测试允许cuDNN自动选择最快算法。
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# =========================================================
# 5. checkpoint加载
# =========================================================

def normalize_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    处理可能存在的：
        module.
        _orig_mod.
    前缀。

    没有这些前缀时不做任何修改。
    """

    normalized = {}

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

        normalized[new_key] = value

    return normalized


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[dict, Dict[str, torch.Tensor]]:
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

    if not isinstance(
        state_dict,
        dict,
    ):
        raise TypeError(
            f"checkpoint中没有有效state_dict：{path}"
        )

    state_dict = normalize_state_dict_keys(
        state_dict
    )

    return (
        checkpoint,
        state_dict,
    )


# =========================================================
# 6. 构建模型
# =========================================================

def build_model(
    model_path: str,
    backbone: str,
    device: torch.device,
):
    checkpoint, state_dict = (
        load_checkpoint(
            checkpoint_path=model_path
        )
    )

    if "arcface.weight" not in state_dict:
        raise KeyError(
            "checkpoint中没有arcface.weight，"
            "无法推断num_classes和embedding_dim。"
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

    checkpoint_epoch = (
        checkpoint.get(
            "epoch",
            "unknown",
        )
        if isinstance(
            checkpoint,
            dict,
        )
        else "unknown"
    )

    print(
        f"[MODEL] backbone="
        f"{backbone}"
    )
    print(
        f"[MODEL] checkpoint_epoch="
        f"{checkpoint_epoch}"
    )
    print(
        f"[MODEL] num_classes="
        f"{num_classes}"
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
            "checkpoint与模型结构不匹配，"
            "该速度结果不能用于公平比较。\n"
            f"backbone={backbone}\n"
            f"missing={len(missing_keys)}\n"
            f"unexpected={len(unexpected_keys)}"
        )

    model = model.to(
        device
    )

    model.eval()

    return {
        "model": model,
        "checkpoint_epoch": (
            checkpoint_epoch
        ),
        "num_classes": (
            num_classes
        ),
        "embedding_dim": (
            embedding_dim
        ),
    }


# =========================================================
# 7. 参数量计算
# =========================================================

def count_parameters(
    model,
) -> Tuple[int, int]:
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return (
        int(total_parameters),
        int(trainable_parameters),
    )


# =========================================================
# 8. GPU计时
# =========================================================

@torch.inference_mode()
def benchmark_cuda(
    model,
    input_tensor: torch.Tensor,
    warmup_steps: int,
    measure_steps: int,
) -> List[float]:
    """
    返回每次batch前向传播的时间，单位为ms。

    使用CUDA Event，只计算GPU执行时间，
    不计算Python调度与数据传输时间。
    """

    # -------------------------------
    # warm-up
    # -------------------------------

    for _ in range(
        warmup_steps
    ):
        _ = model.forward_emb(
            input_tensor
        )

    torch.cuda.synchronize()

    # 重置峰值显存统计。
    torch.cuda.reset_peak_memory_stats()

    start_events = [
        torch.cuda.Event(
            enable_timing=True
        )
        for _ in range(
            measure_steps
        )
    ]

    end_events = [
        torch.cuda.Event(
            enable_timing=True
        )
        for _ in range(
            measure_steps
        )
    ]

    # -------------------------------
    # 正式计时
    # -------------------------------

    for index in range(
        measure_steps
    ):
        start_events[index].record()

        _ = model.forward_emb(
            input_tensor
        )

        end_events[index].record()

    torch.cuda.synchronize()

    batch_times_ms = [
        float(
            start_event.elapsed_time(
                end_event
            )
        )
        for start_event, end_event in zip(
            start_events,
            end_events,
        )
    ]

    return batch_times_ms


# =========================================================
# 9. CPU计时
# =========================================================

@torch.inference_mode()
def benchmark_cpu(
    model,
    input_tensor: torch.Tensor,
    warmup_steps: int,
    measure_steps: int,
) -> List[float]:
    """
    CPU使用time.perf_counter计时。
    """

    for _ in range(
        warmup_steps
    ):
        _ = model.forward_emb(
            input_tensor
        )

    batch_times_ms = []

    for _ in range(
        measure_steps
    ):
        start_time = (
            time.perf_counter()
        )

        _ = model.forward_emb(
            input_tensor
        )

        end_time = (
            time.perf_counter()
        )

        elapsed_ms = (
            end_time - start_time
        ) * 1000.0

        batch_times_ms.append(
            float(elapsed_ms)
        )

    return batch_times_ms


# =========================================================
# 10. 单个batch size测试
# =========================================================

def benchmark_batch_size(
    model,
    batch_size: int,
    img_size: int,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
) -> Dict[str, float]:
    """
    使用固定随机输入。

    输入已经位于GPU中，因此不包含：
    - 磁盘读取；
    - 图片解码；
    - Resize；
    - Normalize；
    - CPU到GPU的数据传输。
    """

    input_tensor = torch.randn(
        batch_size,
        3,
        img_size,
        img_size,
        dtype=torch.float32,
        device=device,
    )

    if device.type == "cuda":
        batch_times_ms = benchmark_cuda(
            model=model,
            input_tensor=input_tensor,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
        )

        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / 1024.0
            / 1024.0
        )

    else:
        batch_times_ms = benchmark_cpu(
            model=model,
            input_tensor=input_tensor,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
        )

        peak_memory_mb = 0.0

    times = np.asarray(
        batch_times_ms,
        dtype=np.float64,
    )

    batch_mean_ms = float(
        np.mean(times)
    )

    batch_std_ms = float(
        np.std(times)
    )

    batch_median_ms = float(
        np.percentile(
            times,
            50,
        )
    )

    batch_p95_ms = float(
        np.percentile(
            times,
            95,
        )
    )

    batch_min_ms = float(
        np.min(times)
    )

    batch_max_ms = float(
        np.max(times)
    )

    # 单张图片平均延迟。
    latency_per_image_ms = (
        batch_mean_ms
        / batch_size
    )

    median_per_image_ms = (
        batch_median_ms
        / batch_size
    )

    p95_per_image_ms = (
        batch_p95_ms
        / batch_size
    )

    # 每秒处理图片数量。
    throughput_fps = (
        batch_size
        * 1000.0
        / batch_mean_ms
    )

    del input_tensor

    clear_memory()

    return {
        "batch_mean_ms": (
            batch_mean_ms
        ),
        "batch_std_ms": (
            batch_std_ms
        ),
        "batch_median_ms": (
            batch_median_ms
        ),
        "batch_p95_ms": (
            batch_p95_ms
        ),
        "batch_min_ms": (
            batch_min_ms
        ),
        "batch_max_ms": (
            batch_max_ms
        ),
        "latency_per_image_ms": float(
            latency_per_image_ms
        ),
        "median_per_image_ms": float(
            median_per_image_ms
        ),
        "p95_per_image_ms": float(
            p95_per_image_ms
        ),
        "throughput_fps": float(
            throughput_fps
        ),
        "peak_memory_mb": float(
            peak_memory_mb
        ),
    }


# =========================================================
# 11. CSV保存
# =========================================================

def save_csv(
    rows: List[dict],
    csv_path: Path,
) -> None:
    if not rows:
        raise ValueError(
            "没有速度测试结果，无法保存CSV。"
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
        f"[SAVE] {csv_path}"
    )


# =========================================================
# 12. 获取设备信息
# =========================================================

def get_device_name() -> str:
    if DEVICE.type == "cuda":
        return torch.cuda.get_device_name(
            DEVICE
        )

    return "CPU"


def get_cudnn_version():
    if torch.backends.cudnn.is_available():
        return torch.backends.cudnn.version()

    return "not_available"


# =========================================================
# 13. main
# =========================================================

def main() -> None:
    set_seed(SEED)
    clear_memory()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device_name = get_device_name()

    print("=" * 100)
    print(f"[DEVICE]          {DEVICE}")
    print(f"[DEVICE NAME]     {device_name}")
    print(f"[TORCH VERSION]   {torch.__version__}")
    print(f"[CUDA VERSION]    {torch.version.cuda}")
    print(f"[CUDNN VERSION]   {get_cudnn_version()}")
    print(f"[INPUT SIZE]      3×{IMG_SIZE}×{IMG_SIZE}")
    print(f"[PRECISION]       FP32")
    print(f"[WARMUP STEPS]    {WARMUP_STEPS}")
    print(f"[MEASURE STEPS]   {MEASURE_STEPS}")
    print(f"[BATCH SIZES]     {BATCH_SIZES}")
    print("=" * 100)

    all_results: List[dict] = []

    for model_name, config in MODELS.items():
        model_path = str(
            config["path"]
        )

        backbone = str(
            config["backbone"]
        )

        print("\n" + "=" * 100)
        print(f"[START MODEL] {model_name}")
        print(f"[BACKBONE]    {backbone}")
        print(f"[CHECKPOINT]  {model_path}")
        print("=" * 100)

        checkpoint_path = Path(
            model_path
        )

        if not checkpoint_path.exists():
            print(
                "[ERROR] checkpoint不存在，"
                f"跳过：{checkpoint_path}"
            )
            continue

        model_info = build_model(
            model_path=model_path,
            backbone=backbone,
            device=DEVICE,
        )

        model = model_info[
            "model"
        ]

        total_parameters, trainable_parameters = (
            count_parameters(
                model
            )
        )

        checkpoint_size_mb = (
            checkpoint_path.stat().st_size
            / 1024.0
            / 1024.0
        )

        print(
            f"[PARAMETERS] total="
            f"{total_parameters:,} "
            f"({total_parameters / 1e6:.3f} M)"
        )

        print(
            f"[CHECKPOINT SIZE] "
            f"{checkpoint_size_mb:.3f} MB"
        )

        for batch_size in BATCH_SIZES:
            print("-" * 90)
            print(
                f"[BENCHMARK] model="
                f"{model_name}, "
                f"batch_size="
                f"{batch_size}"
            )

            speed_metrics = (
                benchmark_batch_size(
                    model=model,
                    batch_size=batch_size,
                    img_size=IMG_SIZE,
                    device=DEVICE,
                    warmup_steps=(
                        WARMUP_STEPS
                    ),
                    measure_steps=(
                        MEASURE_STEPS
                    ),
                )
            )

            result = {
                "model": model_name,
                "backbone": backbone,
                "model_path": model_path,
                "checkpoint_epoch": (
                    model_info[
                        "checkpoint_epoch"
                    ]
                ),
                "device": str(
                    DEVICE
                ),
                "device_name": (
                    device_name
                ),
                "torch_version": (
                    torch.__version__
                ),
                "cuda_version": (
                    torch.version.cuda
                ),
                "cudnn_version": (
                    get_cudnn_version()
                ),
                "precision": "FP32",
                "forward_method": (
                    "forward_emb"
                ),
                "preprocessing_included": (
                    False
                ),
                "input_shape": (
                    f"{batch_size}"
                    f"x3x"
                    f"{IMG_SIZE}"
                    f"x{IMG_SIZE}"
                ),
                "batch_size": (
                    batch_size
                ),
                "warmup_steps": (
                    WARMUP_STEPS
                ),
                "measure_steps": (
                    MEASURE_STEPS
                ),
                "num_classes": (
                    model_info[
                        "num_classes"
                    ]
                ),
                "embedding_dim": (
                    model_info[
                        "embedding_dim"
                    ]
                ),
                "total_parameters": (
                    total_parameters
                ),
                "parameters_million": (
                    total_parameters
                    / 1e6
                ),
                "trainable_parameters": (
                    trainable_parameters
                ),
                "checkpoint_size_mb": (
                    checkpoint_size_mb
                ),
                **speed_metrics,
            }

            all_results.append(
                result
            )

            print(
                f"[RESULT] "
                f"batch_mean="
                f"{speed_metrics['batch_mean_ms']:.4f} ms"
            )

            print(
                f"[RESULT] "
                f"single_image_latency="
                f"{speed_metrics['latency_per_image_ms']:.4f} ms"
            )

            print(
                f"[RESULT] "
                f"single_image_p95="
                f"{speed_metrics['p95_per_image_ms']:.4f} ms"
            )

            print(
                f"[RESULT] "
                f"throughput="
                f"{speed_metrics['throughput_fps']:.2f} images/s"
            )

            print(
                f"[RESULT] "
                f"peak_memory="
                f"{speed_metrics['peak_memory_mb']:.2f} MB"
            )

            # 每完成一个batch size就保存，
            # 防止测试中断后结果丢失。
            save_csv(
                rows=all_results,
                csv_path=OUTPUT_CSV,
            )

        del model
        del model_info

        clear_memory()

    if not all_results:
        raise RuntimeError(
            "没有任何模型完成速度测试，"
            "请检查MODELS中的checkpoint路径。"
        )

    save_csv(
        rows=all_results,
        csv_path=OUTPUT_CSV,
    )

    print("\n" + "=" * 100)
    print("[ALL MODELS FINISHED]")
    print(f"[OUTPUT CSV] {OUTPUT_CSV}")
    print("=" * 100)

    # 单独打印batch=1结果，方便论文制表。
    print("\n[BATCH SIZE = 1 RESULTS]")

    for row in all_results:
        if row["batch_size"] != 1:
            continue

        print({
            "model": row["model"],
            "parameters_million": (
                row[
                    "parameters_million"
                ]
            ),
            "latency_ms": (
                row[
                    "latency_per_image_ms"
                ]
            ),
            "p95_latency_ms": (
                row[
                    "p95_per_image_ms"
                ]
            ),
            "fps": (
                row[
                    "throughput_fps"
                ]
            ),
            "peak_memory_mb": (
                row[
                    "peak_memory_mb"
                ]
            ),
        })


if __name__ == "__main__":
    main()