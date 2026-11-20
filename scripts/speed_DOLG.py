# -*- coding: utf-8 -*-
"""测试 DOLG-ArcFace 的参数量与单图推理速度。

默认口径与 backbone 消融速度实验保持一致：
- 输入：1 x 3 x 224 x 224
- 精度：FP32
- 前向：model.forward_emb
- 预热：50 次
- 正式计时：200 次
- 不包含图片读取和预处理
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch


DEFAULT_PROJECT_ROOT = Path(r"D:\code_project\triplet_code")
DEFAULT_CHECKPOINT = (
    DEFAULT_PROJECT_ROOT
    / "logs"
    / "exp003_7.25_arcface_DOLG"
    / "checkpoints"
    / "latest.pth"
)
DEFAULT_OUTPUT = (
    DEFAULT_PROJECT_ROOT
    / "logs"
    / "exp003_7.25_arcface_DOLG"
    / "dolg_inference_speed_summary.csv"
)

# 已有 ResNet50 + ArcFace 速度表中的参数量，用于计算 DOLG 新增参数。
RESNET50_BASELINE_PARAMETERS = 25_746_624


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测试 DOLG 参数量与 batch=1 单图推理速度。"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--measure-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dilation-rates",
        type=int,
        nargs=3,
        default=(1, 2, 3),
    )
    return parser.parse_args()


def add_project_root(project_root: Path) -> None:
    project_root = project_root.resolve()
    if not (project_root / "src").is_dir():
        raise FileNotFoundError(
            f"项目根目录下不存在 src：{project_root}"
        )
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def torch_load_compatible(path: Path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]):
    normalized = {}
    prefixes = ("module.", "_orig_mod.")

    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        normalized[new_key] = value

    return normalized


def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return normalize_state_dict_keys(value)

        if checkpoint and all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            return normalize_state_dict_keys(checkpoint)

    raise TypeError("checkpoint 中没有找到有效的模型 state_dict。")


def infer_dolg_config(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    required_keys = (
        "arcface.weight",
        "global_projection.0.weight",
        "embedding_head.0.weight",
    )
    missing = [key for key in required_keys if key not in state_dict]
    if missing:
        raise KeyError(
            "checkpoint 缺少 DOLG 关键参数：" + ", ".join(missing)
        )

    arcface_weight = state_dict["arcface.weight"]
    global_projection_weight = state_dict[
        "global_projection.0.weight"
    ]
    embedding_head_weight = state_dict[
        "embedding_head.0.weight"
    ]

    num_classes = int(arcface_weight.shape[0])
    embedding_dim = int(arcface_weight.shape[1])
    local_dim = int(global_projection_weight.shape[0])
    fusion_hidden_dim = int(embedding_head_weight.shape[0])

    if int(embedding_head_weight.shape[1]) != 2 * local_dim:
        raise ValueError(
            "embedding_head 与 local_dim 不匹配："
            f"input={embedding_head_weight.shape[1]}, "
            f"expected={2 * local_dim}"
        )

    return {
        "num_classes": num_classes,
        "embedding_dim": embedding_dim,
        "local_dim": local_dim,
        "fusion_hidden_dim": fusion_hidden_dim,
    }


def count_parameters(modules: Iterable[torch.nn.Module]) -> int:
    parameter_ids = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            parameter_id = id(parameter)
            if parameter_id not in parameter_ids:
                parameter_ids.add(parameter_id)
                total += parameter.numel()
    return total


def model_state_size_bytes(model: torch.nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.state_dict().values()
    )


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


@torch.inference_mode()
def benchmark_cuda(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    warmup_steps: int,
    measure_steps: int,
):
    for _ in range(warmup_steps):
        model.forward_emb(input_tensor)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    timings_ms = []

    for _ in range(measure_steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model.forward_emb(input_tensor)
        end.record()
        torch.cuda.synchronize()
        timings_ms.append(float(start.elapsed_time(end)))

    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    return timings_ms, peak_memory_mb


@torch.inference_mode()
def benchmark_cpu(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    warmup_steps: int,
    measure_steps: int,
):
    for _ in range(warmup_steps):
        model.forward_emb(input_tensor)

    timings_ms = []
    for _ in range(measure_steps):
        start = time.perf_counter()
        model.forward_emb(input_tensor)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    return timings_ms, 0.0


def save_csv(row: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    output_path = args.output_csv.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{model_path}")
    if args.warmup_steps < 1 or args.measure_steps < 2:
        raise ValueError("warmup_steps 至少为1，measure_steps 至少为2。")

    add_project_root(args.project_root)
    from src.models.dolg_arcface_model import DOLGArcFaceCowModel

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = torch_load_compatible(model_path)
    state_dict = extract_state_dict(checkpoint)
    config = infer_dolg_config(state_dict)

    model = DOLGArcFaceCowModel(
        num_classes=config["num_classes"],
        embedding_dim=config["embedding_dim"],
        local_dim=config["local_dim"],
        fusion_hidden_dim=config["fusion_hidden_dim"],
        pretrained=False,
        freeze_backbone=False,
        gem_p=3.0,
        gem_learn_p=True,
        dilation_rates=tuple(args.dilation_rates),
        dropout=0.3,
    )
    model.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    input_tensor = torch.randn(
        1,
        3,
        args.img_size,
        args.img_size,
        device=device,
        dtype=torch.float32,
    )

    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    backbone_parameters = count_parameters(
        [
            model.stem,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        ]
    )
    local_branch_parameters = count_parameters(
        [model.multi_atrous, model.local_attention]
    )
    global_branch_parameters = count_parameters(
        [model.global_pool, model.global_projection]
    )
    fusion_head_parameters = count_parameters(
        [model.orthogonal_fusion, model.embedding_head]
    )
    arcface_parameters = count_parameters([model.arcface])

    if device.type == "cuda":
        timings_ms, peak_memory_mb = benchmark_cuda(
            model,
            input_tensor,
            args.warmup_steps,
            args.measure_steps,
        )
        device_name = torch.cuda.get_device_name(device)
    else:
        timings_ms, peak_memory_mb = benchmark_cpu(
            model,
            input_tensor,
            args.warmup_steps,
            args.measure_steps,
        )
        device_name = "CPU"

    mean_ms = statistics.fmean(timings_ms)
    std_ms = statistics.pstdev(timings_ms)
    median_ms = statistics.median(timings_ms)
    p95_ms = percentile(timings_ms, 95)
    throughput_fps = 1000.0 / mean_ms
    checkpoint_epoch = (
        checkpoint.get("epoch", "unknown")
        if isinstance(checkpoint, dict)
        else "unknown"
    )

    row = {
        "model": "DOLG_ArcFace",
        "backbone": "resnet50",
        "model_path": str(model_path),
        "checkpoint_epoch": checkpoint_epoch,
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "precision": "FP32",
        "forward_method": "forward_emb",
        "preprocessing_included": False,
        "input_shape": f"1x3x{args.img_size}x{args.img_size}",
        "batch_size": 1,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "num_classes": config["num_classes"],
        "embedding_dim": config["embedding_dim"],
        "local_dim": config["local_dim"],
        "fusion_hidden_dim": config["fusion_hidden_dim"],
        "total_parameters": total_parameters,
        "parameters_million": total_parameters / 1_000_000,
        "trainable_parameters": trainable_parameters,
        "delta_vs_resnet50_parameters": (
            total_parameters - RESNET50_BASELINE_PARAMETERS
        ),
        "delta_vs_resnet50_million": (
            total_parameters - RESNET50_BASELINE_PARAMETERS
        )
        / 1_000_000,
        "backbone_parameters": backbone_parameters,
        "local_branch_parameters": local_branch_parameters,
        "global_branch_parameters": global_branch_parameters,
        "fusion_head_parameters": fusion_head_parameters,
        "arcface_parameters": arcface_parameters,
        "model_state_size_mb": model_state_size_bytes(model) / (1024**2),
        "checkpoint_size_mb": model_path.stat().st_size / (1024**2),
        "batch_mean_ms": mean_ms,
        "batch_std_ms": std_ms,
        "batch_median_ms": median_ms,
        "batch_p95_ms": p95_ms,
        "batch_min_ms": min(timings_ms),
        "batch_max_ms": max(timings_ms),
        "latency_per_image_ms": mean_ms,
        "throughput_fps": throughput_fps,
        "peak_memory_mb": peak_memory_mb,
    }

    save_csv(row, output_path)

    print("=" * 80)
    print(f"模型              : {row['model']}")
    print(f"checkpoint        : {model_path}")
    print(f"epoch             : {checkpoint_epoch}")
    print(f"设备              : {device_name}")
    print(f"参数量            : {row['parameters_million']:.6f} M")
    print(
        "较ResNet50新增     : "
        f"{row['delta_vs_resnet50_million']:.6f} M"
    )
    print(f"模型状态大小      : {row['model_state_size_mb']:.2f} MB")
    print(f"单图平均延迟      : {mean_ms:.4f} ms")
    print(f"单图中位延迟      : {median_ms:.4f} ms")
    print(f"单图P95延迟       : {p95_ms:.4f} ms")
    print(f"理论吞吐量        : {throughput_fps:.2f} images/s")
    if device.type == "cuda":
        print(f"峰值显存          : {peak_memory_mb:.2f} MB")
    print(f"CSV               : {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
