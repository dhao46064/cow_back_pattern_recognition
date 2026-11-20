# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Dict

import torch


# =========================================================
# 1. 基础工具
# =========================================================
def format_size(num_bytes: int) -> str:
    """将字节转换为易读单位。"""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.4f} {unit}"
        size /= 1024

    return f"{size:.4f} TB"


def torch_load_compatible(weight_path: str):
    """兼容不同PyTorch版本。"""
    try:
        return torch.load(
            weight_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            weight_path,
            map_location="cpu",
        )


# =========================================================
# 2. 从checkpoint中提取模型state_dict
# =========================================================
def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """
    兼容以下保存格式：

    torch.save(model.state_dict(), path)

    torch.save({
        "model": model.state_dict()
    }, path)

    torch.save({
        "model_state_dict": model.state_dict()
    }, path)

    torch.save({
        "state_dict": model.state_dict()
    }, path)
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "checkpoint不是字典，无法提取state_dict。"
        )

    possible_keys = [
        "model",
        "model_state_dict",
        "state_dict",
        "teacher",
        "student",
    ]

    for key in possible_keys:
        value = checkpoint.get(key)

        if isinstance(value, dict):
            tensor_count = sum(
                torch.is_tensor(item)
                for item in value.values()
            )

            if tensor_count > 0:
                print(
                    f"[INFO] 使用checkpoint中的键：{key}"
                )
                return value

    # checkpoint本身就是state_dict
    tensor_count = sum(
        torch.is_tensor(value)
        for value in checkpoint.values()
    )

    if tensor_count > 0:
        print("[INFO] checkpoint本身就是state_dict")
        return checkpoint

    raise ValueError(
        "checkpoint中没有找到模型state_dict。"
    )


# =========================================================
# 3. 权重统计
# =========================================================
def analyze_weight_file(weight_path: str):
    path = Path(weight_path)

    if not path.exists():
        raise FileNotFoundError(
            f"权重文件不存在：{path}"
        )

    checkpoint = torch_load_compatible(
        str(path)
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    total_numel = 0
    total_tensor_bytes = 0
    tensor_count = 0

    dtype_statistics = {}

    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue

        tensor_count += 1

        numel = tensor.numel()
        tensor_bytes = (
            numel * tensor.element_size()
        )

        total_numel += numel
        total_tensor_bytes += tensor_bytes

        dtype_name = str(tensor.dtype)

        if dtype_name not in dtype_statistics:
            dtype_statistics[dtype_name] = {
                "tensor_count": 0,
                "numel": 0,
                "bytes": 0,
            }

        dtype_statistics[dtype_name][
            "tensor_count"
        ] += 1

        dtype_statistics[dtype_name][
            "numel"
        ] += numel

        dtype_statistics[dtype_name][
            "bytes"
        ] += tensor_bytes

    file_bytes = path.stat().st_size

    # 理论量化大小
    fp32_bytes = total_numel * 4
    fp16_bytes = total_numel * 2
    int8_bytes = total_numel * 1

    print("\n" + "=" * 80)
    print("模型权重大小统计")
    print("=" * 80)

    print(f"权重路径：{path}")
    print(f"张量数量：{tensor_count:,}")
    print(f"张量元素总数：{total_numel:,}")

    print("\n实际大小")
    print(
        f"整个checkpoint文件："
        f"{format_size(file_bytes)}"
    )
    print(
        f"模型state_dict张量："
        f"{format_size(total_tensor_bytes)}"
    )

    extra_bytes = file_bytes - total_tensor_bytes

    print(
        f"checkpoint额外内容："
        f"{format_size(max(extra_bytes, 0))}"
    )

    print("\n按统一数据类型估算")
    print(
        f"FP32理论大小："
        f"{format_size(fp32_bytes)}"
    )
    print(
        f"FP16理论大小："
        f"{format_size(fp16_bytes)}"
    )
    print(
        f"INT8理论大小："
        f"{format_size(int8_bytes)}"
    )

    print("\n当前state_dict数据类型")
    for dtype_name, info in dtype_statistics.items():
        print(
            f"{dtype_name:<18} | "
            f"张量={info['tensor_count']:,} | "
            f"元素={info['numel']:,} | "
            f"大小={format_size(info['bytes'])}"
        )

    print("=" * 80)

    return {
        "weight_path": str(path),
        "checkpoint_file_bytes": file_bytes,
        "state_dict_tensor_bytes": total_tensor_bytes,
        "total_numel": total_numel,
        "tensor_count": tensor_count,
        "fp32_bytes": fp32_bytes,
        "fp16_bytes": fp16_bytes,
        "int8_bytes": int8_bytes,
        "dtype_statistics": dtype_statistics,
    }


# =========================================================
# 4. main：只修改权重路径
# =========================================================
def main():
    weight_path = (
        r"D:\code_project\triplet_code"
        r"\logs\exp006_arcface_supcon"
        r"\checkpoints\latest.pth"
    )

    analyze_weight_file(
        weight_path=weight_path
    )


if __name__ == "__main__":
    main()