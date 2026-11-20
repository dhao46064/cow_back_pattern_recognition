# -*- coding: utf-8 -*-
r"""
牛背浅色区域在线红色增强。

原图与mask目录保持相同相对结构，例如：

原图：
    D:\datasets\cow_recognition8.3\train_dataset
    \200951\normal\abc.jpg

mask：
    D:\datasets\cow_recognition8.3\train_dataset_masks
    \200951\normal\abc.png

mask中：
    255：执行颜色增强
    0：保持原始像素
"""

import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image


class CowBackRedColorShift:
    """
    对牛背浅色区域进行在线红色域增强。

    完整强度strength=1.0时：
        R *= r_target
        G *= g_target
        B *= b_target

    中间强度采用线性插值：
        factor = 1 + strength * (target_factor - 1)
    """

    def __init__(
        self,
        image_root: str,
        mask_root: str,
        probability: float = 0.0,
        strengths: Sequence[float] = (0.4, 0.6, 0.8, 1.0),
        strength_probs: Sequence[float] = (0.35, 0.35, 0.20, 0.10),
        r_target: float = 0.78,
        g_target: float = 0.55,
        b_target: float = 0.50,
        missing_mask_policy: str = "error",
    ):
        self.image_root = Path(image_root).resolve()
        self.mask_root = Path(mask_root).resolve()

        self.probability = float(probability)
        self.strengths = tuple(float(x) for x in strengths)
        self.strength_probs = tuple(float(x) for x in strength_probs)

        self.r_target = float(r_target)
        self.g_target = float(g_target)
        self.b_target = float(b_target)

        self.missing_mask_policy = str(missing_mask_policy).lower()

        self._validate_arguments()

        self.total_calls = 0
        self.augmented_count = 0
        self.missing_mask_count = 0
        self.empty_mask_count = 0
        self.strength_counter = Counter()

    def _validate_arguments(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                f"probability必须位于[0, 1]，当前值为{self.probability}"
            )

        if len(self.strengths) == 0:
            raise ValueError("strengths不能为空。")

        if len(self.strengths) != len(self.strength_probs):
            raise ValueError(
                "strengths与strength_probs长度必须一致。"
            )

        if any(x < 0.0 or x > 1.0 for x in self.strengths):
            raise ValueError(
                f"所有增强强度必须位于[0, 1]，当前为{self.strengths}"
            )

        if any(x < 0.0 for x in self.strength_probs):
            raise ValueError("strength_probs不能包含负数。")

        probability_sum = float(sum(self.strength_probs))
        if probability_sum <= 0:
            raise ValueError("strength_probs之和必须大于0。")

        # 自动归一化，避免浮点和不严格等于1。
        self.strength_probs = tuple(
            x / probability_sum
            for x in self.strength_probs
        )

        if self.missing_mask_policy not in {"error", "original"}:
            raise ValueError(
                "missing_mask_policy只能是'error'或'original'。"
            )

        for name, value in {
            "r_target": self.r_target,
            "g_target": self.g_target,
            "b_target": self.b_target,
        }.items():
            if value <= 0:
                raise ValueError(f"{name}必须大于0，当前值为{value}")

    def build_mask_path(self, image_path: str) -> Path:
        """
        根据原图路径构建对应PNG mask路径。
        """
        image_path = Path(image_path).resolve()

        try:
            relative_path = image_path.relative_to(self.image_root)
        except ValueError:
            # 回退结构：cow_id/subset/filename
            # 适用于split.json中仍是其他根目录绝对路径、
            # 但目录尾部结构未变化的情况。
            if len(image_path.parents) < 2:
                raise ValueError(
                    f"无法根据图片路径推导mask路径：{image_path}"
                )

            relative_path = (
                Path(image_path.parent.parent.name)
                / image_path.parent.name
                / image_path.name
            )

        return (
            self.mask_root
            / relative_path.with_suffix(".png")
        )

    def find_missing_masks(
        self,
        image_paths: Iterable[str],
    ) -> List[Dict[str, str]]:
        """
        训练前检查所有图片是否存在对应mask。
        """
        missing = []

        for image_path in image_paths:
            mask_path = self.build_mask_path(image_path)

            if not mask_path.exists():
                missing.append({
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                })

        return missing

    @staticmethod
    def _read_mask(mask_path: Path) -> np.ndarray:
        """
        PIL读取单通道mask，避免中文路径问题。
        """
        with Image.open(mask_path) as mask_image:
            mask = np.asarray(
                mask_image.convert("L"),
                dtype=np.uint8,
            )

        return mask

    def _handle_missing_mask(
        self,
        image: Image.Image,
        mask_path: Path,
    ) -> Image.Image:
        self.missing_mask_count += 1

        if self.missing_mask_policy == "original":
            return image

        raise FileNotFoundError(
            f"找不到红色增强mask：{mask_path}"
        )

    def __call__(
        self,
        image: Image.Image,
        image_path: str,
    ) -> Image.Image:
        """
        输入和输出均为PIL RGB图像。
        """
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"image必须是PIL.Image，当前类型为{type(image)}"
            )

        self.total_calls += 1

        # 先判断概率。Baseline probability=0时完全不读取mask。
        if self.probability <= 0.0:
            return image

        if random.random() >= self.probability:
            return image

        mask_path = self.build_mask_path(image_path)

        if not mask_path.exists():
            return self._handle_missing_mask(
                image=image,
                mask_path=mask_path,
            )

        mask = self._read_mask(mask_path)

        image_array = np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )

        if mask.shape != image_array.shape[:2]:
            mask = np.asarray(
                Image.fromarray(mask).resize(
                    (image_array.shape[1], image_array.shape[0]),
                    resample=Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )

        selected = mask > 0

        if not np.any(selected):
            self.empty_mask_count += 1
            return image

        strength = random.choices(
            population=self.strengths,
            weights=self.strength_probs,
            k=1,
        )[0]

        r_factor = 1.0 + strength * (self.r_target - 1.0)
        g_factor = 1.0 + strength * (self.g_target - 1.0)
        b_factor = 1.0 + strength * (self.b_target - 1.0)

        output = image_array.astype(np.float32).copy()

        # PIL/numpy为RGB顺序。
        output[..., 0][selected] *= r_factor
        output[..., 1][selected] *= g_factor
        output[..., 2][selected] *= b_factor

        output = np.clip(
            output,
            0.0,
            255.0,
        ).astype(np.uint8)

        self.augmented_count += 1
        self.strength_counter[f"{strength:.2f}"] += 1

        return Image.fromarray(
            output,
            mode="RGB",
        )

    def get_statistics(
        self,
        reset: bool = False,
    ) -> Dict[str, object]:
        """
        返回在线增强统计。

        注意：
        num_workers>0时，每个worker有独立Dataset副本，
        主进程中读取到的计数不完整。默认训练使用num_workers=0。
        """
        actual_ratio = (
            self.augmented_count / self.total_calls
            if self.total_calls > 0
            else 0.0
        )

        statistics = {
            "configured_probability": self.probability,
            "total_calls": int(self.total_calls),
            "augmented_count": int(self.augmented_count),
            "actual_ratio": float(actual_ratio),
            "missing_mask_count": int(self.missing_mask_count),
            "empty_mask_count": int(self.empty_mask_count),
            "strength_counts": dict(self.strength_counter),
        }

        if reset:
            self.reset_statistics()

        return statistics

    def reset_statistics(self) -> None:
        self.total_calls = 0
        self.augmented_count = 0
        self.missing_mask_count = 0
        self.empty_mask_count = 0
        self.strength_counter.clear()

    def get_config(self) -> Dict[str, object]:
        return {
            "image_root": str(self.image_root),
            "mask_root": str(self.mask_root),
            "probability": self.probability,
            "strengths": list(self.strengths),
            "strength_probs": list(self.strength_probs),
            "r_target": self.r_target,
            "g_target": self.g_target,
            "b_target": self.b_target,
            "missing_mask_policy": self.missing_mask_policy,
        }
