# -*- coding: utf-8 -*-
r"""
通用训练日志模块。

日志：
    D:\code_project\cow_recognition\logs\train_logs\<run_name>

checkpoint：
    D:\code_project\cow_recognition\outputs\<run_name>\checkpoints
"""

import json
import logging
import platform
import sys

from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def make_json_serializable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Namespace):
        return {
            key: make_json_serializable(item)
            for key, item in vars(value).items()
        }

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if isinstance(value, np.generic):
        return value.item()

    if torch.is_tensor(value):
        value = value.detach().cpu()

        if value.numel() == 1:
            return value.item()

        return value.tolist()

    return str(value)


def create_training_run(
    log_root: str,
    output_root: str,
    exp_name: str,
    log_level: int = logging.INFO,
) -> Tuple[logging.Logger, Path, Path, str]:
    """
    创建一次独立训练运行。
    """
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]

    run_name = f"{exp_name}_{timestamp}"

    log_run_dir = Path(log_root) / run_name
    output_run_dir = Path(output_root) / run_name

    log_run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    output_run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    log_path = log_run_dir / "train.log"

    logger = logging.getLogger(
        f"training.{run_name}"
    )

    logger.setLevel(log_level)
    logger.propagate = False

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        filename=log_path,
        mode="w",
        encoding="utf-8",
    )

    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(
        sys.stdout
    )

    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 100)
    logger.info("训练运行目录创建完成")
    logger.info(f"run_name       = {run_name}")
    logger.info(f"log_run_dir    = {log_run_dir}")
    logger.info(f"output_run_dir = {output_run_dir}")
    logger.info(f"train_log      = {log_path}")
    logger.info("=" * 100)

    return (
        logger,
        log_run_dir,
        output_run_dir,
        run_name,
    )


def get_environment_config() -> Dict[str, Any]:
    gpu_names = []

    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(
                torch.cuda.device_count()
            )
        ]

    return {
        "record_time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "command": " ".join(sys.argv),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": gpu_names,
    }


def _log_config_section(
    logger: logging.Logger,
    section_name: str,
    values: Any,
) -> None:
    logger.info(f"[{section_name.upper()}]")

    if isinstance(values, dict):
        for key, value in values.items():
            logger.info(
                f"{key:<38} = {value}"
            )
    else:
        logger.info(str(values))

    logger.info("-" * 100)


def save_training_config(
    logger: logging.Logger,
    log_run_dir: Path,
    args: Optional[Namespace] = None,
    model_config: Optional[Dict[str, Any]] = None,
    data_config: Optional[Dict[str, Any]] = None,
    augmentation_config: Optional[Dict[str, Any]] = None,
    optimizer_config: Optional[Dict[str, Any]] = None,
    trainer_config: Optional[Dict[str, Any]] = None,
    resume_config: Optional[Dict[str, Any]] = None,
    extra_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    将所有配置写入train.log并保存run_config.json。
    """
    full_config = {
        "arguments": make_json_serializable(
            vars(args) if args else {}
        ),
        "model": make_json_serializable(
            model_config or {}
        ),
        "data": make_json_serializable(
            data_config or {}
        ),
        "augmentation": make_json_serializable(
            augmentation_config or {}
        ),
        "optimizer": make_json_serializable(
            optimizer_config or {}
        ),
        "trainer": make_json_serializable(
            trainer_config or {}
        ),
        "resume": make_json_serializable(
            resume_config or {}
        ),
        "environment": make_json_serializable(
            get_environment_config()
        ),
        "extra": make_json_serializable(
            extra_config or {}
        ),
    }

    config_path = (
        Path(log_run_dir)
        / "run_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            full_config,
            file,
            ensure_ascii=False,
            indent=4,
        )

    logger.info("=" * 100)
    logger.info("完整训练配置")
    logger.info("=" * 100)

    for section_name, section_values in full_config.items():
        _log_config_section(
            logger=logger,
            section_name=section_name,
            values=section_values,
        )

    logger.info(f"run_config.json = {config_path}")
    logger.info("=" * 100)

    return config_path


def log_metrics(
    logger: logging.Logger,
    stage: str,
    epoch: int,
    metrics: Dict[str, Any],
) -> None:
    """
    所有Trainer通用的epoch指标记录接口。
    """
    parts = []

    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(
                f"{key}={value:.6f}"
            )
        else:
            parts.append(
                f"{key}={value}"
            )

    logger.info(
        f"[{stage}] epoch={epoch} | "
        + " | ".join(parts)
    )


def close_training_logger(
    logger: logging.Logger,
) -> None:
    for handler in logger.handlers[:]:
        try:
            handler.flush()
            handler.close()
        finally:
            logger.removeHandler(handler)
