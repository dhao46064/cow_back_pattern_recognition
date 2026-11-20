# -*- coding: utf-8 -*-
"""遍历 ResNet50 loss-sweep checkpoint，计算类内/类间 Pair Distance。"""

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch


CURRENT_FILE = Path(__file__).resolve()


def find_project_root(start: Path) -> Path:
    for directory in (start.parent, *start.parents):
        if (directory / "src").is_dir():
            return directory
    raise FileNotFoundError(f"找不到包含 src 的项目根目录：{start}")


PROJECT_ROOT = find_project_root(CURRENT_FILE)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "test_analysis"
    / "pair_dis"
    / "pair_dis_ablation.py"
)
SWEEP_ROOT = PROJECT_ROOT / "outputs" / "resnet50_loss_sweep_10ep"
OUTPUT_DIR = SWEEP_ROOT / "pair_distance"

EXPERIMENT_ORDER = (
    "resnet50_arcface_center_lam0001",
    "resnet50_arcface_center_lam0005",
    "resnet50_arcface_center_lam001",
    "resnet50_arcface_supcon_lam005",
    "resnet50_arcface_supcon_lam01",
    "resnet50_arcface_supcon_lam02",
)


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "pair_dis_ablation_base",
        BASE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载基础脚本：{BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def discover_models() -> Dict[str, dict]:
    models: Dict[str, dict] = {}
    missing = []

    for experiment_name in EXPERIMENT_ORDER:
        checkpoint = (
            SWEEP_ROOT
            / experiment_name
            / "checkpoints"
            / "epoch_10.pth"
        )
        if checkpoint.is_file():
            models[experiment_name] = {
                "path": str(checkpoint),
                "backbone": "resnet50",
            }
        else:
            missing.append(experiment_name)

    if not models:
        raise FileNotFoundError(
            f"没有找到任何 epoch_10.pth：{SWEEP_ROOT}"
        )

    print(f"[DISCOVERED] 已完成模型：{len(models)}")
    for name in models:
        print(f"  [OK] {name}")
    for name in missing:
        print(f"  [WAIT] {name} 尚未生成 epoch_10.pth")

    return models


def infer_model_config(state_dict: dict) -> Tuple[int, int, Tuple[int, ...]]:
    arcface_weight = state_dict.get("arcface.weight")
    if arcface_weight is None:
        raise KeyError("checkpoint 中没有 arcface.weight。")

    num_classes = int(arcface_weight.shape[0])
    embedding_dim = int(arcface_weight.shape[1])

    linear_layers = []
    prefix = "embedder.head."
    for key, value in state_dict.items():
        if not key.startswith(prefix) or not key.endswith(".weight"):
            continue
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            continue
        layer_text = key[len(prefix):].split(".", 1)[0]
        if layer_text.isdigit():
            linear_layers.append((int(layer_text), int(value.shape[0])))

    linear_layers.sort()
    if not linear_layers:
        raise KeyError("无法从 embedder.head 推断映射头结构。")

    hidden_dims = tuple(size for _, size in linear_layers[:-1])
    return num_classes, embedding_dim, hidden_dims


def main() -> None:
    base = load_base_module()
    from src.models.arcface_model import ArcFaceCowModel

    def build_sweep_model(
        model_path: str,
        backbone: str,
        device: str,
        logger,
    ):
        checkpoint, state_dict = base.load_checkpoint(model_path)
        num_classes, embedding_dim, hidden_dims = infer_model_config(
            state_dict
        )
        run_config = (
            checkpoint.get("run_config", {})
            if isinstance(checkpoint, dict)
            else {}
        )
        checkpoint_epoch = (
            checkpoint.get("epoch", "unknown")
            if isinstance(checkpoint, dict)
            else "unknown"
        )

        model = ArcFaceCowModel(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            s=float(run_config.get("arc_s", 30.0)),
            m=float(run_config.get("arc_m", 0.25)),
            pretrained=False,
            freeze_backbone=False,
            hidden_dims=hidden_dims,
            dropout=0.3,
            use_gem=("embedder.pool.p" in state_dict),
        )
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()

        logger.log(f"model_path={model_path}")
        logger.log(f"backbone={backbone}")
        logger.log(f"checkpoint_epoch={checkpoint_epoch}")
        logger.log(f"classes={num_classes}")
        logger.log(f"embedding_dim={embedding_dim}")
        logger.log(f"hidden_dims={hidden_dims}")
        logger.log("strict_load=True")
        return model, checkpoint_epoch

    base.MODELS = discover_models()
    base.OUTPUT_DIR = OUTPUT_DIR
    base.SUMMARY_CSV = OUTPUT_DIR / "loss_sweep_pair_distance_summary.csv"
    base.build_model = build_sweep_model
    base.main()


if __name__ == "__main__":
    main()
