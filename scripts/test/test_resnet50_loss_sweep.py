# -*- coding: utf-8 -*-
"""遍历 ResNet50 loss-sweep checkpoint，评估 ori/hard 开放集指标。"""

import argparse
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

BASE_SCRIPT = PROJECT_ROOT / "scripts" / "test" / "test_ablation.py"
SWEEP_ROOT = PROJECT_ROOT / "outputs" / "resnet50_loss_sweep_10ep"

ORI_SPLIT_JSON = Path(
    r"D:\datasets\cow_recognition8.3\openset\openset_ori_8.2\split.json"
)

DATASETS = {
    "ori": {
        "split_json": ORI_SPLIT_JSON,
        "ori_root": Path(r"D:\code_project\triplet_code\data\dataset_obb_ori"),
        "gt_root": Path(
            r"D:\datasets\cow_recognition8.3\openset\openset_ori_8.2"
        ),
    },
    "hard": {
        "split_json": ORI_SPLIT_JSON,
        "ori_root": Path(
            r"D:\datasets\cow_recognition8.3\openset"
            r"\openset_consensus_hard_11to1_v2_mixed"
        ),
        "gt_root": Path(
            r"D:\datasets\cow_recognition8.3\openset"
            r"\openset_consensus_hard_11to1_v2"
        ),
    },
}

EXPERIMENT_ORDER = (
    "resnet50_arcface_center_lam0001",
    "resnet50_arcface_center_lam0005",
    "resnet50_arcface_center_lam001",
    "resnet50_arcface_supcon_lam005",
    "resnet50_arcface_supcon_lam01",
    "resnet50_arcface_supcon_lam02",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="遍历 loss-sweep 模型并评估开放集 ori/hard 指标。"
    )
    parser.add_argument(
        "--dataset",
        choices=("ori", "hard", "both"),
        default="both",
        help="默认依次评估 ori 和 hard。",
    )
    return parser.parse_args()


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "test_ablation_base",
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
    args = parse_args()
    base = load_base_module()
    from src.models.arcface_model import ArcFaceCowModel

    def build_sweep_model(
        model_path: str,
        backbone: str,
        device: str,
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

        print("=" * 90)
        print(f"[MODEL PATH]       {model_path}")
        print(f"[BACKBONE]         {backbone}")
        print(f"[CHECKPOINT EPOCH] {checkpoint_epoch}")
        print(f"[NUM CLASSES]      {num_classes}")
        print(f"[EMBEDDING DIM]    {embedding_dim}")
        print(f"[HIDDEN DIMS]      {hidden_dims}")
        print("=" * 90)

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
        print("[LOAD] strict=True, missing=0, unexpected=0")
        return model, checkpoint_epoch

    base.MODELS = discover_models()
    base.build_model_from_ckpt = build_sweep_model

    selected = ("ori", "hard") if args.dataset == "both" else (args.dataset,)
    for dataset_name in selected:
        config = DATASETS[dataset_name]
        for path_name, path in config.items():
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{dataset_name}.{path_name} 不存在：{path}"
                )

        output_dir = SWEEP_ROOT / f"openset_{dataset_name}"
        base.SPLIT_JSON = str(config["split_json"])
        base.ORI_ROOT = str(config["ori_root"])
        base.GT_ROOT = str(config["gt_root"])
        base.OUTPUT_DIR = output_dir
        base.SUMMARY_CSV = (
            output_dir / f"loss_sweep_openset_{dataset_name}_metrics.csv"
        )

        print("\n" + "#" * 100)
        print(f"[DATASET START] {dataset_name}")
        print("#" * 100)
        base.main()


if __name__ == "__main__":
    main()
