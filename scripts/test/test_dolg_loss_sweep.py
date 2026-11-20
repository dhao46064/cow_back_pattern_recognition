# -*- coding: utf-8 -*-
"""自动评估无红增强DOLG和已完成红增强DOLG的ori/hard开放集指标。"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


CURRENT_FILE = Path(__file__).resolve()


def find_project_root(start: Path) -> Path:
    for directory in (start.parent, *start.parents):
        if (directory / "src").is_dir():
            return directory
    raise FileNotFoundError(f"找不到包含src的项目根目录：{start}")


PROJECT_ROOT = find_project_root(CURRENT_FILE)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_SCRIPT = PROJECT_ROOT / "scripts" / "test" / "test_ablation.py"
NO_RED_ROOT = PROJECT_ROOT / "outputs" / "dolg_selected_loss_10ep"
RED_GRID_ROOT = PROJECT_ROOT / "outputs" / "dolg_red_loss_grid_10ep"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dolg_loss_evaluation"

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="评估DOLG无红/红增强模型的ori和hard开放集指标。"
    )
    parser.add_argument(
        "--dataset",
        choices=("ori", "hard", "both"),
        default="both",
    )
    parser.add_argument(
        "--models",
        choices=("no_red", "red", "all"),
        default="all",
        help="默认评估所有已经完成的无红和红增强模型。",
    )
    return parser.parse_args()


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "dolg_test_ablation_base",
        BASE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载基础评估脚本：{BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def discover_from_root(root: Path):
    models = {}
    if not root.is_dir():
        return models
    for checkpoint in sorted(root.glob("*/checkpoints/epoch_10.pth")):
        experiment_name = checkpoint.parents[1].name
        models[experiment_name] = {
            "path": str(checkpoint),
            "backbone": "dolg_resnet50",
        }
    return models


def discover_models(selection: str):
    models = {}
    if selection in {"no_red", "all"}:
        models.update(discover_from_root(NO_RED_ROOT))
    if selection in {"red", "all"}:
        models.update(discover_from_root(RED_GRID_ROOT))
    if not models:
        raise FileNotFoundError("没有发现任何DOLG epoch_10.pth。")

    print(f"[DISCOVERED] DOLG models={len(models)}")
    for name, config in models.items():
        print(f"  {name}: {config['path']}")
    return models


def main() -> None:
    args = parse_args()
    base = load_base_module()
    from src.models.dolg_arcface_model import DOLGArcFaceCowModel

    def build_dolg_from_checkpoint(model_path: str, backbone: str, device: str):
        checkpoint, state_dict = base.load_checkpoint(model_path)
        if "arcface.weight" not in state_dict:
            raise KeyError("checkpoint中没有arcface.weight。")

        arcface_weight = state_dict["arcface.weight"]
        num_classes = int(arcface_weight.shape[0])
        embedding_dim = int(arcface_weight.shape[1])
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

        model = DOLGArcFaceCowModel(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            local_dim=int(run_config.get("local_dim", 512)),
            fusion_hidden_dim=int(run_config.get("fusion_hidden_dim", 512)),
            s=float(run_config.get("arc_s", 30.0)),
            m=float(run_config.get("arc_m", 0.25)),
            pretrained=False,
            freeze_backbone=False,
            gem_p=float(run_config.get("gem_p", 3.0)),
            gem_learn_p=not bool(run_config.get("fixed_gem_p", False)),
            dilation_rates=tuple(run_config.get("dilation_rates", (1, 2, 3))),
            dropout=float(run_config.get("dropout", 0.3)),
        )
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()

        print("=" * 90)
        print(f"[MODEL] {Path(model_path).parents[1].name}")
        print(f"[CHECKPOINT EPOCH] {checkpoint_epoch}")
        print(f"[CLASSES] {num_classes} | [EMBEDDING] {embedding_dim}")
        print("[LOAD] strict=True, missing=0, unexpected=0")
        print("=" * 90)
        return model, checkpoint_epoch

    base.MODELS = discover_models(args.models)
    base.build_model_from_ckpt = build_dolg_from_checkpoint

    selected_datasets = (
        ("ori", "hard")
        if args.dataset == "both"
        else (args.dataset,)
    )
    for dataset_name in selected_datasets:
        config = DATASETS[dataset_name]
        for config_name, path in config.items():
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{dataset_name}.{config_name}不存在：{path}"
                )

        output_dir = OUTPUT_ROOT / f"openset_{dataset_name}"
        base.SPLIT_JSON = str(config["split_json"])
        base.ORI_ROOT = str(config["ori_root"])
        base.GT_ROOT = str(config["gt_root"])
        base.OUTPUT_DIR = output_dir
        base.SUMMARY_CSV = (
            output_dir / f"dolg_openset_{dataset_name}_metrics.csv"
        )

        print("\n" + "#" * 100)
        print(f"[DATASET START] {dataset_name}")
        print("#" * 100)
        base.main()


if __name__ == "__main__":
    main()
