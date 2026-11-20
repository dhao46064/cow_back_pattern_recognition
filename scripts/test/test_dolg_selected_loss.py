# -*- coding: utf-8 -*-
"""批量测试三组无红增强 DOLG 模型的 ori/hard 开放集指标。"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict


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
MODEL_ROOT = PROJECT_ROOT / "outputs" / "dolg_selected_loss_10ep"

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
    "dolg_arcface",
    "dolg_arcface_center_lam001",
    "dolg_arcface_supcon_lam01",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="测试三组无红增强 DOLG 模型的开放集指标。"
    )
    parser.add_argument(
        "--dataset",
        choices=("ori", "hard", "both"),
        default="both",
        help="默认依次测试 ori 和 hard。",
    )
    return parser.parse_args()


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "test_dolg_selected_base",
        BASE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载开放集基础测试脚本：{BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def discover_models() -> Dict[str, dict]:
    models: Dict[str, dict] = {}
    missing = []

    for experiment_name in EXPERIMENT_ORDER:
        checkpoint = (
            MODEL_ROOT
            / experiment_name
            / "checkpoints"
            / "epoch_10.pth"
        )
        if checkpoint.is_file():
            models[experiment_name] = {
                "path": str(checkpoint),
                "backbone": "dolg_resnet50",
            }
        else:
            missing.append(experiment_name)

    print(f"[DISCOVERED] 已完成模型：{len(models)}/{len(EXPERIMENT_ORDER)}")
    for name in models:
        print(f"  [OK] {name}")
    for name in missing:
        print(f"  [WAIT] {name} 尚未生成 epoch_10.pth")

    if not models:
        raise FileNotFoundError(f"没有找到任何 DOLG epoch_10.pth：{MODEL_ROOT}")
    return models


def main() -> None:
    args = parse_args()
    base = load_base_module()
    from src.models.dolg_arcface_model import DOLGArcFaceCowModel

    def build_dolg_from_checkpoint(model_path: str, backbone: str, device: str):
        checkpoint, state_dict = base.load_checkpoint(model_path)
        arcface_weight = state_dict.get("arcface.weight")
        if arcface_weight is None:
            raise KeyError("checkpoint 中没有 arcface.weight。")

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
        print(f"[MODEL]            {Path(model_path).parents[1].name}")
        print(f"[CHECKPOINT EPOCH] {checkpoint_epoch}")
        print(f"[CLASSES]          {num_classes}")
        print(f"[EMBEDDING]        {embedding_dim}")
        print("[LOAD]             strict=True, missing=0, unexpected=0")
        print("=" * 90)
        return model, checkpoint_epoch

    base.MODELS = discover_models()
    base.build_model_from_ckpt = build_dolg_from_checkpoint

    selected = ("ori", "hard") if args.dataset == "both" else (args.dataset,)
    for dataset_name in selected:
        config = DATASETS[dataset_name]
        for config_name, path in config.items():
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{dataset_name}.{config_name} 不存在：{path}"
                )

        output_dir = MODEL_ROOT / f"openset_{dataset_name}"
        base.SPLIT_JSON = str(config["split_json"])
        base.ORI_ROOT = str(config["ori_root"])
        base.GT_ROOT = str(config["gt_root"])
        base.OUTPUT_DIR = output_dir
        base.SUMMARY_CSV = (
            output_dir / f"dolg_selected_openset_{dataset_name}_metrics.csv"
        )

        print("\n" + "#" * 100)
        print(f"[DATASET START] {dataset_name}")
        print("#" * 100)
        base.main()


if __name__ == "__main__":
    main()
