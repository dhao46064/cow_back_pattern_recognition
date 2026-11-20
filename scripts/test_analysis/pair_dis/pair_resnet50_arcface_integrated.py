# -*- coding: utf-8 -*-
"""补算ResNet50+ArcFace基线的六项整合距离指标。"""

import csv
import importlib.util
import sys
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()


def find_project_root(start: Path) -> Path:
    for directory in (start.parent, *start.parents):
        if (directory / "src").is_dir():
            return directory
    raise FileNotFoundError(f"找不到包含src的项目根目录：{start}")


PROJECT_ROOT = find_project_root(CURRENT_FILE)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INTEGRATED_SCRIPT = (
    PROJECT_ROOT / "scripts" / "test_analysis" / "pair_dis"
    / "pair_dolg_selected_loss.py"
)
RESNET_BASE_SCRIPT = (
    PROJECT_ROOT / "scripts" / "test_analysis" / "pair_dis"
    / "pair_dis_ablation.py"
)
CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "exp_ablation_ResNet50"
    / "checkpoints" / "epoch_10.pth"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "exp_ablation_ResNet50"
    / "pair_distance_integrated"
)
SUMMARY_CSV = OUTPUT_DIR / "resnet50_arcface_pair_distance_integrated.csv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_result(result: dict) -> None:
    row = {
        "model": "ResNet50 + ArcFace",
        "same_normal": result["same_normal_mean"],
        "diff_normal": result["diff_normal_mean"],
        "diff_same_ratio": result["diff_same_ratio"],
        "ori_normal_abnormal": result["ori_normal_abnormal_mean"],
        "hard_normal_abnormal": result["hard_normal_abnormal_mean"],
        "hard_na_same_ratio": result["hard_na_same_ratio"],
        # 计数列仅用于核对计算口径。
        "same_pair_count": result["same_normal_count"],
        "diff_pair_count": result["diff_normal_count"],
        "ori_na_pair_count": result["ori_normal_abnormal_count"],
        "hard_na_pair_count": result["hard_normal_abnormal_count"],
        "normal_cows": result["normal_cows"],
        "ori_matched_cows": result["ori_matched_cows"],
        "hard_matched_cows": result["hard_matched_cows"],
        "checkpoint": result["checkpoint"],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"ResNet50基线checkpoint不存在：{CHECKPOINT}")

    integrated = load_module("resnet50_integrated_base", INTEGRATED_SCRIPT)
    resnet_base = load_module("resnet50_ablation_loader", RESNET_BASE_SCRIPT)

    def build_resnet50_model(checkpoint_path: Path, logger):
        checkpoint, state_dict = resnet_base.load_checkpoint(str(checkpoint_path))
        arcface_weight = state_dict.get("arcface.weight")
        if arcface_weight is None:
            raise KeyError("checkpoint中没有arcface.weight")

        num_classes = int(arcface_weight.shape[0])
        embedding_dim = int(arcface_weight.shape[1])
        epoch = (
            checkpoint.get("epoch", "unknown")
            if isinstance(checkpoint, dict) else "unknown"
        )
        model, loaded_epoch = resnet_base.build_model(
            model_path=str(checkpoint_path),
            backbone="resnet50",
            device=integrated.DEVICE,
            logger=logger,
        )
        return model, loaded_epoch or epoch, num_classes, embedding_dim

    # 只替换模型加载函数，其余数据、采样和距离算法与DOLG表完全相同。
    integrated.build_dolg_model = build_resnet50_model
    integrated.OUTPUT_DIR = OUTPUT_DIR
    integrated.SUMMARY_CSV = SUMMARY_CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    distance_base = integrated.load_base_module()
    result = integrated.run_one_model(
        distance_base,
        "resnet50_arcface",
        CHECKPOINT,
    )
    save_result(result)

    print("\n六项指标：")
    print(f"Same normal      = {result['same_normal_mean']:.6f}")
    print(f"Diff normal      = {result['diff_normal_mean']:.6f}")
    print(f"Diff/Same        = {result['diff_same_ratio']:.6f}")
    print(f"Ori N-A          = {result['ori_normal_abnormal_mean']:.6f}")
    print(f"Hard N-A         = {result['hard_normal_abnormal_mean']:.6f}")
    print(f"Hard N-A/Same    = {result['hard_na_same_ratio']:.6f}")
    print(f"CSV              = {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
