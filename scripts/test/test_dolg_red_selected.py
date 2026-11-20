# -*- coding: utf-8 -*-
"""测试9个已完成的红增强DOLG模型。"""

import argparse
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

BASE_SCRIPT = PROJECT_ROOT / "scripts" / "test" / "test_dolg_loss_sweep.py"
RED_ROOT = PROJECT_ROOT / "outputs" / "dolg_red_loss_grid_10ep"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dolg_red_selected_evaluation"

SELECTED_MODELS = (
    ("dolg_redp005_arcface", RED_ROOT),
    ("dolg_redp005_center_lam001", RED_ROOT),
    ("dolg_redp005_supcon_lam01", RED_ROOT),
    ("dolg_redp010_arcface", RED_ROOT),
    ("dolg_redp010_center_lam001", RED_ROOT),
    ("dolg_redp010_supcon_lam01", RED_ROOT),
    ("dolg_redp015_arcface", RED_ROOT),
    ("dolg_redp015_center_lam001", RED_ROOT),
    ("dolg_redp015_supcon_lam01", RED_ROOT),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="评估p=0/0.05/0.10/0.15的选定DOLG模型。"
    )
    parser.add_argument(
        "--dataset",
        choices=("ori", "hard", "both"),
        default="both",
    )
    return parser.parse_args()


def load_base_module():
    spec = importlib.util.spec_from_file_location("dolg_red_selected_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载基础测试脚本：{BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def discover_selected_models():
    models = {}
    for experiment_name, root in SELECTED_MODELS:
        checkpoint = root / experiment_name / "checkpoints" / "epoch_10.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"选定模型缺少epoch_10.pth：{checkpoint}")
        models[experiment_name] = {
            "path": str(checkpoint),
            "backbone": "dolg_resnet50",
        }
    return models


def main() -> None:
    args = parse_args()
    base = load_base_module()
    models = discover_selected_models()

    # 复用已经验证过的DOLG模型加载与ori/hard测试流程，
    # 只替换模型清单和输出目录。
    base.discover_models = lambda _selection: models
    base.OUTPUT_ROOT = OUTPUT_ROOT
    sys.argv = [str(BASE_SCRIPT), "--dataset", args.dataset, "--models", "all"]

    print(f"[SELECTED] models={len(models)}")
    for name in models:
        print(f"  {name}")
    print(f"[OUTPUT] {OUTPUT_ROOT}")
    base.main()


if __name__ == "__main__":
    main()
