# -*- coding: utf-8 -*-
"""计算9个已完成红增强DOLG模型的整合距离。"""

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

BASE_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "test_analysis"
    / "pair_dis"
    / "pair_dolg_selected_loss.py"
)
RED_ROOT = PROJECT_ROOT / "outputs" / "dolg_red_loss_grid_10ep"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "dolg_red_selected_evaluation"
    / "pair_distance_integrated"
)
SUMMARY_CSV = OUTPUT_DIR / "dolg_red_selected_pair_distance_integrated.csv"

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


def load_base_module():
    spec = importlib.util.spec_from_file_location("pair_dolg_red_selected_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载基础距离脚本：{BASE_SCRIPT}")
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
        models[experiment_name] = checkpoint
    return models


def save_summary(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base = load_base_module()
    models = discover_selected_models()

    # run_one_model内部仍保持：normal只提取一次，ori/hard abnormal分别计算。
    base.OUTPUT_DIR = OUTPUT_DIR
    base.SUMMARY_CSV = SUMMARY_CSV
    distance_base = base.load_base_module()
    rows = []

    print(f"[SELECTED] models={len(models)}")
    for index, (model_name, checkpoint_path) in enumerate(models.items(), 1):
        print(f"\n[{index}/{len(models)}] START {model_name}")
        row = base.run_one_model(
            distance_base,
            model_name,
            checkpoint_path,
        )
        rows.append(row)
        save_summary(rows)

    print(f"\n[FINISHED] {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
