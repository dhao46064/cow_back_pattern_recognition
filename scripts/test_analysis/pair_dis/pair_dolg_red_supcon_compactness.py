# -*- coding: utf-8 -*-
"""计算红色增强0.05/0.10/0.15的DOLG+ArcFace+SupCon整合距离。"""

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
    PROJECT_ROOT / "scripts" / "test_analysis" / "pair_dis"
    / "pair_dolg_selected_loss.py"
)
MODEL_ROOT = PROJECT_ROOT / "outputs" / "dolg_red_loss_grid_10ep"
OUTPUT_DIR = MODEL_ROOT / "supcon_distance_integrated"
SUMMARY_CSV = OUTPUT_DIR / "dolg_red_supcon_distance_integrated.csv"

SELECTED_MODELS = (
    (0.05, "dolg_redp005_supcon_lam01"),
    (0.10, "dolg_redp010_supcon_lam01"),
    (0.15, "dolg_redp015_supcon_lam01"),
)


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "pair_dolg_red_supcon_integrated_base", BASE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载基础脚本：{BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def discover_models():
    models = []
    for red_probability, model_name in SELECTED_MODELS:
        checkpoint = MODEL_ROOT / model_name / "checkpoints" / "epoch_10.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"缺少模型checkpoint：{checkpoint}")
        models.append((red_probability, model_name, checkpoint))
    return models


def to_paper_row(red_probability: float, result: dict) -> dict:
    return {
        "red_probability": red_probability,
        "model": result["model"],
        "same_normal": result["same_normal_mean"],
        "diff_normal": result["diff_normal_mean"],
        "diff_same_ratio": result["diff_same_ratio"],
        "ori_normal_abnormal": result["ori_normal_abnormal_mean"],
        "hard_normal_abnormal": result["hard_normal_abnormal_mean"],
        "hard_na_same_ratio": result["hard_na_same_ratio"],
        # 以下计数只用于核对统计口径。
        "same_pair_count": result["same_normal_count"],
        "diff_pair_count": result["diff_normal_count"],
        "ori_na_pair_count": result["ori_normal_abnormal_count"],
        "hard_na_pair_count": result["hard_normal_abnormal_count"],
        "normal_cows": result["normal_cows"],
        "ori_matched_cows": result["ori_matched_cows"],
        "hard_matched_cows": result["hard_matched_cows"],
        "checkpoint": result["checkpoint"],
    }


def save_rows(rows):
    if not rows:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not BASE_SCRIPT.is_file():
        raise FileNotFoundError(f"基础脚本不存在：{BASE_SCRIPT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_module()
    distance_base = base.load_base_module()
    models = discover_models()
    rows = []

    print("=" * 90)
    print("DOLG+ArcFace+SupCon红色增强整合距离")
    print("概率：0.05、0.10、0.15")
    print("Same | Diff | Diff/Same | Ori N-A | Hard N-A | Hard N-A/Same")
    print(f"output={SUMMARY_CSV}")
    print("=" * 90)

    for index, (red_probability, model_name, checkpoint) in enumerate(models, 1):
        print(f"\n[{index}/{len(models)}] START {model_name}")
        full_result = base.run_one_model(
            distance_base, model_name, checkpoint
        )
        rows.append(to_paper_row(red_probability, full_result))
        save_rows(rows)

    print(f"\n全部完成：{SUMMARY_CSV}")


if __name__ == "__main__":
    main()
