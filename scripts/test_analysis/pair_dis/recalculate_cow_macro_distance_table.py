# -*- coding: utf-8 -*-
"""基于雨云图逐牛CSV，补算逐牛Diff并重建六模型Macro距离表。"""

import argparse
import gc
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

PAIR_DIR = PROJECT_ROOT / "scripts" / "test_analysis" / "pair_dis"
RAINCLOUD_SCRIPT = PAIR_DIR / "plot_compactness_raincloud.py"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compactness_raincloud"
SOURCE_CSV = OUTPUT_DIR / "compactness_per_cow.csv"
ENHANCED_CSV = OUTPUT_DIR / "compactness_per_cow_with_diff.csv"
TABLE_CSV = OUTPUT_DIR / "compactness_cow_macro_table.csv"

DIFF_PAIRS_PER_COW = 1000
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true",
        help="重新计算已经缓存的逐牛Diff。",
    )
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compute_diff_per_cow(normal_by_cow, pairs_per_cow=1000, seed=42):
    """每头牛作为anchor，与其他牛随机采样固定数量normal pair。"""
    cow_ids = sorted(normal_by_cow)
    if len(cow_ids) < 2:
        raise RuntimeError("有效牛只不足2头，无法计算Diff normal。")

    rows = []
    for cow_id in cow_ids:
        anchor = normal_by_cow[cow_id]
        other_ids = [item for item in cow_ids if item != cow_id]
        rng = random.Random(f"{seed}:{cow_id}:diff")

        anchor_vectors = []
        other_vectors = []
        for _ in range(pairs_per_cow):
            other_id = rng.choice(other_ids)
            other = normal_by_cow[other_id]
            anchor_vectors.append(anchor[rng.randrange(anchor.shape[0])])
            other_vectors.append(other[rng.randrange(other.shape[0])])

        anchor_matrix = torch.stack(anchor_vectors)
        other_matrix = torch.stack(other_vectors)
        distances = 1.0 - torch.sum(anchor_matrix * other_matrix, dim=1)
        distances = torch.clamp(distances, min=0.0, max=2.0).numpy()
        rows.append({
            "cow_id": str(cow_id),
            "diff_pair_count": int(distances.size),
            "diff_mean": float(np.mean(distances)),
            "diff_median": float(np.median(distances)),
            "diff_std": float(np.std(distances)),
        })
    return pd.DataFrame(rows)


def calculate_macro_table(frame, model_order):
    rows = []
    for model in model_order:
        group = frame[frame["model"] == model]
        same = float(group["same_mean"].mean())
        diff = float(group["diff_mean"].mean())
        ori_na = float(group["ori_na_mean"].mean())
        hard_na = float(group["hard_na_mean"].mean())
        rows.append({
            "Model": model,
            "Cow count": int(group["cow_id"].nunique()),
            "Same normal": same,
            "Diff normal": diff,
            "Diff/Same": diff / same,
            "Ori N-A": ori_na,
            "Hard N-A": hard_na,
            "Hard N-A/Same": hard_na / same,
            "Hard gap": hard_na - same,
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(
            f"找不到雨云图逐牛结果：{SOURCE_CSV}\n"
            "请先运行plot_compactness_raincloud.py。"
        )

    analysis = load_module("raincloud_analysis", RAINCLOUD_SCRIPT)
    integrated = analysis.load_module(
        "macro_integrated", analysis.INTEGRATED_SCRIPT
    )
    distance_base = integrated.load_base_module()
    resnet_ablation = analysis.load_module(
        "macro_resnet_ablation", analysis.RESNET_ABLATION_SCRIPT
    )
    resnet_sweep = analysis.load_module(
        "macro_resnet_sweep", analysis.RESNET_SWEEP_SCRIPT
    )

    source = pd.read_csv(SOURCE_CSV, dtype={"cow_id": str})
    merged_parts = []

    for index, config in enumerate(analysis.MODELS, 1):
        diff_cache = OUTPUT_DIR / f"{config['key']}_per_cow_diff.csv"
        if diff_cache.is_file() and not args.overwrite:
            print(f"[{index}/{len(analysis.MODELS)}] REUSE {diff_cache.name}")
            diff_frame = pd.read_csv(diff_cache, dtype={"cow_id": str})
        else:
            print(f"[{index}/{len(analysis.MODELS)}] COMPUTE {config['label']} Diff")
            logger = analysis.SimpleLogger(
                OUTPUT_DIR / f"{config['key']}_diff.log"
            )
            model = None
            try:
                distance_base.set_seed(SEED)
                distance_base.fresh_start()
                model, _ = analysis.build_model(
                    config, integrated, resnet_ablation, resnet_sweep, logger
                )
                from src.datasets.split import get_cow_ids
                cow_ids = [
                    str(item)
                    for item in get_cow_ids(analysis.ORI_ROOT, "unseen_cows")
                ]
                normal_by_cow = integrated.extract_by_cow(
                    distance_base,
                    model,
                    cow_ids,
                    analysis.ORI_ROOT,
                    "normal",
                    integrated.MAX_NORMAL_PER_COW,
                )
                diff_frame = compute_diff_per_cow(
                    normal_by_cow,
                    pairs_per_cow=DIFF_PAIRS_PER_COW,
                    seed=SEED,
                )
                diff_frame.insert(0, "model", config["label"])
                diff_frame.to_csv(diff_cache, index=False, encoding="utf-8-sig")
            finally:
                if model is not None:
                    del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.close()

        model_source = source[source["model"] == config["label"]].copy()
        model_source["cow_id"] = model_source["cow_id"].astype(str)
        diff_frame["cow_id"] = diff_frame["cow_id"].astype(str)
        if "model" in diff_frame.columns:
            diff_frame = diff_frame.drop(columns="model")
        merged = model_source.merge(
            diff_frame, on="cow_id", how="inner", validate="one_to_one"
        )
        merged_parts.append(merged)

    enhanced = pd.concat(merged_parts, ignore_index=True)
    enhanced.to_csv(ENHANCED_CSV, index=False, encoding="utf-8-sig")

    table = calculate_macro_table(enhanced, analysis.MODEL_ORDER)
    table.to_csv(TABLE_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")

    print("\n按先逐牛、后平均重新计算的结果：")
    print(table.to_string(index=False))
    print(f"\n逐牛完整数据：{ENHANCED_CSV}")
    print(f"论文汇总表：{TABLE_CSV}")


if __name__ == "__main__":
    main()
