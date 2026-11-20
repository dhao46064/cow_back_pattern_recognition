# -*- coding: utf-8 -*-
"""三组无红增强 DOLG 的整合 Pair Distance 测试。

每个模型只提取一次共同 normal，并统一输出：
1. same_normal：同一编号内部 normal-normal 的全部不重复 pair；
2. diff_normal：不同编号 normal-normal 的固定随机 pair；
3. ori_normal_abnormal：ori 中同一编号 normal×abnormal 的全部 pair；
4. hard_normal_abnormal：hard 中同一编号 normal×abnormal 的全部 pair。
"""

import csv
import gc
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm


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
    / "pair_dis_with_abnormal.py"
)
MODEL_ROOT = PROJECT_ROOT / "outputs" / "dolg_selected_loss_10ep"
OUTPUT_DIR = MODEL_ROOT / "pair_distance_integrated"
SUMMARY_CSV = OUTPUT_DIR / "dolg_selected_pair_distance_integrated.csv"

ORI_ROOT = Path(
    r"D:\datasets\cow_recognition8.3\openset\openset_ori_8.2"
)
HARD_ROOT = Path(
    r"D:\datasets\cow_recognition8.3\openset"
    r"\openset_consensus_hard_11to1_v2"
)

EXPERIMENT_ORDER = (
    "dolg_arcface",
    "dolg_arcface_center_lam001",
    "dolg_arcface_supcon_lam01",
)

SPLIT_MODE = "unseen"
MAX_NORMAL_PER_COW = 200
MAX_ABNORMAL_PER_COW = None  # None 表示 abnormal 全部使用。
RANDOM_DIFF_PAIRS = 10000
BATCH_SIZE = 64
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "pair_dis_with_abnormal_base",
        BASE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 Pair Distance 基础脚本：{BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def discover_models() -> Dict[str, Path]:
    models: Dict[str, Path] = {}
    for experiment_name in EXPERIMENT_ORDER:
        checkpoint = (
            MODEL_ROOT
            / experiment_name
            / "checkpoints"
            / "epoch_10.pth"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"三组模型必须全部存在，缺少：{checkpoint}"
            )
        models[experiment_name] = checkpoint
    return models


def load_checkpoint(path: Path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint 中没有有效模型权重：{path}")
    return checkpoint, state_dict


def build_dolg_model(checkpoint_path: Path, logger):
    from src.models.dolg_arcface_model import DOLGArcFaceCowModel

    checkpoint, state_dict = load_checkpoint(checkpoint_path)
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
    model = model.to(DEVICE)
    model.eval()

    logger.log(f"checkpoint={checkpoint_path}")
    logger.log(f"checkpoint_epoch={checkpoint_epoch}")
    logger.log(f"classes={num_classes}")
    logger.log(f"embedding_dim={embedding_dim}")
    logger.log("strict_load=True, missing=0, unexpected=0")
    return model, checkpoint_epoch, num_classes, embedding_dim


def concatenate(parts: List[np.ndarray]) -> np.ndarray:
    valid = [part for part in parts if part.size > 0]
    return (
        np.concatenate(valid)
        if valid
        else np.empty(0, dtype=np.float32)
    )


def mean_or_nan(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def ratio_or_nan(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return float("nan")
    return numerator / (denominator + 1e-12)


def extract_by_cow(base, model, cow_ids, root: Path, label: str, max_count):
    result = {}
    transform = base.build_eval_transforms(img_size=224)

    for cow_id in tqdm(cow_ids, desc=f"提取 {root.name}/{label}"):
        paths = base.list_images_for_cow(root, cow_id, label)
        paths = base.deterministic_sample(
            paths=paths,
            max_count=max_count,
            seed=SEED,
            cow_id=cow_id,
            label=f"{root.name}:{label}",
        )

        # normal 至少需要2张；abnormal 至少需要1张。
        minimum = 2 if label == "normal" else 1
        if len(paths) < minimum:
            continue

        embeddings, _ = base.extract_embeddings_for_paths(
            model=model,
            img_paths=paths,
            transform=transform,
            device=DEVICE,
            batch_size=BATCH_SIZE,
        )
        if embeddings.shape[0] >= minimum:
            result[str(cow_id)] = embeddings

    return result


def compute_same_all(base, normal_by_cow) -> np.ndarray:
    return concatenate([
        base.compute_same_pairs(embeddings)
        for embeddings in normal_by_cow.values()
    ])


def compute_normal_abnormal_all(base, normal_by_cow, abnormal_by_cow):
    parts = []
    matched_cows = 0
    for cow_id, abnormal_embeddings in abnormal_by_cow.items():
        normal_embeddings = normal_by_cow.get(cow_id)
        if normal_embeddings is None:
            continue
        parts.append(
            base.compute_cross_pairs(normal_embeddings, abnormal_embeddings)
        )
        matched_cows += 1
    return concatenate(parts), matched_cows


def save_summary(rows: List[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_one_model(base, model_name: str, checkpoint_path: Path) -> dict:
    from src.datasets.split import get_cow_ids

    report_path = OUTPUT_DIR / f"{model_name}_pair_distance_integrated.txt"
    logger = base.ReportLogger(report_path)
    model = None

    try:
        base.set_seed(SEED)
        base.fresh_start()
        logger.log("=" * 80)
        logger.log(f"model={model_name}")
        logger.log(f"device={DEVICE}")
        logger.log(f"normal_root={ORI_ROOT}")
        logger.log(f"ori_abnormal_root={ORI_ROOT}")
        logger.log(f"hard_abnormal_root={HARD_ROOT}")
        logger.log(f"random_diff_pairs={RANDOM_DIFF_PAIRS}")
        logger.log("=" * 80)

        model, epoch, num_classes, embedding_dim = build_dolg_model(
            checkpoint_path,
            logger,
        )

        split_name = "unseen_cows" if SPLIT_MODE == "unseen" else "seen_cows"
        cow_ids = [str(cow_id) for cow_id in get_cow_ids(ORI_ROOT, split_name)]

        # normal 只提取一次，并同时用于 same、diff、ori/hard normal-abnormal。
        normal_by_cow = extract_by_cow(
            base, model, cow_ids, ORI_ROOT, "normal", MAX_NORMAL_PER_COW
        )
        ori_abnormal_by_cow = extract_by_cow(
            base, model, cow_ids, ORI_ROOT, "abnormal", MAX_ABNORMAL_PER_COW
        )
        hard_abnormal_by_cow = extract_by_cow(
            base, model, cow_ids, HARD_ROOT, "abnormal", MAX_ABNORMAL_PER_COW
        )

        same_values = compute_same_all(base, normal_by_cow)
        diff_values = base.sample_random_inter_cow_pairs(
            normal_by_cow,
            max_pairs=RANDOM_DIFF_PAIRS,
            seed=SEED,
        )
        ori_na_values, ori_matched_cows = compute_normal_abnormal_all(
            base, normal_by_cow, ori_abnormal_by_cow
        )
        hard_na_values, hard_matched_cows = compute_normal_abnormal_all(
            base, normal_by_cow, hard_abnormal_by_cow
        )

        same_mean = mean_or_nan(same_values)
        diff_mean = mean_or_nan(diff_values)
        ori_na_mean = mean_or_nan(ori_na_values)
        hard_na_mean = mean_or_nan(hard_na_values)

        row = {
            "model": model_name,
            "checkpoint": str(checkpoint_path),
            "epoch": epoch,
            "num_classes": num_classes,
            "embedding_dim": embedding_dim,
            "normal_cows": len(normal_by_cow),
            "same_normal_count": int(same_values.size),
            "same_normal_mean": same_mean,
            "diff_normal_count": int(diff_values.size),
            "diff_normal_mean": diff_mean,
            "diff_same_ratio": ratio_or_nan(diff_mean, same_mean),
            "ori_matched_cows": ori_matched_cows,
            "ori_normal_abnormal_count": int(ori_na_values.size),
            "ori_normal_abnormal_mean": ori_na_mean,
            "ori_na_same_ratio": ratio_or_nan(ori_na_mean, same_mean),
            "hard_matched_cows": hard_matched_cows,
            "hard_normal_abnormal_count": int(hard_na_values.size),
            "hard_normal_abnormal_mean": hard_na_mean,
            "hard_na_same_ratio": ratio_or_nan(hard_na_mean, same_mean),
        }

        logger.log("")
        logger.log("=" * 80)
        logger.log("最终均值")
        logger.log("=" * 80)
        logger.log(f"same_normal_mean={same_mean:.6f} (pairs={same_values.size})")
        logger.log(f"diff_normal_mean={diff_mean:.6f} (pairs={diff_values.size})")
        logger.log(f"diff_same_ratio={row['diff_same_ratio']:.6f}")
        logger.log(
            f"ori_normal_abnormal_mean={ori_na_mean:.6f} "
            f"(pairs={ori_na_values.size}, cows={ori_matched_cows})"
        )
        logger.log(
            f"hard_normal_abnormal_mean={hard_na_mean:.6f} "
            f"(pairs={hard_na_values.size}, cows={hard_matched_cows})"
        )
        return row

    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.close()


def main() -> None:
    if not ORI_ROOT.is_dir():
        raise FileNotFoundError(f"ori 数据集不存在：{ORI_ROOT}")
    if not HARD_ROOT.is_dir():
        raise FileNotFoundError(f"hard 数据集不存在：{HARD_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_module()
    models = discover_models()
    rows: List[dict] = []

    print("=" * 100)
    print("DOLG 三模型整合 Pair Distance")
    print(f"normal/same/diff: {ORI_ROOT}")
    print(f"ori abnormal:     {ORI_ROOT}")
    print(f"hard abnormal:    {HARD_ROOT}")
    print(f"output:           {SUMMARY_CSV}")
    print("=" * 100)

    for index, (model_name, checkpoint_path) in enumerate(models.items(), 1):
        print(f"\n[{index}/{len(models)}] START {model_name}")
        row = run_one_model(base, model_name, checkpoint_path)
        rows.append(row)
        # 每跑完一个模型立即保存，意外中断也保留已有结果。
        save_summary(rows)

    print("\n全部完成：", SUMMARY_CSV)


if __name__ == "__main__":
    main()
