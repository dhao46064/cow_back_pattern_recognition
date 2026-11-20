# -*- coding: utf-8 -*-
"""六模型类内紧致性与Hard异常分离分析（以牛只为统计单位）。

输出：
1. compactness_per_cow.csv：每个模型、每头牛的距离指标；
2. compactness_summary.csv：逐牛宏统计；
3. compactness_statistical_tests.csv：Friedman + Wilcoxon-Holm；
4. compactness_raincloud.png/pdf：Same、Hard N-A、Hard gap三面板雨云图。

运行：
    python scripts/test_analysis/pair_dis/plot_compactness_raincloud.py

重新计算已有模型：
    python scripts/test_analysis/pair_dis/plot_compactness_raincloud.py --overwrite
"""

import argparse
import csv
import gc
import importlib.util
import itertools
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import friedmanchisquare, wilcoxon


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
INTEGRATED_SCRIPT = PAIR_DIR / "pair_dolg_selected_loss.py"
RESNET_ABLATION_SCRIPT = PAIR_DIR / "pair_dis_ablation.py"
RESNET_SWEEP_SCRIPT = PAIR_DIR / "pair_resnet50_loss_sweep.py"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compactness_raincloud"
PER_COW_CSV = OUTPUT_DIR / "compactness_per_cow.csv"
SUMMARY_CSV = OUTPUT_DIR / "compactness_summary.csv"
STATS_CSV = OUTPUT_DIR / "compactness_statistical_tests.csv"
FIGURE_PNG = OUTPUT_DIR / "compactness_raincloud.png"
FIGURE_PDF = OUTPUT_DIR / "compactness_raincloud.pdf"

# 与原整合距离脚本保持相同数据源。
ORI_ROOT = Path(r"D:\datasets\cow_recognition8.3\openset\openset_ori_8.2")
HARD_ROOT = Path(
    r"D:\datasets\cow_recognition8.3\openset\openset_consensus_hard_11to1_v2"
)

MODELS = (
    {
        "key": "r50_af",
        "label": "R50-AF",
        "family": "resnet_ablation",
        "checkpoint": PROJECT_ROOT / "outputs" / "exp_ablation_ResNet50"
        / "checkpoints" / "epoch_10.pth",
    },
    {
        "key": "r50_center",
        "label": "R50-AF-Center",
        "family": "resnet_sweep",
        "checkpoint": PROJECT_ROOT / "outputs" / "resnet50_loss_sweep_10ep"
        / "resnet50_arcface_center_lam001" / "checkpoints" / "epoch_10.pth",
    },
    {
        "key": "r50_supcon",
        "label": "R50-AF-SupCon",
        "family": "resnet_sweep",
        "checkpoint": PROJECT_ROOT / "outputs" / "resnet50_loss_sweep_10ep"
        / "resnet50_arcface_supcon_lam01" / "checkpoints" / "epoch_10.pth",
    },
    {
        "key": "dolg_af",
        "label": "DOLG-AF",
        "family": "dolg",
        "checkpoint": PROJECT_ROOT / "outputs" / "dolg_selected_loss_10ep"
        / "dolg_arcface" / "checkpoints" / "epoch_10.pth",
    },
    {
        "key": "dolg_center",
        "label": "DOLG-AF-Center",
        "family": "dolg",
        "checkpoint": PROJECT_ROOT / "outputs" / "dolg_selected_loss_10ep"
        / "dolg_arcface_center_lam001" / "checkpoints" / "epoch_10.pth",
    },
    {
        "key": "dolg_supcon",
        "label": "DOLG-AF-SupCon",
        "family": "dolg",
        "checkpoint": PROJECT_ROOT / "outputs" / "dolg_selected_loss_10ep"
        / "dolg_arcface_supcon_lam01" / "checkpoints" / "epoch_10.pth",
    },
)

MODEL_ORDER = [item["label"] for item in MODELS]
COLORS = ["#9ECAE1", "#4292C6", "#08519C", "#FDD0A2", "#F16913", "#A63603"]
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="绘制六模型逐牛距离雨云图")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="忽略已有的单模型逐牛CSV，重新提取embedding。",
    )
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SimpleLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8")

    def log(self, message):
        print(message)
        self.file.write(str(message) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def build_model(config, integrated, resnet_ablation, resnet_sweep, logger):
    checkpoint_path = config["checkpoint"]
    family = config["family"]

    if family == "dolg":
        model, epoch, _, _ = integrated.build_dolg_model(checkpoint_path, logger)
        return model, epoch

    if family == "resnet_ablation":
        return resnet_ablation.build_model(
            model_path=str(checkpoint_path),
            backbone="resnet50",
            device=integrated.DEVICE,
            logger=logger,
        )

    # loss-sweep模型使用新版ArcFaceCowModel，需要从checkpoint推断映射头。
    from src.models.arcface_model import ArcFaceCowModel

    checkpoint, state_dict = resnet_ablation.load_checkpoint(str(checkpoint_path))
    num_classes, embedding_dim, hidden_dims = resnet_sweep.infer_model_config(state_dict)
    run_config = checkpoint.get("run_config", {}) if isinstance(checkpoint, dict) else {}
    epoch = checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown"
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
    model = model.to(integrated.DEVICE)
    model.eval()
    logger.log(f"checkpoint={checkpoint_path}")
    logger.log(f"checkpoint_epoch={epoch}")
    logger.log("strict_load=True")
    return model, epoch


def finite_mean(values: np.ndarray):
    return float(values.mean()) if values.size else float("nan")


def finite_median(values: np.ndarray):
    return float(np.median(values)) if values.size else float("nan")


def compute_model_rows(config, integrated, distance_base, resnet_ablation, resnet_sweep):
    from src.datasets.split import get_cow_ids

    cache_csv = OUTPUT_DIR / f"{config['key']}_per_cow_distances.csv"
    logger = SimpleLogger(OUTPUT_DIR / f"{config['key']}_distance.log")
    model = None
    try:
        distance_base.set_seed(SEED)
        distance_base.fresh_start()
        model, epoch = build_model(
            config, integrated, resnet_ablation, resnet_sweep, logger
        )

        cow_ids = [str(x) for x in get_cow_ids(ORI_ROOT, "unseen_cows")]
        normal_by_cow = integrated.extract_by_cow(
            distance_base, model, cow_ids, ORI_ROOT, "normal",
            integrated.MAX_NORMAL_PER_COW,
        )
        ori_abnormal_by_cow = integrated.extract_by_cow(
            distance_base, model, cow_ids, ORI_ROOT, "abnormal",
            integrated.MAX_ABNORMAL_PER_COW,
        )
        hard_abnormal_by_cow = integrated.extract_by_cow(
            distance_base, model, cow_ids, HARD_ROOT, "abnormal",
            integrated.MAX_ABNORMAL_PER_COW,
        )

        rows = []
        for cow_id, normal_embeddings in normal_by_cow.items():
            same = distance_base.compute_same_pairs(normal_embeddings)
            ori_ab = ori_abnormal_by_cow.get(cow_id)
            hard_ab = hard_abnormal_by_cow.get(cow_id)
            ori_na = (
                distance_base.compute_cross_pairs(normal_embeddings, ori_ab)
                if ori_ab is not None else np.empty(0, dtype=np.float32)
            )
            hard_na = (
                distance_base.compute_cross_pairs(normal_embeddings, hard_ab)
                if hard_ab is not None else np.empty(0, dtype=np.float32)
            )
            same_mean = finite_mean(same)
            ori_mean = finite_mean(ori_na)
            hard_mean = finite_mean(hard_na)
            rows.append({
                "model": config["label"],
                "model_key": config["key"],
                "checkpoint_epoch": epoch,
                "cow_id": cow_id,
                "normal_images": int(normal_embeddings.shape[0]),
                "ori_abnormal_images": int(ori_ab.shape[0]) if ori_ab is not None else 0,
                "hard_abnormal_images": int(hard_ab.shape[0]) if hard_ab is not None else 0,
                "same_pair_count": int(same.size),
                "same_mean": same_mean,
                "same_median": finite_median(same),
                "ori_na_pair_count": int(ori_na.size),
                "ori_na_mean": ori_mean,
                "hard_na_pair_count": int(hard_na.size),
                "hard_na_mean": hard_mean,
                "ori_gap": ori_mean - same_mean if math.isfinite(ori_mean) else float("nan"),
                "hard_gap": hard_mean - same_mean if math.isfinite(hard_mean) else float("nan"),
                "hard_ratio": hard_mean / same_mean
                if math.isfinite(hard_mean) and same_mean > 0 else float("nan"),
            })

        frame = pd.DataFrame(rows)
        frame.to_csv(cache_csv, index=False, encoding="utf-8-sig")
        return frame
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.close()


def bootstrap_ci(values, repeats=5000, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repeats, values.size), replace=True)
    means = samples.mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def make_summary(frame):
    rows = []
    for model in MODEL_ORDER:
        group = frame[frame["model"] == model]
        for metric in ("same_mean", "ori_na_mean", "hard_na_mean", "hard_gap", "hard_ratio"):
            values = group[metric].dropna().to_numpy(float)
            low, high = bootstrap_ci(values)
            rows.append({
                "model": model,
                "metric": metric,
                "cow_count": int(values.size),
                "mean": float(np.mean(values)) if values.size else float("nan"),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
                "median": float(np.median(values)) if values.size else float("nan"),
                "q25": float(np.percentile(values, 25)) if values.size else float("nan"),
                "q75": float(np.percentile(values, 75)) if values.size else float("nan"),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
            })
    return pd.DataFrame(rows)


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def statistical_tests(frame):
    results = []
    for metric in ("same_mean", "hard_na_mean", "hard_gap"):
        pivot = frame.pivot_table(index="cow_id", columns="model", values=metric)
        pivot = pivot.reindex(columns=MODEL_ORDER).dropna()
        if len(pivot) < 3:
            continue
        statistic, p_value = friedmanchisquare(
            *[pivot[column].to_numpy() for column in MODEL_ORDER]
        )
        results.append({
            "metric": metric,
            "test": "Friedman",
            "model_a": "all",
            "model_b": "all",
            "n_cows": len(pivot),
            "statistic": statistic,
            "p_raw": p_value,
            "p_holm": p_value,
        })

        pair_rows = []
        for model_a, model_b in itertools.combinations(MODEL_ORDER, 2):
            paired = pivot[[model_a, model_b]].dropna()
            try:
                stat, p = wilcoxon(
                    paired[model_a], paired[model_b],
                    alternative="two-sided", zero_method="wilcox",
                )
            except ValueError:
                stat, p = float("nan"), 1.0
            pair_rows.append({
                "metric": metric,
                "test": "Wilcoxon paired",
                "model_a": model_a,
                "model_b": model_b,
                "n_cows": len(paired),
                "statistic": stat,
                "p_raw": p,
            })
        adjusted = holm_adjust([row["p_raw"] for row in pair_rows])
        for row, p_holm in zip(pair_rows, adjusted):
            row["p_holm"] = p_holm
            results.append(row)
    return pd.DataFrame(results)


def raincloud(ax, frame, metric, title, ylabel, direction):
    rng = np.random.default_rng(SEED)
    positions = np.arange(len(MODEL_ORDER), dtype=float)

    for index, (position, model, color) in enumerate(zip(positions, MODEL_ORDER, COLORS)):
        values = frame.loc[frame["model"] == model, metric].dropna().to_numpy(float)
        if values.size == 0:
            continue

        # 左半边小提琴（云）。
        violin = ax.violinplot(
            [values], positions=[position], widths=0.70,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.35)
            vertices = body.get_paths()[0].vertices
            vertices[:, 0] = np.minimum(vertices[:, 0], position)

        # 中间箱线图。
        box = ax.boxplot(
            [values], positions=[position + 0.04], widths=0.13,
            patch_artist=True, showfliers=False,
            medianprops={"color": "#202020", "linewidth": 1.5},
            boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1.2},
            whiskerprops={"color": color, "linewidth": 1.1},
            capprops={"color": color, "linewidth": 1.1},
        )

        # 右侧逐牛散点（雨）；每个点是一头牛，不是一个pair。
        jitter = rng.normal(loc=0.21, scale=0.035, size=values.size)
        ax.scatter(
            position + jitter, values, s=11, alpha=0.35,
            color=color, edgecolors="none", rasterized=True,
        )

        # 菱形表示逐牛均值，误差棒表示牛只bootstrap 95% CI。
        ci_low, ci_high = bootstrap_ci(values)
        mean = float(values.mean())
        ax.errorbar(
            position + 0.04, mean,
            yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="D", markersize=4.5, color="#111111",
            capsize=3, linewidth=1.1, zorder=6,
        )

    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(positions + 0.05)
    ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right", fontsize=8.5)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.02, 0.97, direction, transform=ax.transAxes,
        ha="left", va="top", fontsize=8.5, color="#555555",
    )


def make_figure(frame):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 5.0), constrained_layout=True)
    raincloud(
        axes[0], frame, "same_mean",
        "(a) Intra-class compactness", "Same-normal cosine distance", "Lower is better",
    )
    raincloud(
        axes[1], frame, "hard_na_mean",
        "(b) Hard abnormal separation", "Hard normal–abnormal distance", "Higher is better",
    )
    raincloud(
        axes[2], frame, "hard_gap",
        "(c) Absolute separation margin", "Hard N–A − Same normal", "Higher is better",
    )
    figure.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    args = parse_args()
    for path in (ORI_ROOT, HARD_ROOT):
        if not path.is_dir():
            raise FileNotFoundError(f"数据集不存在：{path}")
    for config in MODELS:
        if not config["checkpoint"].is_file():
            raise FileNotFoundError(f"checkpoint不存在：{config['checkpoint']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    integrated = load_module("compactness_integrated", INTEGRATED_SCRIPT)
    distance_base = integrated.load_base_module()
    resnet_ablation = load_module("compactness_resnet_ablation", RESNET_ABLATION_SCRIPT)
    resnet_sweep = load_module("compactness_resnet_sweep", RESNET_SWEEP_SCRIPT)

    frames = []
    for index, config in enumerate(MODELS, 1):
        cache_csv = OUTPUT_DIR / f"{config['key']}_per_cow_distances.csv"
        if cache_csv.is_file() and not args.overwrite:
            print(f"[{index}/{len(MODELS)}] REUSE {cache_csv.name}")
            model_frame = pd.read_csv(cache_csv, dtype={"cow_id": str})
        else:
            print(f"[{index}/{len(MODELS)}] COMPUTE {config['label']}")
            model_frame = compute_model_rows(
                config, integrated, distance_base, resnet_ablation, resnet_sweep
            )
        frames.append(model_frame)

    frame = pd.concat(frames, ignore_index=True)
    frame["model"] = pd.Categorical(frame["model"], MODEL_ORDER, ordered=True)
    frame = frame.sort_values(["model", "cow_id"])
    frame.to_csv(PER_COW_CSV, index=False, encoding="utf-8-sig")

    summary = make_summary(frame)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    tests = statistical_tests(frame)
    tests.to_csv(STATS_CSV, index=False, encoding="utf-8-sig")
    make_figure(frame)

    print("\n完成：")
    print(PER_COW_CSV)
    print(SUMMARY_CSV)
    print(STATS_CSV)
    print(FIGURE_PNG)
    print(FIGURE_PDF)


if __name__ == "__main__":
    main()
