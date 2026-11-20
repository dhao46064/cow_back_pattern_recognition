# -*- coding: utf-8 -*-
"""六个模型逐个计算牛只级（macro）距离指标。

每头牛先计算：Same normal、Diff normal、Ori N-A、Hard N-A、Hard gap，
再对牛只等权平均。运行时用 --model 指定一个模型，避免六个模型连续占用显存。
"""

import argparse
import csv
import gc
import importlib.util
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


FILE = Path(__file__).resolve()


def find_project_root() -> Path:
    candidates = [FILE.parent, *FILE.parents, Path(r"D:\code_project\cow_recognition")]
    for path in candidates:
        if (path / "src").is_dir():
            return path
    raise FileNotFoundError("找不到包含 src 的项目根目录")


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_FILE = ROOT / "scripts/test_analysis/pair_dis/pair_dis_with_abnormal.py"
ORI_ROOT = Path(r"D:\datasets\cow_recognition8.3\openset\openset_ori_8.2")
HARD_ROOT = Path(r"D:\datasets\cow_recognition8.3\openset\openset_consensus_hard_11to1_v2")
OUT_DIR = ROOT / "outputs/cow_macro_distance"
SUMMARY = OUT_DIR / "cow_macro_distance_summary.csv"

SEED = 42
BATCH_SIZE = 64
MAX_NORMAL = 200
MAX_ABNORMAL = None
TOTAL_DIFF_PAIRS = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 只保留论文最终六组。若你的checkpoint位置不同，只修改这里。
MODELS = {
    "r50_arcface": {
        "label": "ResNet50 + ArcFace",
        "type": "resnet50",
        "ckpt": ROOT / "outputs/exp_ablation_ResNet50/checkpoints/epoch_10.pth",
    },
    "r50_center": {
        "label": "ResNet50 + ArcFace + Center (0.01)",
        "type": "resnet50",
        "ckpt": ROOT / "outputs/resnet50_loss_sweep_10ep/resnet50_arcface_center_lam001/checkpoints/epoch_10.pth",
    },
    "r50_supcon": {
        "label": "ResNet50 + ArcFace + SupCon (0.1)",
        "type": "resnet50",
        "ckpt": ROOT / "outputs/resnet50_loss_sweep_10ep/resnet50_arcface_supcon_lam01/checkpoints/epoch_10.pth",
    },
    "dolg_arcface": {
        "label": "DOLG + ArcFace",
        "type": "dolg",
        "ckpt": ROOT / "outputs/dolg_selected_loss_10ep/dolg_arcface/checkpoints/epoch_10.pth",
    },
    "dolg_center": {
        "label": "DOLG + ArcFace + Center (0.01)",
        "type": "dolg",
        "ckpt": ROOT / "outputs/dolg_selected_loss_10ep/dolg_arcface_center_lam001/checkpoints/epoch_10.pth",
    },
    "dolg_supcon": {
        "label": "DOLG + ArcFace + SupCon (0.1)",
        "type": "dolg",
        "ckpt": ROOT / "outputs/dolg_selected_loss_10ep/dolg_arcface_supcon_lam01/checkpoints/epoch_10.pth",
    },
}


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", *MODELS],
        help="默认all顺序运行六组，也可指定单个模型。",
    )
    parser.add_argument(
        "--diff-pairs",
        type=int,
        default=TOTAL_DIFF_PAIRS,
        help="全局随机异牛normal pair总数，默认与原实验一致为10000。",
    )
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("pair_macro_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载基础脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_checkpoint(path: Path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)))
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint没有有效state_dict：{path}")
    return checkpoint, state


def infer_hidden_dims(state):
    layers = []
    for key, value in state.items():
        if key.startswith("embedder.head.") and key.endswith(".weight") and getattr(value, "ndim", 0) == 2:
            index = key[len("embedder.head."):].split(".", 1)[0]
            if index.isdigit():
                layers.append((int(index), int(value.shape[0])))
    layers.sort()
    if not layers:
        raise KeyError("无法从embedder.head推断ResNet50映射头")
    return tuple(size for _, size in layers[:-1])


def remap_legacy_resnet50_keys(state):
    """兼容旧基线的 backbone.* / head.* 权重命名。"""
    if not any(key.startswith("backbone.") or key.startswith("head.") for key in state):
        return state
    mapped = {}
    for key, value in state.items():
        if key.startswith("backbone."):
            key = "embedder.backbone." + key[len("backbone."):]
        elif key.startswith("head."):
            key = "embedder.head." + key[len("head."):]
        mapped[key] = value
    return mapped


def build_model(config):
    checkpoint, state = load_checkpoint(config["ckpt"])
    arc = state.get("arcface.weight")
    if arc is None:
        raise KeyError("checkpoint中没有arcface.weight")
    classes, emb_dim = int(arc.shape[0]), int(arc.shape[1])
    run = checkpoint.get("run_config", {}) if isinstance(checkpoint, dict) else {}

    if config["type"] == "dolg":
        from src.models.dolg_arcface_model import DOLGArcFaceCowModel
        model = DOLGArcFaceCowModel(
            num_classes=classes, embedding_dim=emb_dim,
            local_dim=int(run.get("local_dim", 512)),
            fusion_hidden_dim=int(run.get("fusion_hidden_dim", 512)),
            s=float(run.get("arc_s", 30.0)), m=float(run.get("arc_m", 0.25)),
            pretrained=False, freeze_backbone=False,
            gem_p=float(run.get("gem_p", 3.0)),
            gem_learn_p=not bool(run.get("fixed_gem_p", False)),
            dilation_rates=tuple(run.get("dilation_rates", (1, 2, 3))),
            dropout=float(run.get("dropout", 0.3)),
        )
    else:
        from src.models.arcface_model import ArcFaceCowModel
        state = remap_legacy_resnet50_keys(state)
        model = ArcFaceCowModel(
            num_classes=classes, embedding_dim=emb_dim,
            s=float(run.get("arc_s", 30.0)), m=float(run.get("arc_m", 0.25)),
            pretrained=False, freeze_backbone=False,
            hidden_dims=infer_hidden_dims(state), dropout=0.3,
            use_gem=("embedder.pool.p" in state),
        )
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def extract_by_cow(base, model, cow_ids, root, label, max_count):
    transform = base.build_eval_transforms(img_size=224)
    result = {}
    minimum = 2 if label == "normal" else 1
    for cow_id in tqdm(cow_ids, desc=f"extract {root.name}/{label}"):
        paths = base.list_images_for_cow(root, cow_id, label)
        paths = base.deterministic_sample(paths, max_count, SEED, cow_id, f"{root.name}:{label}")
        if len(paths) < minimum:
            continue
        emb, _ = base.extract_embeddings_for_paths(
            model, paths, transform, DEVICE, batch_size=BATCH_SIZE
        )
        if len(emb) >= minimum:
            result[str(cow_id)] = emb.cpu()
    return result


def all_same_mean(emb):
    # L2归一化embedding的余弦距离；只取上三角，不包含自身。
    n = emb.shape[0]
    idx = torch.triu_indices(n, n, offset=1)
    values = 1.0 - torch.sum(emb[idx[0]] * emb[idx[1]], dim=1)
    return float(values.clamp(0.0, 2.0).mean())


def all_cross_mean(a, b, chunk=1024):
    # 使用全部normal×abnormal pair，分块仅为节省内存。
    total, count = 0.0, 0
    for start in range(0, a.shape[0], chunk):
        values = 1.0 - a[start:start + chunk] @ b.T
        total += float(values.clamp(0.0, 2.0).sum())
        count += values.numel()
    return total / count


def sample_global_diff_per_cow(normal_by_cow, total_pairs=10000, seed=42):
    """复现原micro的全局类间采样，并转换为逐牛统计。

    全局只生成total_pairs个异牛normal pair。每个pair的距离同时归属于
    该pair涉及的两头牛；随后每头牛内部求均值。这样不改变原始采样量，
    最终又能让每头牛作为一个macro统计单位。
    """
    cows = sorted(normal_by_cow)
    if len(cows) < 2:
        raise RuntimeError("有效normal牛只不足2头，无法计算Diff normal")

    rng = random.Random(seed)
    values_by_cow = {cow_id: [] for cow_id in cows}
    pooled_values = []

    for _ in tqdm(range(total_pairs), desc="sample global diff pairs"):
        cow_a, cow_b = rng.sample(cows, 2)
        emb_a = normal_by_cow[cow_a]
        emb_b = normal_by_cow[cow_b]
        index_a = rng.randrange(emb_a.shape[0])
        index_b = rng.randrange(emb_b.shape[0])
        distance = float(
            (1.0 - torch.dot(emb_a[index_a], emb_b[index_b]))
            .clamp(0.0, 2.0)
        )
        pooled_values.append(distance)
        values_by_cow[cow_a].append(distance)
        values_by_cow[cow_b].append(distance)

    per_cow = {
        cow_id: float(np.mean(values))
        for cow_id, values in values_by_cow.items()
        if values
    }
    pair_counts = {
        cow_id: len(values)
        for cow_id, values in values_by_cow.items()
    }
    return per_cow, pair_counts, float(np.mean(pooled_values))


def mean_ci(values, seed=42, repeats=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        boot[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(values.mean()), float(values.std(ddof=1)), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def save_summary(row):
    rows = []
    if SUMMARY.is_file():
        with SUMMARY.open("r", newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
    rows = [old for old in rows if old["model_key"] != row["model_key"]]
    rows.append(row)
    order = {key: i for i, key in enumerate(MODELS)}
    rows.sort(key=lambda item: order[item["model_key"]])
    fieldnames = list(row)
    # 兼容先前版本总表：只写入当前固定字段，避免旧字段造成写入失败。
    rows = [{key: old.get(key, "") for key in fieldnames} for old in rows]
    with SUMMARY.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one(model_key, diff_pairs, base):
    config = MODELS[model_key]
    if not config["ckpt"].is_file():
        raise FileNotFoundError(f"checkpoint不存在：{config['ckpt']}")
    base.set_seed(SEED)
    model = build_model(config)

    from src.datasets.split import get_cow_ids
    cow_ids = [str(x) for x in get_cow_ids(ORI_ROOT, "unseen_cows")]
    normal = extract_by_cow(base, model, cow_ids, ORI_ROOT, "normal", MAX_NORMAL)
    ori_abnormal = extract_by_cow(base, model, cow_ids, ORI_ROOT, "abnormal", MAX_ABNORMAL)
    hard_abnormal = extract_by_cow(base, model, cow_ids, HARD_ROOT, "abnormal", MAX_ABNORMAL)

    # 与原micro实验完全相同：全局固定抽取10000个异牛normal pair。
    # 区别只在汇总：这里保留pair两端牛号，先逐牛平均，再对牛等权平均。
    diff_by_cow, diff_pair_counts, diff_micro_check = sample_global_diff_per_cow(
        normal,
        total_pairs=diff_pairs,
        seed=SEED,
    )

    rows = []
    for cow_id in tqdm(sorted(normal), desc="compute per-cow macro units"):
        same = all_same_mean(normal[cow_id])
        diff = diff_by_cow.get(cow_id, float("nan"))
        ori = (
            all_cross_mean(normal[cow_id], ori_abnormal[cow_id])
            if cow_id in ori_abnormal else float("nan")
        )
        hard = (
            all_cross_mean(normal[cow_id], hard_abnormal[cow_id])
            if cow_id in hard_abnormal else float("nan")
        )
        rows.append({
            "model_key": model_key, "model": config["label"], "cow_id": cow_id,
            "normal_count": normal[cow_id].shape[0],
            "diff_pair_count": diff_pair_counts.get(cow_id, 0),
            "ori_abnormal_count": ori_abnormal[cow_id].shape[0] if cow_id in ori_abnormal else 0,
            "hard_abnormal_count": hard_abnormal[cow_id].shape[0] if cow_id in hard_abnormal else 0,
            "same_normal": same, "diff_normal": diff,
            "ori_na": ori, "hard_na": hard, "hard_gap": hard - same,
        })
    if not rows:
        raise RuntimeError("没有同时具备normal、ori abnormal、hard abnormal的牛只")

    per_cow_file = OUT_DIR / f"{model_key}_per_cow.csv"
    with per_cow_file.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    summary = {
        "model_key": model_key,
        "model": config["label"],
        "diff_global_pair_count": diff_pairs,
        "diff_micro_check": f"{diff_micro_check:.6f}",
    }
    for column in ("same_normal", "diff_normal", "ori_na", "hard_na", "hard_gap"):
        valid = [row[column] for row in rows if np.isfinite(row[column])]
        if not valid:
            raise RuntimeError(f"{column}没有有效牛只")
        mean, sd, low, high = mean_ci(valid)
        summary[f"{column}_cow_count"] = len(valid)
        summary[f"{column}_mean"] = f"{mean:.6f}"
        summary[f"{column}_sd"] = f"{sd:.6f}"
        summary[f"{column}_ci95_low"] = f"{low:.6f}"
        summary[f"{column}_ci95_high"] = f"{high:.6f}"
    save_summary(summary)

    print("\n完成：", config["label"])
    for column in ("same_normal", "diff_normal", "ori_na", "hard_na", "hard_gap"):
        print(
            f"{column:12s} = {summary[column + '_mean']}  "
            f"95%CI [{summary[column + '_ci95_low']}, {summary[column + '_ci95_high']}]  "
            f"cows={summary[column + '_cow_count']}"
        )
    print("逐牛结果：", per_cow_file)
    print("汇总结果：", SUMMARY)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    opt = args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_module(BASE_FILE)
    selected = list(MODELS) if opt.model == "all" else [opt.model]

    print("=" * 80)
    print(f"将顺序运行 {len(selected)} 个模型：{', '.join(selected)}")
    print("统计单位：牛只；所有最终均值均为cow-macro mean")
    print("=" * 80)

    for index, model_key in enumerate(selected, start=1):
        print(f"\n[{index}/{len(selected)}] START {model_key}")
        run_one(model_key, opt.diff_pairs, base)

    print(f"\n全部完成：{SUMMARY}")


if __name__ == "__main__":
    main()
