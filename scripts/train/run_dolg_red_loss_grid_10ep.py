# -*- coding: utf-8 -*-
"""DOLG × 红色增强概率 × 辅助损失的12组交叉实验。

红色增强概率：0.05、0.10、0.15、0.20
损失配置：
1. ArcFace
2. ArcFace + Center Loss，lambda_center=0.01
3. ArcFace + SupCon，lambda_supcon=0.1

所有实验固定：10 epochs、seen_train only、P=4、K=4、batch=16。
"""

import argparse
import csv
import importlib.util
import json
import logging
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler


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
    / "train"
    / "run_resnet50_loss_sweep_10ep.py"
)


def load_training_base():
    spec = importlib.util.spec_from_file_location(
        "dolg_red_grid_training_base",
        BASE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载基础训练脚本：{BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DeterministicPKBatchSampler(Sampler):
    """使用独立随机数生成器，避免红增强随机数改变PK采样序列。"""

    def __init__(self, labels, p: int, k: int, seed: int):
        self.labels = [int(label) for label in labels]
        self.p = int(p)
        self.k = int(k)
        self.seed = int(seed)
        self.batch_size = self.p * self.k
        self.epoch_index = 0

        self.label_to_indices = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.label_to_indices[label].append(index)

        self.label_set = sorted(self.label_to_indices)
        if self.p > len(self.label_set):
            raise ValueError(
                f"P={self.p}超过训练类别数{len(self.label_set)}。"
            )
        self.num_batches = len(self.labels) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch_index)
        self.epoch_index += 1
        batches = []

        for _ in range(self.num_batches):
            selected_labels = rng.sample(self.label_set, self.p)
            batch = []
            for label in selected_labels:
                indices = self.label_to_indices[label]
                if len(indices) >= self.k:
                    batch.extend(rng.sample(indices, self.k))
                else:
                    batch.extend(rng.choices(indices, k=self.k))
            batches.append(batch)

        rng.shuffle(batches)
        yield from batches

    def __len__(self):
        return self.num_batches


def parse_args():
    parser = argparse.ArgumentParser(
        description="运行DOLG红色增强与损失函数12组交叉实验。"
    )
    parser.add_argument(
        "--obb_root",
        default=r"D:\datasets\cow_recognition8.3\train_dataset",
    )
    parser.add_argument(
        "--red_mask_root",
        default=r"D:\datasets\cow_recognition8.3\train_dataset_masks",
    )
    parser.add_argument(
        "--output_root",
        default=(
            r"D:\code_project\cow_recognition\outputs"
            r"\dolg_red_loss_grid_10ep"
        ),
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--p", type=int, default=16)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--red_probs",
        type=float,
        nargs="+",
        default=(0.05, 0.10, 0.15, 0.20),
    )
    parser.add_argument(
        "--red_strengths",
        type=float,
        nargs="+",
        default=(0.4, 0.6, 0.8, 1.0),
    )
    parser.add_argument(
        "--red_strength_probs",
        type=float,
        nargs="+",
        default=(0.35, 0.35, 0.20, 0.10),
    )
    parser.add_argument("--red_r_target", type=float, default=0.78)
    parser.add_argument("--red_g_target", type=float, default=0.55)
    parser.add_argument("--red_b_target", type=float, default=0.50)
    parser.add_argument(
        "--missing_mask_policy",
        choices=("error", "original"),
        default="error",
    )
    parser.add_argument(
        "--validate_red_masks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--local_dim", type=int, default=512)
    parser.add_argument("--fusion_hidden_dim", type=int, default=512)
    parser.add_argument("--arc_s", type=float, default=30.0)
    parser.add_argument("--arc_m", type=float, default=0.25)
    parser.add_argument("--gem_p", type=float, default=3.0)
    parser.add_argument(
        "--fixed_gem_p",
        action="store_true",
        help="固定GeM p；默认允许学习。",
    )
    parser.add_argument(
        "--dilation_rates",
        type=int,
        nargs=3,
        default=(1, 2, 3),
        metavar=("D1", "D2", "D3"),
    )
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--lambda_center", type=float, default=0.01)
    parser.add_argument("--lambda_supcon", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--projection_hidden_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--center_lr", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="重新训练已经存在epoch_10.pth的实验。",
    )
    parser.add_argument(
        "--continue_on_error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认某组失败后继续后续实验。",
    )
    return parser.parse_args()


# def validate_args(args) -> None:
#     if args.epochs != 10:
#         raise ValueError("本实验固定比较第10个epoch，请保持--epochs 10。")
#     if args.p * args.k != 16:
#         raise ValueError(
#             f"8GB显存下DOLG固定batch=16，当前P*K={args.p * args.k}。"
#         )
#     if args.p < 2 or args.k < 2:
#         raise ValueError("SupCon要求P>=2且K>=2。")
#     if args.num_workers != 0:
#         raise ValueError("为准确记录红增强次数，本脚本固定num_workers=0。")
#     if len(args.red_strengths) != len(args.red_strength_probs):
#         raise ValueError("red_strengths与red_strength_probs长度必须一致。")
#     if any(probability < 0.0 or probability > 1.0 for probability in args.red_probs):
#         raise ValueError("所有red_probs必须位于[0,1]。")
#     if args.device.startswith("cuda") and not torch.cuda.is_available():
#         raise RuntimeError("CUDA不可用；DOLG实验不自动切换到CPU。")


def probability_tag(probability: float) -> str:
    return f"{int(round(probability * 100)):03d}"


def write_summary_row(summary_path: Path, row: dict) -> None:
    fieldnames = [
        "experiment",
        "red_aug_probability",
        "auxiliary",
        "auxiliary_weight",
        "epoch",
        "train_loss",
        "train_arcface_loss",
        "train_auxiliary_loss_raw",
        "train_auxiliary_loss_weighted",
        "train_accuracy",
        "actual_red_ratio",
        "red_total_calls",
        "red_augmented_count",
        "red_strength_counts",
        "duration_minutes",
        "checkpoint",
        "status",
        "error",
    ]

    rows = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))

    rows = [item for item in rows if item.get("experiment") != row["experiment"]]
    rows.append(row)

    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    # validate_args(args)
    base = load_training_base()

    from src.augmentations.red_color_shift import CowBackRedColorShift
    from src.datasets.dataset_OBB_normal import CowTrainDatasetNormal
    from src.datasets.split import get_samples
    from src.models.dolg_arcface_model import DOLGArcFaceCowModel
    from src.utils.transforms import build_transforms

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "dolg_red_loss_grid_summary.csv"

    logger = logging.getLogger("dolg_red_loss_grid")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_root / "grid_master.log",
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    train_samples = get_samples(args.obb_root, "seen_train")
    train_cow_ids = sorted({str(sample["cow_id"]) for sample in train_samples})
    id2label = {cow_id: index for index, cow_id in enumerate(train_cow_ids)}
    train_transform = build_transforms(train=False)

    if not train_samples:
        raise RuntimeError("seen_train为空。")
    if not Path(args.red_mask_root).is_dir():
        raise FileNotFoundError(f"红增强mask目录不存在：{args.red_mask_root}")

    red_common_config = {
        "strengths": list(args.red_strengths),
        "strength_probs": list(args.red_strength_probs),
        "r_target": args.red_r_target,
        "g_target": args.red_g_target,
        "b_target": args.red_b_target,
        "missing_mask_policy": args.missing_mask_policy,
    }

    if args.validate_red_masks:
        logger.info("开始一次性检查全部seen_train红增强mask。")
        validator = CowBackRedColorShift(
            image_root=args.obb_root,
            mask_root=args.red_mask_root,
            probability=min(args.red_probs),
            **red_common_config,
        )
        missing_masks = validator.find_missing_masks(
            sample["path"] for sample in train_samples
        )
        if missing_masks:
            preview = "\n".join(
                f"image={item['image_path']} | mask={item['mask_path']}"
                for item in missing_masks[:20]
            )
            raise FileNotFoundError(
                f"共有{len(missing_masks)}张训练图缺少mask。前20条：\n{preview}"
            )
        logger.info("mask检查通过：train_samples=%d", len(train_samples))

    def build_dolg_model(num_classes, embedding_dim, arc_s, arc_m):
        return DOLGArcFaceCowModel(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            local_dim=args.local_dim,
            fusion_hidden_dim=args.fusion_hidden_dim,
            s=arc_s,
            m=arc_m,
            pretrained=True,
            freeze_backbone=False,
            gem_p=args.gem_p,
            gem_learn_p=not args.fixed_gem_p,
            dilation_rates=tuple(args.dilation_rates),
            dropout=args.dropout,
        )

    def build_deterministic_loader(
        train_dataset,
        p,
        k,
        num_workers,
        device,
    ):
        labels = [int(label) for _, label in train_dataset.data]
        sampler = DeterministicPKBatchSampler(
            labels=labels,
            p=p,
            k=k,
            seed=args.seed,
        )
        return DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=0,
            pin_memory=device.startswith("cuda"),
        )

    # base.run_experiment负责10轮训练、latest/epoch_10 checkpoint和history.json。
    base.build_model = build_dolg_model
    base.build_loaders = build_deterministic_loader
    base.update_summary = lambda **kwargs: None

    grid = []
    for red_probability in args.red_probs:
        tag = probability_tag(red_probability)
        grid.extend([
            {
                "experiment": base.Experiment(
                    f"dolg_redp{tag}_arcface",
                    "none",
                    0.0,
                ),
                "red_probability": float(red_probability),
            },
            {
                "experiment": base.Experiment(
                    f"dolg_redp{tag}_center_lam001",
                    "center",
                    float(args.lambda_center),
                ),
                "red_probability": float(red_probability),
            },
            {
                "experiment": base.Experiment(
                    f"dolg_redp{tag}_supcon_lam01",
                    "supcon",
                    float(args.lambda_supcon),
                ),
                "red_probability": float(red_probability),
            },
        ])

    master_config = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_experiments": len(grid),
        "epochs": args.epochs,
        "seed": args.seed,
        "device": args.device,
        "dataset_root": args.obb_root,
        "mask_root": args.red_mask_root,
        "train_samples": len(train_samples),
        "classes": len(id2label),
        "pk_sampler": {"P": args.p, "K": args.k, "batch": args.p * args.k},
        "dolg": {
            "embedding_dim": args.embedding_dim,
            "local_dim": args.local_dim,
            "fusion_hidden_dim": args.fusion_hidden_dim,
            "gem_p": args.gem_p,
            "gem_learnable": not args.fixed_gem_p,
            "dilation_rates": list(args.dilation_rates),
            "dropout": args.dropout,
        },
        "red_augmentation": {
            "probabilities": list(args.red_probs),
            **red_common_config,
        },
        "losses": {
            "arcface": {"s": args.arc_s, "m": args.arc_m},
            "center": {"lambda": args.lambda_center, "center_lr": args.center_lr},
            "supcon": {
                "lambda": args.lambda_supcon,
                "temperature": args.temperature,
                "projection_hidden_dim": args.projection_hidden_dim,
                "projection_dim": args.projection_dim,
            },
        },
        "optimizer": {"lr": args.lr, "weight_decay": args.weight_decay},
        "experiments": [
            {
                "name": item["experiment"].name,
                "red_probability": item["red_probability"],
                "auxiliary": item["experiment"].auxiliary,
                "auxiliary_weight": item["experiment"].auxiliary_weight,
            }
            for item in grid
        ],
    }
    (output_root / "grid_config.json").write_text(
        json.dumps(master_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("=" * 90)
    logger.info("DOLG红增强×损失交叉实验：共%d组", len(grid))
    logger.info("train=%d | classes=%d | validation=disabled", len(train_samples), len(id2label))
    logger.info("P=%d | K=%d | batch=%d | epochs=%d", args.p, args.k, args.p * args.k, args.epochs)
    logger.info("red_probs=%s", list(args.red_probs))
    logger.info("Center lambda=%s | SupCon lambda=%s", args.lambda_center, args.lambda_supcon)
    logger.info("=" * 90)

    failures = []
    grid_start = time.perf_counter()

    for index, item in enumerate(grid, 1):
        experiment = item["experiment"]
        red_probability = item["red_probability"]
        experiment_dir = output_root / experiment.name
        final_checkpoint = experiment_dir / "checkpoints" / "epoch_10.pth"

        if final_checkpoint.exists() and not args.overwrite:
            logger.info("[%d/%d] SKIP已完成：%s", index, len(grid), experiment.name)
            continue

        red_augment = CowBackRedColorShift(
            image_root=args.obb_root,
            mask_root=args.red_mask_root,
            probability=red_probability,
            **red_common_config,
        )
        train_dataset = CowTrainDatasetNormal(
            samples=train_samples,
            transform=train_transform,
            id2label=id2label,
            red_augment=red_augment,
        )

        experiment_dir.mkdir(parents=True, exist_ok=True)
        experiment_config = {
            "experiment": experiment.name,
            "index": index,
            "total": len(grid),
            "red_augmentation": red_augment.get_config(),
            "auxiliary": experiment.auxiliary,
            "auxiliary_weight": experiment.auxiliary_weight,
            "epochs": args.epochs,
            "P": args.p,
            "K": args.k,
            "batch_size": args.p * args.k,
            "seed": args.seed,
        }
        (experiment_dir / "experiment_config.json").write_text(
            json.dumps(experiment_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info("#" * 90)
        logger.info("[%d/%d] START %s", index, len(grid), experiment.name)
        logger.info(
            "red_prob=%.2f | auxiliary=%s | lambda=%s",
            red_probability,
            experiment.auxiliary,
            experiment.auxiliary_weight,
        )
        logger.info("checkpoint=%s", final_checkpoint)

        run_start = time.perf_counter()
        status = "success"
        error_text = ""

        try:
            base.run_experiment(
                experiment=experiment,
                args=args,
                train_dataset=train_dataset,
                num_classes=len(id2label),
                device=args.device,
                output_root=output_root,
            )
        except Exception as error:
            status = "failed"
            error_text = f"{type(error).__name__}: {error}"
            failures.append(f"{experiment.name}: {error_text}")
            logger.error("FAILED %s", error_text)
            logger.error(traceback.format_exc())
            base.release_memory()

        duration_minutes = (time.perf_counter() - run_start) / 60.0
        red_stats = train_dataset.get_red_augmentation_statistics() or {}

        history_path = experiment_dir / "history.json"
        final_train = {}
        if history_path.exists():
            history_payload = json.loads(history_path.read_text(encoding="utf-8"))
            history = history_payload.get("history", [])
            if history:
                final_train = history[-1]

        result_row = {
            "experiment": experiment.name,
            "red_aug_probability": red_probability,
            "auxiliary": experiment.auxiliary,
            "auxiliary_weight": experiment.auxiliary_weight,
            "epoch": final_train.get("epoch", ""),
            "train_loss": final_train.get("loss", ""),
            "train_arcface_loss": final_train.get("arcface_loss", ""),
            "train_auxiliary_loss_raw": final_train.get("auxiliary_loss_raw", ""),
            "train_auxiliary_loss_weighted": final_train.get("auxiliary_loss_weighted", ""),
            "train_accuracy": final_train.get("classification_accuracy", ""),
            "actual_red_ratio": red_stats.get("actual_ratio", ""),
            "red_total_calls": red_stats.get("total_calls", ""),
            "red_augmented_count": red_stats.get("augmented_count", ""),
            "red_strength_counts": json.dumps(
                red_stats.get("strength_counts", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "duration_minutes": f"{duration_minutes:.3f}",
            "checkpoint": str(final_checkpoint),
            "status": status,
            "error": error_text,
        }
        write_summary_row(summary_path, result_row)

        # 同时把增强统计追加到该实验自己的train.log。
        experiment_logger = logging.getLogger(f"experiment:{experiment.name}")
        experiment_logger.info("red_augmentation_statistics=%s", red_stats)
        experiment_logger.info("duration_minutes=%.3f | status=%s", duration_minutes, status)

        logger.info("END %s | status=%s | minutes=%.2f", experiment.name, status, duration_minutes)
        logger.info("red_stats=%s", red_stats)

        if status == "failed" and not args.continue_on_error:
            raise RuntimeError(f"实验失败并停止：{experiment.name}: {error_text}")

    total_hours = (time.perf_counter() - grid_start) / 3600.0
    logger.info("=" * 90)
    logger.info("GRID FINISHED | total_hours=%.3f | failures=%d", total_hours, len(failures))
    logger.info("summary=%s", summary_path)
    if failures:
        failure_path = output_root / "failures.txt"
        failure_path.write_text("\n".join(failures), encoding="utf-8")
        logger.warning("failure_details=%s", failure_path)
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
