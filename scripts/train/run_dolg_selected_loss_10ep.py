# -*- coding: utf-8 -*-
"""DOLG 三组主实验：ArcFace、Center(0.01)、SupCon(0.1)。

固定设置：
- 10 epochs；
- seen_train only，不运行 validation；
- PKSampler: P=16, K=4, batch size=64；
- 不使用红色增强；
- 每组从相同 ImageNet 预训练 ResNet50-DOLG 初始化开始。
"""

import argparse
import importlib.util
import logging
import random
import sys
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


class DeterministicPKBatchSampler(Sampler):
    """与红增强网格使用相同、且不受数据增强随机数影响的PK采样。"""

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


def load_training_base():
    spec = importlib.util.spec_from_file_location(
        "loss_sweep_training_base",
        BASE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载基础训练脚本：{BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description="顺序训练 DOLG ArcFace/Center/SupCon 三组主实验。"
    )
    parser.add_argument(
        "--obb_root",
        type=str,
        default=r"D:\datasets\cow_recognition8.3\train_dataset",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            r"D:\code_project\cow_recognition\outputs"
            r"\dolg_selected_loss_10ep"
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--p", type=int, default=16)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--local_dim", type=int, default=512)
    parser.add_argument("--fusion_hidden_dim", type=int, default=512)
    parser.add_argument("--arc_s", type=float, default=30.0)
    parser.add_argument("--arc_m", type=float, default=0.25)
    parser.add_argument("--gem_p", type=float, default=3.0)
    parser.add_argument(
        "--fixed_gem_p",
        action="store_true",
        help="固定 GeM p；默认允许学习。",
    )
    parser.add_argument(
        "--dilation_rates",
        type=int,
        nargs=3,
        default=(1, 2, 3),
        metavar=("D1", "D2", "D3"),
    )
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--center_lr", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--projection_hidden_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=128)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="重新训练已经存在 epoch_10.pth 的实验。",
    )
    parser.add_argument(
        "--continue_on_error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认某组失败后继续下一组。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epochs != 10:
        raise ValueError("本实验固定比较第10个epoch，请保持 --epochs 10。")
    if args.p * args.k != 64:
        raise ValueError(
            f"本轮公平比较固定batch=64，当前P*K={args.p * args.k}。"
        )
    if args.p < 2 or args.k < 2:
        raise ValueError("SupCon要求P>=2且K>=2。")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；DOLG训练不自动切换到CPU。")

    base = load_training_base()
    from src.models.dolg_arcface_model import DOLGArcFaceCowModel

    experiments = (
        base.Experiment("dolg_arcface", "none", 0.0),
        base.Experiment("dolg_arcface_center_lam001", "center", 0.01),
        base.Experiment("dolg_arcface_supcon_lam01", "supcon", 0.1),
    )

    def build_dolg_model(
        num_classes: int,
        embedding_dim: int,
        arc_s: float,
        arc_m: float,
    ):
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

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dolg_summary_path = output_root / "dolg_selected_loss_summary.csv"

    original_update_summary = base.update_summary

    def update_dolg_summary(
        summary_path,
        experiment,
        checkpoint_path,
        train_metrics,
    ):
        return original_update_summary(
            summary_path=dolg_summary_path,
            experiment=experiment,
            checkpoint_path=checkpoint_path,
            train_metrics=train_metrics,
        )

    base.build_model = build_dolg_model
    base.update_summary = update_dolg_summary

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
            num_workers=num_workers,
            pin_memory=device.startswith("cuda"),
        )

    base.build_loaders = build_deterministic_loader

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("dolg_selected_loss")

    base.set_seed(args.seed)
    train_dataset, id2label = base.build_datasets(args.obb_root)

    logger.info("项目根目录：%s", PROJECT_ROOT)
    logger.info("训练集：%s", args.obb_root)
    logger.info("train=%d | classes=%d | validation=disabled", len(train_dataset), len(id2label))
    logger.info(
        "DOLG: local_dim=%d | fusion_hidden_dim=%d | dilation=%s | GeM learnable=%s",
        args.local_dim,
        args.fusion_hidden_dim,
        tuple(args.dilation_rates),
        not args.fixed_gem_p,
    )
    logger.info("PKSampler: P=%d | K=%d | batch=%d", args.p, args.k, args.p * args.k)

    failures = []
    for index, experiment in enumerate(experiments, 1):
        logger.info("[%d/%d] 即将运行：%s", index, len(experiments), experiment.name)
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
            message = f"{experiment.name}: {type(error).__name__}: {error}"
            failures.append(message)
            logger.error(message)
            logger.error(traceback.format_exc())
            base.release_memory()
            if not args.continue_on_error:
                raise

    if failures:
        failure_path = output_root / "failures.txt"
        failure_path.write_text("\n".join(failures), encoding="utf-8")
        logger.warning("存在失败实验：%s", failure_path)
    else:
        logger.info("DOLG三组实验全部完成。")
        logger.info("汇总结果：%s", dolg_summary_path)


if __name__ == "__main__":
    main()
