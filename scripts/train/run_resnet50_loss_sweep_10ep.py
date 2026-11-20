# -*- coding: utf-8 -*-
"""
ResNet50损失函数与辅助权重顺序实验。

按照以下顺序自动训练，每组训练10个epoch：
1. ArcFace + Center Loss，lambda = 0.001
2. ArcFace + Center Loss，lambda = 0.005
3. ArcFace + Center Loss，lambda = 0.01
4. ArcFace + SupCon，lambda = 0.05
5. ArcFace + SupCon，lambda = 0.1
6. ArcFace + SupCon，lambda = 0.2

公平性设置：
- 每组均从相同的ImageNet预训练ResNet50重新开始；
- seed=42；
- 使用split.json中的同一seen_train划分，不运行seen_val；
- 同一PKSampler：P=16、K=4，有效batch size=64；
- ArcFace s=30、m=0.25；
- embedding_dim=128；
- 不使用红色增强，也不使用额外图像增强；
- 训练阶段只记录train指标并保存第10个epoch；
- 最终统一在相同的unseen原始集和hard集上比较。
"""

import argparse
import csv
import gc
import json
import logging
import random
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir():
            return parent
    raise RuntimeError("找不到包含src目录的项目根目录。")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset_OBB_normal import CowTrainDatasetNormal
from src.datasets.split import get_samples
from src.engine.arcface_supcon_trainer import SupConProjectionHead
from src.losses.center_loss import CenterLoss
from src.losses.supcon_loss import SupConLoss
from src.models.arcface_model import ArcFaceCowModel
from src.sampler.pk_sampler import PKSampler
from src.utils.transforms import build_transforms


LOGGER = logging.getLogger("resnet50_loss_sweep")


@dataclass(frozen=True)
class Experiment:
    name: str
    auxiliary: str
    auxiliary_weight: float


EXPERIMENTS: Tuple[Experiment, ...] = (
    Experiment("resnet50_arcface_center_lam0001", "center", 0.001),
    Experiment("resnet50_arcface_center_lam0005", "center", 0.005),
    Experiment("resnet50_arcface_center_lam001", "center", 0.01),
    Experiment("resnet50_arcface_supcon_lam005", "supcon", 0.05),
    Experiment("resnet50_arcface_supcon_lam01", "supcon", 0.1),
    Experiment("resnet50_arcface_supcon_lam02", "supcon", 0.2),
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def create_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"experiment:{log_path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_path,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def build_datasets(obb_root: str):
    train_samples = get_samples(obb_root, "seen_train")

    train_cow_ids = sorted(
        {str(sample["cow_id"]) for sample in train_samples}
    )
    id2label = {
        cow_id: label
        for label, cow_id in enumerate(train_cow_ids)
    }

    # 与已有baseline一致，不增加红色增强和额外训练增强。
    train_transform = build_transforms(train=False)

    train_dataset = CowTrainDatasetNormal(
        samples=train_samples,
        transform=train_transform,
        id2label=id2label,
        red_augment=None,
    )
    return train_dataset, id2label


def build_loaders(
    train_dataset,
    p: int,
    k: int,
    num_workers: int,
    device: str,
):
    train_labels = [
        int(label)
        for _, label in train_dataset.data
    ]

    sampler = PKSampler(
        labels=train_labels,
        P=p,
        K=k,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    return train_loader


def build_model(
    num_classes: int,
    embedding_dim: int,
    arc_s: float,
    arc_m: float,
):
    # hidden_dims=(1024,)与原ResNet50+ArcFace baseline的
    # 2048 -> 1024 -> embedding_dim映射头保持一致。
    return ArcFaceCowModel(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        s=arc_s,
        m=arc_m,
        pretrained=True,
        freeze_backbone=False,
        hidden_dims=(1024,),
        dropout=0.3,
        use_gem=False,
    )


def build_auxiliary_components(
    experiment: Experiment,
    num_classes: int,
    embedding_dim: int,
    device: str,
    projection_hidden_dim: int,
    projection_dim: int,
    temperature: float,
):
    center_loss = None
    projector = None
    supcon_loss = None

    if experiment.auxiliary == "center":
        center_loss = CenterLoss(
            num_classes=num_classes,
            feat_dim=embedding_dim,
            device=device,
        )

    elif experiment.auxiliary == "supcon":
        projector = SupConProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
        ).to(device)
        supcon_loss = SupConLoss(
            temperature=temperature,
        ).to(device)

    return center_loss, projector, supcon_loss


def build_optimizer(
    model,
    center_loss,
    projector,
    model_lr: float,
    center_lr: float,
    weight_decay: float,
):
    if center_loss is not None:
        return optim.AdamW(
            [
                {
                    "params": model.parameters(),
                    "lr": model_lr,
                    "weight_decay": weight_decay,
                },
                {
                    "params": center_loss.parameters(),
                    "lr": center_lr,
                    "weight_decay": 0.0,
                },
            ]
        )

    parameters = list(model.parameters())
    if projector is not None:
        parameters += list(projector.parameters())

    return optim.AdamW(
        parameters,
        lr=model_lr,
        weight_decay=weight_decay,
    )


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device: str,
    experiment: Experiment,
    center_loss=None,
    projector=None,
    supcon_loss=None,
):
    model.train()
    if center_loss is not None:
        center_loss.train()
    if projector is not None:
        projector.train()

    arcface_criterion = nn.CrossEntropyLoss()
    total_samples = 0
    total_correct = 0
    total_loss = 0.0
    total_arcface = 0.0
    total_auxiliary = 0.0
    total_weighted_auxiliary = 0.0

    progress = tqdm(
        train_loader,
        desc=f"TRAIN {experiment.name}",
        leave=False,
    )

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        embeddings, logits = model(images, labels)
        loss_arcface = arcface_criterion(logits, labels)

        if experiment.auxiliary == "center":
            loss_auxiliary = center_loss(embeddings, labels)
        elif experiment.auxiliary == "supcon":
            projected = projector(embeddings)
            loss_auxiliary = supcon_loss(projected, labels)
        else:
            loss_auxiliary = torch.zeros((), device=device)

        weighted_auxiliary = (
            experiment.auxiliary_weight
            * loss_auxiliary
        )
        loss = loss_arcface + weighted_auxiliary

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_samples += batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_loss += loss.item() * batch_size
        total_arcface += loss_arcface.item() * batch_size
        total_auxiliary += loss_auxiliary.item() * batch_size
        total_weighted_auxiliary += (
            weighted_auxiliary.item() * batch_size
        )

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            arc=f"{total_arcface / total_samples:.4f}",
            aux=f"{total_weighted_auxiliary / total_samples:.4f}",
        )

    return {
        "loss": total_loss / total_samples,
        "arcface_loss": total_arcface / total_samples,
        "auxiliary_loss_raw": total_auxiliary / total_samples,
        "auxiliary_loss_weighted": (
            total_weighted_auxiliary / total_samples
        ),
        "classification_accuracy": total_correct / total_samples,
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    model,
    optimizer,
    experiment: Experiment,
    args,
    history,
    center_loss=None,
    projector=None,
):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "experiment": asdict(experiment),
        "run_config": vars(args),
        "history": history,
    }

    if center_loss is not None:
        checkpoint["center_loss"] = center_loss.state_dict()
    if projector is not None:
        checkpoint["projector"] = projector.state_dict()

    torch.save(checkpoint, path)


def update_summary(
    summary_path: Path,
    experiment: Experiment,
    checkpoint_path: Path,
    train_metrics: Dict[str, float],
):
    fieldnames = [
        "experiment",
        "auxiliary",
        "auxiliary_weight",
        "checkpoint",
        "epoch",
        "train_loss",
        "train_arcface_loss",
        "train_auxiliary_loss_raw",
        "train_auxiliary_loss_weighted",
        "train_accuracy",
    ]

    rows: List[Dict[str, str]] = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))

    new_row = {
        "experiment": experiment.name,
        "auxiliary": experiment.auxiliary,
        "auxiliary_weight": str(experiment.auxiliary_weight),
        "checkpoint": str(checkpoint_path),
        "epoch": "10",
        "train_loss": f"{train_metrics['loss']:.8f}",
        "train_arcface_loss": f"{train_metrics['arcface_loss']:.8f}",
        "train_auxiliary_loss_raw": (
            f"{train_metrics['auxiliary_loss_raw']:.8f}"
        ),
        "train_auxiliary_loss_weighted": (
            f"{train_metrics['auxiliary_loss_weighted']:.8f}"
        ),
        "train_accuracy": (
            f"{train_metrics['classification_accuracy']:.8f}"
        ),
    }

    rows = [
        row for row in rows
        if row.get("experiment") != experiment.name
    ]
    rows.append(new_row)

    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    experiment: Experiment,
    args,
    train_dataset,
    num_classes: int,
    device: str,
    output_root: Path,
):
    experiment_dir = output_root / experiment.name
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = checkpoint_dir / "epoch_10.pth"

    if final_checkpoint.exists() and not args.overwrite:
        LOGGER.info(
            "[SKIP] %s：已经存在%s",
            experiment.name,
            final_checkpoint,
        )
        return

    experiment_logger = create_logger(experiment_dir / "train.log")
    experiment_logger.info("=" * 72)
    experiment_logger.info("开始实验：%s", experiment.name)
    experiment_logger.info("配置：%s", asdict(experiment))
    experiment_logger.info(
        "PKSampler: P=%d, K=%d, effective_batch=%d",
        args.p,
        args.k,
        args.p * args.k,
    )

    # 每一组重新设置种子，确保初始化和采样序列具有可比性。
    set_seed(args.seed)

    train_loader = build_loaders(
        train_dataset=train_dataset,
        p=args.p,
        k=args.k,
        num_workers=args.num_workers,
        device=device,
    )

    model = build_model(
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        arc_s=args.arc_s,
        arc_m=args.arc_m,
    ).to(device)

    center_loss, projector, supcon_loss = build_auxiliary_components(
        experiment=experiment,
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        device=device,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dim=args.projection_dim,
        temperature=args.temperature,
    )

    optimizer = build_optimizer(
        model=model,
        center_loss=center_loss,
        projector=projector,
        model_lr=args.lr,
        center_lr=args.center_lr,
        weight_decay=args.weight_decay,
    )

    history = []
    final_train_metrics = None

    for epoch in range(1, args.epochs + 1):
        experiment_logger.info(
            "Epoch %d/%d",
            epoch,
            args.epochs,
        )
        final_train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            experiment=experiment,
            center_loss=center_loss,
            projector=projector,
            supcon_loss=supcon_loss,
        )
        history.append({
            "epoch": epoch,
            **final_train_metrics,
        })
        experiment_logger.info(
            "train | loss=%.6f | arc=%.6f | aux_raw=%.6f | "
            "aux_weighted=%.6f | acc=%.6f",
            final_train_metrics["loss"],
            final_train_metrics["arcface_loss"],
            final_train_metrics["auxiliary_loss_raw"],
            final_train_metrics["auxiliary_loss_weighted"],
            final_train_metrics["classification_accuracy"],
        )

        # 保存latest，意外中断时至少保留最近一轮；正式比较仍使用epoch_10。
        latest_path = checkpoint_dir / "latest.pth"
        save_checkpoint(
            path=latest_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            experiment=experiment,
            args=args,
            history=history,
            center_loss=center_loss,
            projector=projector,
        )

    save_checkpoint(
        path=final_checkpoint,
        epoch=args.epochs,
        model=model,
        optimizer=optimizer,
        experiment=experiment,
        args=args,
        history=history,
        center_loss=center_loss,
        projector=projector,
    )

    with (experiment_dir / "history.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "experiment": asdict(experiment),
                "history": history,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    update_summary(
        summary_path=output_root / "resnet50_loss_sweep_summary.csv",
        experiment=experiment,
        checkpoint_path=final_checkpoint,
        train_metrics=final_train_metrics,
    )

    experiment_logger.info("完成实验：%s", experiment.name)

    del model, optimizer, train_loader
    if center_loss is not None:
        del center_loss
    if projector is not None:
        del projector
    if supcon_loss is not None:
        del supcon_loss
    release_memory()


def parse_args():
    parser = argparse.ArgumentParser(
        description="顺序运行ResNet50 ArcFace/Center/SupCon权重实验"
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
            r"\resnet50_loss_sweep_10ep"
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--p", type=int, default=16)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--arc_s", type=float, default=30.0)
    parser.add_argument("--arc_m", type=float, default=0.25)
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
        help="重新训练已经存在epoch_10.pth的实验。",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="某组失败时记录错误并继续下一组；默认失败即停止。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.epochs != 10:
        raise ValueError("本实验脚本固定用于epoch=10的比较，请保持--epochs 10。")
    if args.p * args.k != 64:
        raise ValueError(
            f"要求有效batch size=64，但当前P*K={args.p * args.k}。"
        )
    if args.k < 2:
        raise ValueError("SupCon要求K>=2。")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；七组训练不建议自动切换到CPU。")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    set_seed(args.seed)
    train_dataset, id2label = build_datasets(args.obb_root)

    LOGGER.info("项目根目录：%s", PROJECT_ROOT)
    LOGGER.info("数据集：%s", args.obb_root)
    LOGGER.info("train=%d | validation=disabled", len(train_dataset))
    LOGGER.info("classes=%d", len(id2label))
    LOGGER.info(
        "共%d组实验，每组10 epochs，P=%d，K=%d，batch=%d",
        len(EXPERIMENTS),
        args.p,
        args.k,
        args.p * args.k,
    )

    failures = []
    for index, experiment in enumerate(EXPERIMENTS, 1):
        LOGGER.info(
            "[%d/%d] 即将运行：%s",
            index,
            len(EXPERIMENTS),
            experiment.name,
        )
        try:
            run_experiment(
                experiment=experiment,
                args=args,
                train_dataset=train_dataset,
                num_classes=len(id2label),
                device=device,
                output_root=output_root,
            )
        except Exception as error:
            failure_text = (
                f"{experiment.name}: {type(error).__name__}: {error}"
            )
            failures.append(failure_text)
            LOGGER.error(failure_text)
            LOGGER.error(traceback.format_exc())
            release_memory()
            if not args.continue_on_error:
                raise

    if failures:
        failure_path = output_root / "failures.txt"
        failure_path.write_text("\n".join(failures), encoding="utf-8")
        LOGGER.warning("存在失败实验，详情：%s", failure_path)
    else:
        LOGGER.info("全部ResNet50实验完成。")
        LOGGER.info(
            "汇总结果：%s",
            output_root / "resnet50_loss_sweep_summary.csv",
        )


if __name__ == "__main__":
    main()
