# -*- coding: utf-8 -*-

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np

import torch
import torch.optim as optim

from torch.utils.data import DataLoader


# =========================================================
# 项目路径
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.sup.datasets.split_cow import (
    get_samples
)

from src.sup.datasets.dataset_OBB_normal import (
    CowTrainDatasetNormal,
    CowValDatasetFromSplit,
)

from src.sup.models.arcface_model import (
    ArcFaceCowModel
)

from src.sup.losses.supcon_loss import (
    SupConLoss
)

from src.sup.sampler.pk_sampler import (
    PKSampler
)

from src.sup.engine.arcface_supcon_trainer import (
    ArcFaceSupConTrainer,
    SupConProjectionHead,
)

from src.sup.utils.common import (
    set_seed
)

from src.sup.utils.transforms import (
    build_transforms
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s - "
        "%(levelname)s - "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "arcface_supcon"
)


def safe_torch_load(
    checkpoint_path: str,
    device: str,
):
    """
    同时兼容新旧PyTorch版本。
    """

    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location=device,
        )


def load_checkpoint(
    model,
    projector,
    optimizer,
    checkpoint_path: str,
    device: str,
):
    checkpoint = safe_torch_load(
        checkpoint_path,
        device,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "SupCon checkpoint必须为字典格式"
        )

    if "model" not in checkpoint:
        raise KeyError(
            "checkpoint中缺少model"
        )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    if "projector" in checkpoint:
        projector.load_state_dict(
            checkpoint["projector"],
            strict=True,
        )
    else:
        logger.warning(
            "checkpoint中不存在projector，"
            "投影头将重新初始化"
        )

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    start_epoch = (
        int(
            checkpoint.get(
                "epoch",
                0,
            )
        )
        + 1
    )

    logger.info(
        f"Resume from epoch {start_epoch}"
    )

    return start_epoch, checkpoint


def main():

    parser = argparse.ArgumentParser()

    # =========================
    # 数据参数
    # =========================

    parser.add_argument(
        "--obb_root",
        type=str,
        default=(
            r"D:\code_project\triplet_code"
            r"\data\dataset_OBB"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    # =========================
    # PKSampler参数
    # 实际batch size = P * K
    # =========================

    parser.add_argument(
        "--P",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--K",
        type=int,
        default=4,
    )

    # =========================
    # ArcFace参数
    # =========================

    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--arc_s",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--arc_m",
        type=float,
        default=0.25,
    )

    # =========================
    # SupCon参数
    # =========================

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--lambda_supcon",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--projection_hidden_dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--projection_dim",
        type=int,
        default=128,
    )

    # =========================
    # 优化参数
    # =========================

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--wd",
        type=float,
        default=1e-4,
    )

    # =========================
    # 运行参数
    # =========================

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default=(
            "exp006_arcface_supcon"
            "_P4K4_t007_lam01"
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="",
    )

    args = parser.parse_args()

    # =====================================================
    # 参数检查
    # =====================================================

    if args.K < 2:
        raise ValueError(
            "SupCon要求K >= 2，"
            "否则每个anchor没有正样本"
        )

    if args.P < 2:
        raise ValueError(
            "P至少为2，"
            "否则batch中没有异类负样本"
        )

    effective_batch_size = (
        args.P
        * args.K
    )

    # =====================================================
    # 随机种子
    # =====================================================

    set_seed(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    # =====================================================
    # device
    # =====================================================

    device = args.device

    if (
        device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        logger.warning(
            "CUDA不可用，切换到CPU"
        )
        device = "cpu"

    logger.info(
        f"device={device}"
    )

    logger.info(
        f"PKSampler: "
        f"P={args.P}, "
        f"K={args.K}, "
        f"batch_size={effective_batch_size}"
    )

    logger.info(
        f"SupCon: "
        f"temperature={args.temperature}, "
        f"lambda={args.lambda_supcon}"
    )

    # =====================================================
    # 数据划分
    # =====================================================

    train_samples = get_samples(
        args.obb_root,
        "seen_train",
    )

    val_samples = get_samples(
        args.obb_root,
        "seen_val",
    )

    logger.info(
        f"train_samples={len(train_samples)}"
    )

    logger.info(
        f"val_samples={len(val_samples)}"
    )

    train_cow_ids = sorted(
        {
            sample["cow_id"]
            for sample in train_samples
        }
    )

    id2label = {
        cow_id: label
        for label, cow_id in enumerate(
            train_cow_ids
        )
    }

    num_classes = len(id2label)

    if args.P > num_classes:
        raise ValueError(
            f"P={args.P}超过训练类别数"
            f"{num_classes}"
        )

    logger.info(
        f"num_classes={num_classes}"
    )

    # =====================================================
    # Transform
    # 与ArcFace baseline保持一致，不额外增强
    # =====================================================

    train_transform = build_transforms(
        train=False
    )

    val_transform = build_transforms(
        train=False
    )

    # =====================================================
    # Dataset
    # =====================================================

    train_dataset = CowTrainDatasetNormal(
        samples=train_samples,
        transform=train_transform,
        id2label=id2label,
    )

    val_dataset = CowValDatasetFromSplit(
        samples=val_samples,
        transform=val_transform,
        id2label=id2label,
    )

    # 必须从dataset内部读取label，
    # 保证顺序和Dataset索引完全一致
    train_labels = [
        int(label)
        for _, label in train_dataset.data
    ]

    # =====================================================
    # PKSampler
    # =====================================================

    pk_sampler = PKSampler(
        labels=train_labels,
        P=args.P,
        K=args.K,
    )

    train_loader = DataLoader(
        train_dataset,

        # PKSampler已经返回完整batch
        batch_sampler=pk_sampler,

        num_workers=args.num_workers,

        pin_memory=(
            device.startswith("cuda")
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.startswith("cuda")
        ),
    )

    logger.info(
        f"train_batches_per_epoch="
        f"{len(train_loader)}"
    )

    # =====================================================
    # ArcFace模型
    # =====================================================

    model = ArcFaceCowModel(
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        s=args.arc_s,
        m=args.arc_m,
        pretrained=True,
        freeze_backbone=False,
    )

    # =====================================================
    # SupCon投影头
    # =====================================================

    projector = SupConProjectionHead(
        input_dim=args.embedding_dim,
        hidden_dim=(
            args.projection_hidden_dim
        ),
        output_dim=args.projection_dim,
    )

    # =====================================================
    # SupCon Loss
    # =====================================================

    supcon_loss = SupConLoss(
        temperature=args.temperature,
    )

    # =====================================================
    # Optimizer
    # ArcFace主模型和投影头一起训练
    # =====================================================

    optimizer = optim.AdamW(
        list(model.parameters())
        + list(projector.parameters()),

        lr=args.lr,
        weight_decay=args.wd,
    )

    # =====================================================
    # Resume
    # =====================================================

    start_epoch = 1
    resume_checkpoint = None

    if args.resume:

        start_epoch, resume_checkpoint = (
            load_checkpoint(
                model=model,
                projector=projector,
                optimizer=optimizer,
                checkpoint_path=args.resume,
                device=device,
            )
        )

    # =====================================================
    # Trainer
    # =====================================================

    trainer = ArcFaceSupConTrainer(
        model=model,
        projector=projector,
        supcon_loss=supcon_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        lambda_supcon=(
            args.lambda_supcon
        ),
        log_dir=args.log_dir,
        exp_name=args.exp_name,
        seed=args.seed,
        max_pairs_per_class=2000,
    )

    if resume_checkpoint is not None:
        trainer.load_state_from_checkpoint(
            resume_checkpoint
        )

    trainer.fit(
        epochs=args.epochs,
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    main()