# -*- coding: utf-8 -*-
"""
阶段一实验：
    官方 DINO-ResNet50 初始化
        -> seen_train 上 ArcFace 监督训练
        -> 使用原 ArcFace pair_dis 测试 unseen_cows

建议位置：
    scripts/train_sup_exp007_dino_arcface.py

注意：
1. 这不是在本地重新训练 DINO。
2. 不使用 src/unsup。
3. 模型结构、数据、ArcFace、Trainer 均沿用现有 baseline。
4. 第一轮默认使用与 baseline 相同的统一学习率 3e-4，
   使唯一变量尽可能只是 backbone 初始化。
"""

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )
)

from src.sup.datasets.dataset_OBB_normal import (
    CowTrainDatasetNormal,
    CowValDatasetFromSplit,
)
from src.sup.datasets.split_cow import get_samples
from src.sup.engine.arcface_trainer import ArcFaceTrainer
from src.sup.models.arcface_model import ArcFaceCowModel
from src.sup.models.dino_weight_loader import (
    load_dino_resnet50_into_arcface,
    split_backbone_and_head_parameters,
)
from src.sup.utils.common import set_seed
from src.sup.utils.transforms import build_transforms


logger = logging.getLogger("cow_dino_arcface")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_checkpoint(
    model,
    optimizer,
    ckpt_path,
    device="cpu",
    resume_only_model=False,
):
    if not ckpt_path:
        return model, optimizer, 1, None

    try:
        checkpoint = torch.load(
            ckpt_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            ckpt_path,
            map_location=device,
        )

    if "model" in checkpoint:
        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
    else:
        model.load_state_dict(
            checkpoint,
            strict=True,
        )

    if (
        not resume_only_model
        and "optimizer" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    start_epoch = int(
        checkpoint.get("epoch", 0)
    ) + 1

    return (
        model,
        optimizer,
        start_epoch,
        checkpoint,
    )


def build_optimizer(
    model,
    lr: float,
    wd: float,
    differential_lr: bool,
    backbone_lr: float,
):
    if not differential_lr:
        logger.info(
            "Optimizer: unified lr | "
            f"all parameters lr={lr}"
        )

        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
        )

    backbone_parameters, head_parameters = (
        split_backbone_and_head_parameters(model)
    )

    logger.info(
        "Optimizer: differential lr | "
        f"backbone_lr={backbone_lr} | "
        f"head_lr={lr}"
    )

    return optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": backbone_lr,
            },
            {
                "params": head_parameters,
                "lr": lr,
            },
        ],
        weight_decay=wd,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--obb_root",
        type=str,
        default=(
            r"D:\code_project\triplet_code"
            r"\data\dataset_OBB"
        ),
    )

    parser.add_argument(
        "--dino_weights",
        type=str,
        default=(
            r"D:\code_project\triplet_code"
            r"\weights\dino"
            r"\dino_resnet50_pretrain.pth"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=32,
    )

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

    # 第一轮严格对照使用统一 lr=3e-4。
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

    # 第二轮消融实验才开启。
    parser.add_argument(
        "--differential_lr",
        action="store_true",
    )
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=3e-5,
    )

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
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp007_dino_r50_arcface",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="",
    )
    parser.add_argument(
        "--resume_only_model",
        action="store_true",
    )

    args = parser.parse_args()

    print("PYTHON =", sys.executable)
    print("TORCH  =", torch.__version__)
    print("NUMPY  =", np.__version__)

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device

    if (
        device == "cuda"
        and not torch.cuda.is_available()
    ):
        logger.warning(
            "CUDA unavailable, switching to CPU."
        )
        device = "cpu"

    train_samples = get_samples(
        args.obb_root,
        "seen_train",
    )
    val_samples = get_samples(
        args.obb_root,
        "seen_val",
    )

    logger.info(
        f"seen train samples: {len(train_samples)} | "
        f"seen val samples: {len(val_samples)}"
    )

    # 严格沿用 baseline，不额外修改数据增强。
    train_transform = build_transforms(
        train=False
    )
    val_transform = build_transforms(
        train=False
    )

    train_cow_ids = sorted(
        {
            sample["cow_id"]
            for sample in train_samples
        }
    )

    id2label = {
        cow_id: index
        for index, cow_id in enumerate(
            train_cow_ids
        )
    }

    num_train_classes = len(id2label)

    train_ds = CowTrainDatasetNormal(
        samples=train_samples,
        transform=train_transform,
        id2label=id2label,
    )

    val_ds = CowValDatasetFromSplit(
        samples=val_samples,
        id2label=id2label,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
            device == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device == "cuda"
        ),
    )

    # 先建立与 baseline 完全相同的 ArcFace 模型，
    # 但不加载 ImageNet 监督权重。
    model = ArcFaceCowModel(
        num_classes=num_train_classes,
        embedding_dim=args.embedding_dim,
        s=args.arc_s,
        m=args.arc_m,
        pretrained=False,
        freeze_backbone=False,
    )

    # 只替换 ResNet50 backbone 初始化。
    load_dino_resnet50_into_arcface(
        model=model,
        dino_weights_path=args.dino_weights,
    )

    optimizer = build_optimizer(
        model=model,
        lr=args.lr,
        wd=args.wd,
        differential_lr=args.differential_lr,
        backbone_lr=args.backbone_lr,
    )

    start_epoch = 1
    resume_ckpt = None

    if args.resume:
        (
            model,
            optimizer,
            start_epoch,
            resume_ckpt,
        ) = load_checkpoint(
            model=model,
            optimizer=optimizer,
            ckpt_path=args.resume,
            device=device,
            resume_only_model=(
                args.resume_only_model
            ),
        )

    trainer = ArcFaceTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        log_dir=args.log_dir,
        exp_name=args.exp_name,
        seed=args.seed,
        return_theta=False,
        lambda_intra=0.02,
    )

    if resume_ckpt is not None:
        trainer.load_state_from_ckpt(
            resume_ckpt
        )

    logger.info(
        "Experiment config | "
        "init=DINO-R50 | "
        f"classes={num_train_classes} | "
        f"embedding_dim={args.embedding_dim} | "
        f"ArcFace(s={args.arc_s}, m={args.arc_m}) | "
        f"epochs={args.epochs} | "
        f"batch={args.batch_size}"
    )

    trainer.fit(
        args.epochs,
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    main()
