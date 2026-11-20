"""DOLG-ArcFace 奶牛背纹识别训练脚本。

建议放置位置：
    scripts/train_sup_exp003_dolg_arcface.py

数据集、Trainer、验证、断点续训和 checkpoint 逻辑沿用原 ArcFace 工程。
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sup.datasets.dataset_OBB_normal import (
    CowTrainDatasetNormal,
    CowValDatasetFromSplit,
)
from src.sup.datasets.split_cow import get_samples
from src.sup.engine.arcface_trainer import ArcFaceTrainer
from src.sup.models.dolg_arcface_model import DOLGArcFaceCowModel
from src.sup.utils.common import set_seed
from src.sup.utils.transforms import build_transforms

logger = logging.getLogger("cow_arcface")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

print("PYTHON =", sys.executable)
print("NUMPY  =", np.__version__)


def load_checkpoint(
    model,
    optimizer,
    ckpt_path,
    device="cpu",
    resume_only_model=False,
):
    if not ckpt_path:
        return model, optimizer, 1, None

    checkpoint = torch.load(ckpt_path, map_location=device)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    if (not resume_only_model) and ("optimizer" in checkpoint):
        optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    return model, optimizer, start_epoch, checkpoint


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--obb_root",
        type=str,
        default=r"D:\datasets\openset_CowBack\openset_quality_P99.5",
    )

    # 固定默认训练到 150 epoch；Trainer 中没有早停。
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=32)

    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume_only_model", action="store_true")

    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--local_dim", type=int, default=512)
    parser.add_argument("--fusion_hidden_dim", type=int, default=512)
    parser.add_argument("--arc_s", type=float, default=30.0)
    parser.add_argument("--arc_m", type=float, default=0.25)

    parser.add_argument("--gem_p", type=float, default=3.0)
    parser.add_argument(
        "--fixed_gem_p",
        action="store_true",
        help="设置后固定 GeM 的 p；默认 p 可学习。",
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
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp003_7.25_arcface_DOLG_open995",
    )
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)

    # 与现有 Trainer 的 L5 可选逻辑兼容；默认关闭，不改变原训练损失。
    parser.add_argument("--return_theta", action="store_true")
    parser.add_argument("--lambda_intra", type=float, default=0.02)

    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU.")
        device = "cpu"

    train_samples = get_samples(args.obb_root, "seen_train")
    val_samples = get_samples(args.obb_root, "seen_val")

    logger.info(
        f"seen train samples: {len(train_samples)} | "
        f"seen val samples: {len(val_samples)}"
    )

    # 严格沿用原脚本，不额外修改数据增强逻辑。
    train_transform = build_transforms(train=False)
    val_transform = build_transforms(train=False)

    train_cow_ids = sorted(list(set(x["cow_id"] for x in train_samples)))
    id2label = {cid: i for i, cid in enumerate(train_cow_ids)}
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
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = DOLGArcFaceCowModel(
        num_classes=num_train_classes,
        embedding_dim=args.embedding_dim,
        local_dim=args.local_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        s=args.arc_s,
        m=args.arc_m,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
        gem_p=args.gem_p,
        gem_learn_p=not args.fixed_gem_p,
        dilation_rates=tuple(args.dilation_rates),
        dropout=args.dropout,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

    start_epoch = 1
    resume_ckpt = None

    if args.resume:
        model, optimizer, start_epoch, resume_ckpt = load_checkpoint(
            model=model,
            optimizer=optimizer,
            ckpt_path=args.resume,
            device=device,
            resume_only_model=args.resume_only_model,
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
        return_theta=args.return_theta,
        lambda_intra=args.lambda_intra,
    )

    if resume_ckpt is not None:
        trainer.load_state_from_ckpt(resume_ckpt)

    logger.info(
        "DOLG config | "
        f"embedding_dim={args.embedding_dim} | "
        f"local_dim={args.local_dim} | "
        f"dilation_rates={tuple(args.dilation_rates)} | "
        f"GeM p={args.gem_p} | "
        f"GeM learnable={not args.fixed_gem_p}"
    )

    trainer.fit(args.epochs, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
