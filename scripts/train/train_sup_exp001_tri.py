import argparse
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.sup.datasets.dataset import (
    CowTrainDataset,
    CowTrainDatasetNormal,
    CowValAllBucketsDataset,
    CowValAbnormalDataset,
    split_cow_ids,
)
from src.sup.engine.triplet_trainer import Trainer
from src.sup.losses.triplet_loss import BatchHardTripletLoss
from src.sup.models.triplet_model import TripletCowModelWithCE
from src.sup.sampler.pk_sampler import PKBatchSampler
from src.sup.utils.common import set_seed
from src.sup.utils.transforms import build_transforms

logger = logging.getLogger("cow_newid_outlier")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=r"D:\code_project\triplet_code\data\data_supervised_output_cleanhardnoisy")
    parser.add_argument("--obb_root", type=str, default=r"D:\code_project\triplet_code\data\dataset_OBB")
    parser.add_argument("--train_root_normal", type=str, default=r"D:\code_project\triplet_code\data\dataset_OBB")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--val_batch_size", type=int, default=64)

    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--ce_weight", type=float, default=1.0)

    parser.add_argument("--train_source", type=str, default="normal", choices=["cleanhard", "normal"])
    parser.add_argument("--prefer_clean_hard_pair", action="store_true")

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--exp_name", type=str, default="triLoss_obb_emb128_Noenhence_m0.3")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--q_threshold", type=float, default=0.9)
    parser.add_argument("--clean_calib_ratio", type=float, default=0.5)
    parser.add_argument("--early_patience", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU.")
        device = "cpu"

    train_cows, val_cows = split_cow_ids(args.data_root, val_ratio=args.val_ratio, seed=args.seed)
    logger.info(f"Total cows: {len(train_cows) + len(val_cows)} | train cows: {len(train_cows)} | val cows: {len(val_cows)}")

    transform = build_transforms(img_size=224)

    val_ds_normal = CowValAllBucketsDataset(args.data_root, val_cows, transform=transform)
    val_ds_ab = CowValAbnormalDataset(args.obb_root, val_cows, transform=transform)

    P, K = 4, 2
    if args.train_source == "cleanhard":
        train_ds = CowTrainDataset(args.data_root, train_cows, transform=transform)
        train_sampler = PKBatchSampler(
            train_ds,
            P=P,
            K=K,
            steps_per_epoch=args.steps_per_epoch,
            seed=args.seed,
            prefer_clean_hard_pair=args.prefer_clean_hard_pair,
        )
    else:
        train_ds = CowTrainDatasetNormal(args.train_root_normal, train_cows, transform=transform)
        train_sampler = PKBatchSampler(
            train_ds,
            P=P,
            K=K,
            steps_per_epoch=args.steps_per_epoch,
            seed=args.seed,
            prefer_clean_hard_pair=False,
        )

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader_normal = DataLoader(val_ds_normal, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_loader_abnormal = DataLoader(val_ds_ab, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    num_train_classes = len(train_cows)
    model = TripletCowModelWithCE(
        num_train_classes=num_train_classes,
        embedding_dim=args.embedding_dim,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    )

    triplet_loss = BatchHardTripletLoss(margin=args.margin)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader_normal=val_loader_normal,
        val_loader_abnormal=val_loader_abnormal,
        triplet_loss=triplet_loss,
        ce_loss=ce_loss,
        ce_weight=args.ce_weight,
        optimizer=optimizer,
        device=device,
        log_dir=args.log_dir,
        exp_name=args.exp_name,
        early_patience=args.early_patience,
        clean_calib_ratio=args.clean_calib_ratio,
        seed=args.seed,
    )

    trainer.fit(args.epochs, q_threshold=args.q_threshold)


if __name__ == "__main__":
    main()