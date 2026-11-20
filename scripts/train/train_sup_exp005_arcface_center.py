import os
import sys
import random
import logging
import argparse


import numpy as np

import torch
import torch.optim as optim

from torch.utils.data import DataLoader



sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)



from src.sup.datasets.split_cow import get_samples


from src.sup.datasets.dataset_OBB_normal import (
    CowTrainDatasetNormal,
    CowValDatasetFromSplit,
)


from src.sup.models.arcface_model import ArcFaceCowModel


from src.sup.losses.center_loss import CenterLoss


from src.sup.engine.arcface_center_trainer import (
    ArcFaceCenterTrainer
)


from src.sup.utils.common import set_seed

from src.sup.utils.transforms import build_transforms



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


logger=logging.getLogger(
    "arcface_center"
)




def main():


    parser=argparse.ArgumentParser()



    parser.add_argument(
        "--obb_root",
        default=r"D:\code_project\triplet_code\data\dataset_OBB"
    )


    parser.add_argument(
        "--epochs",
        type=int,
        default=35
    )


    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )


    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128
    )


    parser.add_argument(
        "--arc_s",
        type=float,
        default=30
    )


    parser.add_argument(
        "--arc_m",
        type=float,
        default=0.25
    )


    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4
    )


    parser.add_argument(
        "--wd",
        type=float,
        default=1e-4
    )


    parser.add_argument(
        "--lambda_center",
        type=float,
        default=0.005
    )


    parser.add_argument(
        "--center_lr",
        type=float,
        default=0.05
    )


    parser.add_argument(
        "--device",
        default="cuda"
    )


    parser.add_argument(
        "--num_workers",
        type=int,
        default=0
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )


    parser.add_argument(
        "--log_dir",
        default="./logs"
    )


    parser.add_argument(
        "--exp_name",
        default="exp005_arcface_center"
    )


    args=parser.parse_args()



    set_seed(
        args.seed
    )


    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )



    device=args.device



    # ===============================
    # split加载
    # ===============================

    train_samples=get_samples(
        args.obb_root,
        "seen_train"
    )


    val_samples=get_samples(
        args.obb_root,
        "seen_val"
    )



    logger.info(
        f"train samples={len(train_samples)}"
    )

    logger.info(
        f"val samples={len(val_samples)}"
    )



    # ===============================
    # label mapping
    # ===============================

    train_cows=sorted(
        list(
            set(
                x["cow_id"]
                for x in train_samples
            )
        )
    )


    id2label={

        cid:i

        for i,cid in enumerate(train_cows)

    }



    num_classes=len(id2label)



    logger.info(
        f"classes={num_classes}"
    )



    # ===============================
    # dataset
    # ===============================

    train_ds=CowTrainDatasetNormal(

        samples=train_samples,

        transform=build_transforms(train = False),

        id2label=id2label

    )


    val_ds=CowValDatasetFromSplit(

        samples=val_samples,

        id2label=id2label,

        transform=build_transforms(train = False)

    )



    train_loader=DataLoader(

        train_ds,

        batch_size=args.batch_size,

        shuffle=True,

        num_workers=args.num_workers,

        pin_memory=True

    )



    val_loader=DataLoader(

        val_ds,

        batch_size=32,

        shuffle=False,

        num_workers=args.num_workers

    )



    # ===============================
    # model
    # ===============================

    model=ArcFaceCowModel(

        num_classes=num_classes,

        embedding_dim=args.embedding_dim,

        s=args.arc_s,

        m=args.arc_m,

        pretrained=True

    )



    # ===============================
    # center loss
    # ===============================

    center_loss=CenterLoss(

        num_classes=num_classes,

        feat_dim=args.embedding_dim,

        device=device

    )



    # ===============================
    # optimizer
    # ===============================

    optimizer=optim.AdamW(

        [

            {
                "params":
                model.parameters(),

                "lr":
                args.lr
            },


            {

                "params":
                center_loss.parameters(),

                "lr":
                args.center_lr

            }

        ],

        weight_decay=args.wd

    )



    # ===============================
    # trainer
    # ===============================

    trainer=ArcFaceCenterTrainer(

        model=model,

        center_loss=center_loss,

        train_loader=train_loader,

        val_loader=val_loader,

        optimizer=optimizer,

        device=device,

        lambda_center=args.lambda_center,

        log_dir=args.log_dir,

        exp_name=args.exp_name

    )



    trainer.fit(
        args.epochs
    )




if __name__=="__main__":

    main()