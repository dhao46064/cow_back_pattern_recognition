# -*- coding: utf-8 -*-

"""
Backbone Ablation Training

Experiment:

ResNet18
ResNet50
DenseNet121
EfficientNet-B0
Swin-T
HRNet-W32


Fixed:

ArcFace
GAP
embedding=128

No:
- red augmentation
- ordinary augmentation
- GeM
- L5 constraint

"""


import sys
import random

from pathlib import Path


import numpy as np

import torch

import torch.optim as optim


from torch.utils.data import DataLoader



# =====================================================
# project path
# =====================================================


CURRENT_FILE = Path(
    __file__
).resolve()


PROJECT_ROOT = CURRENT_FILE.parents[2]


if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )



# =====================================================
# import your project
# =====================================================


from src.datasets.split import (
    get_samples
)


from src.datasets.dataset_OBB_normal import (

    CowTrainDatasetNormal,

    CowValDatasetFromSplit,

)


from src.models.arcface_model_ablation import (

    ArcFaceCowModel,

)


from src.engine.arcface_ablation_trainer import (

    ArcFaceTrainer,

)


from src.utils.common import (

    set_seed,

)


from src.utils.transforms import (

    build_transforms,

)


from src.utils.train_logger import (

    create_training_run,

    save_training_config,

    close_training_logger,

)





# =====================================================
# path
# =====================================================


DATA_ROOT = Path(
    r"D:\datasets\cow_recognition8.3\train_dataset"
)


LOG_ROOT = Path(
    r"D:\code_project\cow_recognition\logs\train_logs"
)


OUTPUT_ROOT = Path(
    r"D:\code_project\cow_recognition\outputs"
)





# =====================================================
# seed
# =====================================================


def seed_worker(worker_id):

    worker_seed = (
        torch.initial_seed()
        %
        2**32
    )


    np.random.seed(
        worker_seed
    )


    random.seed(
        worker_seed
    )






# =====================================================
# args
# =====================================================


import argparse



def build_parser():


    parser = argparse.ArgumentParser()


    # experiment name

    parser.add_argument(

        "--exp_name",

        type=str,

        default="exp_ablation_ResNet50",

    )


    # backbone

    parser.add_argument(

        "--backbone",

        type=str,

        default="ResNet50",

        choices=[

            "resnet18",

            "resnet50",

            "densenet121",

            "efficientnet_b0",

            "swin_t",

            "hrnet_w32",

        ]

    )



    # training

    parser.add_argument(

        "--epochs",

        type=int,

        default=20,

    )
    parser.add_argument(

    "--start_epoch",

    type=int,

    default=1,

    help="训练开始epoch"

)


    parser.add_argument(

        "--batch_size",

        type=int,

        default=64,

    )


    parser.add_argument(

        "--num_workers",

        type=int,

        default=4,

    )



    parser.add_argument(

        "--embedding_dim",

        type=int,

        default=128,

    )



    # arcface

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



    # optimizer


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



    parser.add_argument(

        "--device",

        type=str,

        default="cuda",

    )



    return parser







# =====================================================
# main
# =====================================================



def main():


    parser = build_parser()

    args = parser.parse_args()



    set_seed(42)



    logger, log_dir, out_dir, run_name = create_training_run(

        log_root=str(LOG_ROOT),

        output_root=str(OUTPUT_ROOT),

        exp_name=args.exp_name,

    )



    device = torch.device(

        args.device

        if torch.cuda.is_available()

        else "cpu"

    )



    logger.info(
        f"Backbone={args.backbone}"
    )



    # =================================================
    # dataset
    # =================================================


    train_samples = get_samples(

        str(DATA_ROOT),

        "seen_train",

    )


    val_samples = get_samples(

        str(DATA_ROOT),

        "seen_val",

    )



    train_cow_ids = sorted(

        list(

            set(

                str(x["cow_id"])

                for x in train_samples

            )

        )

    )



    id2label = {


        cow_id:i

        for i,cow_id in enumerate(
            train_cow_ids
        )


    }




    # =================================================
    # no augmentation
    # =================================================


    train_transform = build_transforms(

        train=False

    )


    val_transform = build_transforms(

        train=False

    )




    train_dataset = CowTrainDatasetNormal(

        samples=train_samples,

        transform=train_transform,

        id2label=id2label,

        red_augment=None,

    )



    val_dataset = CowValDatasetFromSplit(

        samples=val_samples,

        id2label=id2label,

        transform=val_transform,

    )




    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

        num_workers=args.num_workers,

        pin_memory=True,

        worker_init_fn=seed_worker,

    )




    val_loader = DataLoader(

        val_dataset,

        batch_size=32,

        shuffle=False,

        num_workers=args.num_workers,

        pin_memory=True,

        worker_init_fn=seed_worker,

    )




    # =================================================
    # model
    # =================================================


    model = ArcFaceCowModel(

        num_classes=len(id2label),

        backbone=args.backbone,

        embedding_dim=args.embedding_dim,

        s=args.arc_s,

        m=args.arc_m,

        pretrained=True,

    )




    optimizer = optim.AdamW(

        model.parameters(),

        lr=args.lr,

        weight_decay=args.wd,

    )





    trainer = ArcFaceTrainer(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        optimizer=optimizer,

        device=device,

        logger=logger,

        log_run_dir=log_dir,

        output_run_dir=out_dir,

        run_name=run_name,

        return_theta=False,

        lambda_intra=0.0,

        save_every_epoch=True,

    )




    trainer.fit(

    epochs=args.epochs,

    start_epoch=args.start_epoch,

)




    close_training_logger(

        logger

    )





if __name__=="__main__":

    main()