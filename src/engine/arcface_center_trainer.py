import logging
from pathlib import Path
import json

import torch
import torch.nn as nn

from tqdm import tqdm


logger = logging.getLogger(
    "arcface_center"
)



class ArcFaceCenterTrainer:


    def __init__(
        self,
        model,
        center_loss,
        train_loader,
        val_loader,
        optimizer,
        device,
        lambda_center=0.001,
        log_dir="./logs",
        exp_name="arcface_center"
    ):


        self.model=model.to(device)

        self.center_loss=center_loss.to(device)


        self.train_loader=train_loader
        self.val_loader=val_loader

        self.optimizer=optimizer

        self.device=device


        self.lambda_center=lambda_center


        self.ce_loss=nn.CrossEntropyLoss()


        self.log_dir=Path(log_dir)/exp_name

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        self.ckpt_dir=self.log_dir/"checkpoints"

        self.ckpt_dir.mkdir(
            exist_ok=True
        )


        self.history={

            "train":[]

        }



    def save_ckpt(
        self,
        epoch,
        tag
    ):


        ckpt={

            "epoch":epoch,

            "model":
                self.model.state_dict(),

            "center_loss":
                self.center_loss.state_dict(),

            "optimizer":
                self.optimizer.state_dict(),

            "history":
                self.history

        }


        path=self.ckpt_dir/f"{tag}.pth"


        torch.save(
            ckpt,
            path
        )


        logger.info(
            f"save:{path}"
        )



    def train_one_epoch(
        self,
        epoch
    ):


        self.model.train()

        self.center_loss.train()


        total_loss=0

        total_arc=0

        total_center=0


        n=0


        bar=tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}"
        )


        for images,labels in bar:


            images=images.to(
                self.device
            )

            labels=labels.to(
                self.device
            )


            emb,logits=self.model(
                images,
                labels
            )


            loss_arc=self.ce_loss(
                logits,
                labels
            )


            loss_center=self.center_loss(
                emb,
                labels
            )


            loss=(
                loss_arc
                +
                self.lambda_center*loss_center
            )


            self.optimizer.zero_grad()


            loss.backward()


            self.optimizer.step()



            total_loss+=loss.item()

            total_arc+=loss_arc.item()

            total_center+=loss_center.item()


            n+=1


            bar.set_postfix({

                "loss":
                f"{total_loss/n:.4f}",

                "arc":
                f"{total_arc/n:.4f}",

                "center":
                f"{total_center/n:.4f}"

            })


        result={

            "epoch":epoch,

            "loss":total_loss/n,

            "arc_loss":total_arc/n,

            "center_loss":total_center/n

        }


        self.history["train"].append(
            result
        )


        return result



    def fit(
        self,
        epochs
    ):


        for epoch in range(
            1,
            epochs+1
        ):


            self.train_one_epoch(
                epoch
            )


            self.save_ckpt(
                epoch,
                f"epoch_{epoch}"
            )


            self.save_ckpt(
                epoch,
                "latest"
            )


            with open(
                self.log_dir/"history.json",
                "w"
            ) as f:

                json.dump(
                    self.history,
                    f,
                    indent=2
                )