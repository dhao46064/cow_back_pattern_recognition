import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger("cow_proxyanchor")


# =====================================================
# Embedding validation
# =====================================================

@torch.no_grad()
def compute_embeddings_by_label(
    model,
    loader,
    device
):

    model.eval()

    result = {}

    for images, labels in tqdm(
        loader,
        desc="Embedding"
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        emb = model(images)

        emb = F.normalize(
            emb,
            dim=1
        )

        emb = emb.cpu().numpy()

        labels = labels.numpy()


        for e,l in zip(emb,labels):

            l=int(l)

            if l not in result:
                result[l]=[]

            result[l].append(e)


    return result



def compute_same_mean(
    embeddings,
    max_pairs=2000,
    seed=42
):

    rng=np.random.RandomState(seed)

    class_mean=[]


    for label, arr in embeddings.items():

        if len(arr)<2:
            continue


        arr=np.stack(arr)


        n=len(arr)


        pairs=[
            (i,j)
            for i in range(n)
            for j in range(i+1,n)
        ]


        if len(pairs)>max_pairs:

            idx=rng.choice(
                len(pairs),
                max_pairs,
                replace=False
            )

            pairs=[
                pairs[k]
                for k in idx
            ]


        dist=[]


        for i,j in pairs:

            sim=np.dot(
                arr[i],
                arr[j]
            )

            dist.append(
                1-sim
            )


        class_mean.append(
            np.mean(dist)
        )


    return float(
        np.mean(class_mean)
    )



# =====================================================
# Trainer
# =====================================================


class ProxyanchorTrainer:


    def __init__(
        self,
        model,
        proxy_loss,
        train_loader,
        val_loader,
        optimizer,
        device,
        log_dir,
        exp_name,
        seed=42
    ):


        self.model=model.to(device)

        self.proxy_loss=proxy_loss.to(device)


        self.train_loader=train_loader

        self.val_loader=val_loader

        self.optimizer=optimizer

        self.device=device

        self.seed=seed



        self.save_dir=Path(log_dir)/exp_name

        self.ckpt_dir=self.save_dir/"checkpoints"

        self.ckpt_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        self.history={

            "train":[],
            "val":[]

        }



    # -------------------------------------------------

    def save_ckpt(
        self,
        epoch,
        tag="latest"
    ):
        checkpoint = {
            "epoch": epoch,

            "model": self.model.state_dict(),

            "proxy_loss": self.proxy_loss.state_dict(),

            "optimizer": self.optimizer.state_dict(),

            "history": self.history,
        }

        # 兼容你已有的最佳指标
        if hasattr(
            self,
            "best_val_same_mean"
        ):
            checkpoint["best_val_same_mean"] = (
                self.best_val_same_mean
            )

        if hasattr(
            self,
            "best_val_cls_acc"
        ):
            checkpoint["best_val_cls_acc"] = (
                self.best_val_cls_acc
            )

        if hasattr(
            self,
            "early_counter"
        ):
            checkpoint["early_counter"] = (
                self.early_counter
            )

        checkpoint_path = (
            self.ckpt_dir
            /
            f"{tag}.pth"
        )

        torch.save(
            checkpoint,
            checkpoint_path
        )

        logger.info(
            f"checkpoint saved：{checkpoint_path}"
        )


        # -------------------------------------------------

    def train_one_epoch(
        self,
        epoch
    ):


        self.model.train()


        self.proxy_loss.train()



        total_loss=0

        n=0



        bar=tqdm(
            self.train_loader,
            desc=f"Train {epoch}"
        )



        for images,labels in bar:


            images=images.to(
                self.device
            )

            labels=labels.to(
                self.device
            )
        if n == 0:

            print(images.shape)

            print(labels)

            print(
                torch.unique(
                    labels,
                    return_counts=True
                )
            )



            emb=self.model(images)



            loss=self.proxy_loss(
                emb,
                labels
            )



            self.optimizer.zero_grad()



            loss.backward()



            self.optimizer.step()



            total_loss+=loss.item()

            n+=1



            bar.set_postfix(

                loss=f"{total_loss/n:.4f}"

            )



        info={

            "epoch":epoch,

            "loss":
                total_loss/n

        }


        self.history["train"].append(info)



        return info




    # -------------------------------------------------

    @torch.no_grad()
    def validate(
        self,
        epoch
    ):


        emb_dict=compute_embeddings_by_label(

            self.model,

            self.val_loader,

            self.device

        )


        same_mean=compute_same_mean(

            emb_dict

        )



        info={

            "epoch":epoch,

            "same_mean":
                same_mean

        }


        self.history["val"].append(info)



        logger.info(

            f"epoch={epoch}, same_mean={same_mean:.6f}"

        )


        return info
    def fit(self,epochs,start_epoch =1):
    
    
            for epoch in range(
                start_epoch,
                epochs+1
            ):
    
    
                logger.info("=" * 80)
                logger.info(
                    f"Epoch {epoch}/{epochs}")
    
    
                self.train_one_epoch(
                    epoch
                )
    
    
                self.validate(
                    epoch
                )
    
    
                self.save_ckpt(
                    epoch
                )
    
    
    
                with open(
    
                    self.save_dir/"history.json",
    
                    "w",
    
                    encoding="utf-8"
    
                ) as f:
    
    
                    json.dump(
    
                        self.history,
    
                        f,
    
                        indent=2
    
                    )
    
    
            logger.info(
                "Training finished"
            )

   
        


    # -------------------------------------------------
