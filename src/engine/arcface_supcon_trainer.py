# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


logger = logging.getLogger("arcface_supcon")


class SupConProjectionHead(nn.Module):
    """
    SupCon训练专用投影头。

    ArcFace和最终测试使用原始embedding；
    SupCon只在projected feature上计算。
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:

        projected = self.net(embeddings)

        projected = F.normalize(
            projected,
            p=2,
            dim=1,
        )

        return projected


@torch.no_grad()
def validate_classification(
    model,
    val_loader,
    device: str,
) -> Dict[str, float]:

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(
        val_loader,
        desc="VAL Classification",
    )

    for images, labels in progress:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        ).long()

        _, logits = model(
            images,
            labels,
        )

        loss = criterion(
            logits,
            labels,
        )

        predictions = logits.argmax(
            dim=1
        )

        batch_size = labels.size(0)

        total_loss += (
            float(loss.item())
            * batch_size
        )

        total_correct += int(
            (predictions == labels)
            .sum()
            .item()
        )

        total_samples += batch_size

    return {
        "val_cls_loss": (
            total_loss
            / max(total_samples, 1)
        ),
        "val_cls_acc": (
            total_correct
            / max(total_samples, 1)
        ),
        "val_num_samples": float(
            total_samples
        ),
    }


@torch.no_grad()
def extract_embeddings_by_label(
    model,
    val_loader,
    device: str,
) -> Dict[int, List[np.ndarray]]:

    model.eval()

    per_label: Dict[
        int,
        List[np.ndarray]
    ] = {}

    progress = tqdm(
        val_loader,
        desc="VAL Embedding",
    )

    for images, labels in progress:

        images = images.to(
            device,
            non_blocking=True,
        )

        embeddings = model.forward_emb(
            images
        )

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        embeddings_np = (
            embeddings
            .cpu()
            .numpy()
        )

        labels_np = (
            labels
            .cpu()
            .numpy()
        )

        for embedding, label in zip(
            embeddings_np,
            labels_np,
        ):
            label = int(label)

            per_label.setdefault(
                label,
                [],
            ).append(embedding)

    return per_label


def compute_same_mean(
    per_label: Dict[int, List[np.ndarray]],
    max_pairs_per_class: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    对每头牛内部计算余弦距离均值，
    再对所有牛等权平均。
    """

    rng = np.random.RandomState(seed)

    class_means = []
    total_pairs = 0

    for embeddings in per_label.values():

        if len(embeddings) < 2:
            continue

        array = np.stack(
            embeddings,
            axis=0,
        ).astype(np.float32)

        count = array.shape[0]

        upper_i, upper_j = np.triu_indices(
            count,
            k=1,
        )

        pair_count = len(upper_i)

        if pair_count == 0:
            continue

        if pair_count > max_pairs_per_class:

            selected = rng.choice(
                pair_count,
                size=max_pairs_per_class,
                replace=False,
            )

            upper_i = upper_i[selected]
            upper_j = upper_j[selected]

        similarities = np.sum(
            array[upper_i]
            * array[upper_j],
            axis=1,
        )

        distances = 1.0 - similarities

        class_means.append(
            float(distances.mean())
        )

        total_pairs += int(
            distances.size
        )

    if not class_means:
        return {
            "val_same_mean": float("inf"),
            "val_classes_used": 0.0,
            "val_same_pairs": 0.0,
        }

    return {
        "val_same_mean": float(
            np.mean(class_means)
        ),
        "val_classes_used": float(
            len(class_means)
        ),
        "val_same_pairs": float(
            total_pairs
        ),
    }


class ArcFaceSupConTrainer:
    def __init__(
        self,
        model,
        projector,
        supcon_loss,
        train_loader,
        val_loader,
        optimizer,
        device: str,
        lambda_supcon: float = 0.1,
        log_dir: str = "./logs",
        exp_name: str = "arcface_supcon",
        seed: int = 42,
        max_pairs_per_class: int = 2000,
    ):
        self.device = device

        self.model = model.to(device)
        self.projector = projector.to(device)
        self.supcon_loss = supcon_loss.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer

        self.lambda_supcon = float(
            lambda_supcon
        )

        self.seed = int(seed)
        self.max_pairs_per_class = int(
            max_pairs_per_class
        )

        self.arcface_loss = (
            nn.CrossEntropyLoss()
        )

        self.log_dir = (
            Path(log_dir)
            / exp_name
        )

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.checkpoint_dir = (
            self.log_dir
            / "checkpoints"
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_val_cls_acc = -1.0
        self.best_val_same_mean = float(
            "inf"
        )

        self.history = {
            "train": [],
            "val": [],
        }

        logger.info(
            "ArcFaceSupConTrainer initialized | "
            f"lambda_supcon={self.lambda_supcon}"
        )

    def save_checkpoint(
        self,
        epoch: int,
        tag: str,
    ) -> None:

        checkpoint = {
            "epoch": int(epoch),

            # 推理与pair distance脚本只需要这个
            "model": self.model.state_dict(),

            # 仅训练恢复时需要
            "projector": (
                self.projector.state_dict()
            ),

            "optimizer": (
                self.optimizer.state_dict()
            ),

            "history": self.history,

            "best_val_cls_acc": (
                self.best_val_cls_acc
            ),

            "best_val_same_mean": (
                self.best_val_same_mean
            ),

            "lambda_supcon": (
                self.lambda_supcon
            ),
        }

        path = (
            self.checkpoint_dir
            / f"{tag}.pth"
        )

        torch.save(
            checkpoint,
            path,
        )

        logger.info(
            f"[checkpoint] saved: {path}"
        )

    def load_state_from_checkpoint(
        self,
        checkpoint: dict,
    ) -> None:

        if not isinstance(
            checkpoint,
            dict,
        ):
            return

        self.history = checkpoint.get(
            "history",
            {
                "train": [],
                "val": [],
            },
        )

        self.best_val_cls_acc = float(
            checkpoint.get(
                "best_val_cls_acc",
                -1.0,
            )
        )

        self.best_val_same_mean = float(
            checkpoint.get(
                "best_val_same_mean",
                float("inf"),
            )
        )

        logger.info(
            "[trainer restored] "
            f"best_val_cls_acc="
            f"{self.best_val_cls_acc:.6f} | "
            f"best_val_same_mean="
            f"{self.best_val_same_mean:.6f}"
        )

    def train_one_epoch(
        self,
        epoch: int,
    ) -> Dict[str, float]:

        self.model.train()
        self.projector.train()

        total_loss = 0.0
        total_arcface_loss = 0.0
        total_supcon_loss = 0.0
        total_correct = 0
        total_samples = 0

        progress = tqdm(
            self.train_loader,
            desc=f"TRAIN Epoch {epoch}",
        )

        for images, labels in progress:

            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            ).long()

            embeddings, logits = self.model(
                images,
                labels,
            )

            projected = self.projector(
                embeddings
            )

            loss_arcface = self.arcface_loss(
                logits,
                labels,
            )

            loss_supcon = self.supcon_loss(
                projected,
                labels,
            )

            loss = (
                loss_arcface
                + self.lambda_supcon
                * loss_supcon
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            self.optimizer.step()

            with torch.no_grad():

                predictions = logits.argmax(
                    dim=1
                )

                correct = (
                    predictions == labels
                ).sum().item()

            batch_size = labels.size(0)

            total_loss += (
                float(loss.item())
                * batch_size
            )

            total_arcface_loss += (
                float(loss_arcface.item())
                * batch_size
            )

            total_supcon_loss += (
                float(loss_supcon.item())
                * batch_size
            )

            total_correct += int(correct)
            total_samples += batch_size

            progress.set_postfix({
                "loss": (
                    f"{total_loss / max(total_samples, 1):.4f}"
                ),
                "arc": (
                    f"{total_arcface_loss / max(total_samples, 1):.4f}"
                ),
                "sup": (
                    f"{total_supcon_loss / max(total_samples, 1):.4f}"
                ),
                "acc": (
                    f"{total_correct / max(total_samples, 1):.4f}"
                ),
            })

        metrics = {
            "epoch": int(epoch),

            "loss": (
                total_loss
                / max(total_samples, 1)
            ),

            "arcface_loss": (
                total_arcface_loss
                / max(total_samples, 1)
            ),

            "supcon_loss": (
                total_supcon_loss
                / max(total_samples, 1)
            ),

            "acc": (
                total_correct
                / max(total_samples, 1)
            ),
        }

        self.history["train"].append(
            metrics
        )

        logger.info(
            f"[train] epoch={epoch} | "
            f"loss={metrics['loss']:.6f} | "
            f"arc={metrics['arcface_loss']:.6f} | "
            f"sup={metrics['supcon_loss']:.6f} | "
            f"acc={metrics['acc']:.6f}"
        )

        return metrics

    def validate(
        self,
        epoch: int,
    ) -> Dict[str, float]:

        classification_metrics = (
            validate_classification(
                self.model,
                self.val_loader,
                self.device,
            )
        )

        embeddings_by_label = (
            extract_embeddings_by_label(
                self.model,
                self.val_loader,
                self.device,
            )
        )

        same_metrics = compute_same_mean(
            embeddings_by_label,
            max_pairs_per_class=(
                self.max_pairs_per_class
            ),
            seed=self.seed,
        )

        metrics = {
            "epoch": int(epoch),
            **classification_metrics,
            **same_metrics,
        }

        self.history["val"].append(
            metrics
        )

        logger.info(
            f"[val] epoch={epoch} | "
            f"cls_acc={metrics['val_cls_acc']:.6f} | "
            f"cls_loss={metrics['val_cls_loss']:.6f} | "
            f"same_mean={metrics['val_same_mean']:.6f} | "
            f"classes={int(metrics['val_classes_used'])} | "
            f"pairs={int(metrics['val_same_pairs'])}"
        )

        return metrics

    def is_better(
        self,
        metrics: Dict[str, float],
    ) -> bool:

        new_acc = metrics[
            "val_cls_acc"
        ]

        new_same = metrics[
            "val_same_mean"
        ]

        if new_acc > (
            self.best_val_cls_acc
            + 1e-4
        ):
            return True

        if abs(
            new_acc
            - self.best_val_cls_acc
        ) <= 1e-4:

            if new_same < (
                self.best_val_same_mean
                - 1e-4
            ):
                return True

        return False

    def fit(
        self,
        epochs: int,
        start_epoch: int = 1,
    ) -> None:

        for epoch in range(
            start_epoch,
            epochs + 1,
        ):

            logger.info("=" * 80)
            logger.info(
                f"Epoch {epoch}/{epochs}"
            )

            self.train_one_epoch(
                epoch
            )

            val_metrics = self.validate(
                epoch
            )

            self.save_checkpoint(
                epoch,
                tag=f"epoch_{epoch}",
            )

            self.save_checkpoint(
                epoch,
                tag="latest",
            )

            if self.is_better(
                val_metrics
            ):
                self.best_val_cls_acc = (
                    val_metrics[
                        "val_cls_acc"
                    ]
                )

                self.best_val_same_mean = (
                    val_metrics[
                        "val_same_mean"
                    ]
                )

                self.save_checkpoint(
                    epoch,
                    tag="best",
                )

                logger.info(
                    f"[best] epoch={epoch} | "
                    f"cls_acc="
                    f"{self.best_val_cls_acc:.6f} | "
                    f"same_mean="
                    f"{self.best_val_same_mean:.6f}"
                )

            history_path = (
                self.log_dir
                / "history.json"
            )

            with open(
                history_path,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    self.history,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

        logger.info(
            "Training completed | "
            f"best_val_cls_acc="
            f"{self.best_val_cls_acc:.6f} | "
            f"best_val_same_mean="
            f"{self.best_val_same_mean:.6f}"
        )