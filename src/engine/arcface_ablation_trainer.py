# -*- coding: utf-8 -*-

import json
import math

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from src.utils.train_logger import log_metrics


@torch.no_grad()
def validate_classification(
    model,
    val_loader,
    device,
) -> Dict[str, float]:
    """
    使用ArcFace分类头进行seen_val闭集验证。
    """
    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    ce_loss = nn.CrossEntropyLoss()

    for images, labels in tqdm(
        val_loader,
        desc="VAL Classification",
    ):
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        _, logits = model(
            images,
            labels,
            return_theta=False,
        )

        loss = ce_loss(
            logits,
            labels,
        )

        predictions = logits.argmax(
            dim=1
        )

        batch_size = int(
            labels.size(0)
        )

        total_correct += int(
            (predictions == labels)
            .sum()
            .item()
        )

        total_samples += batch_size
        total_loss += float(
            loss.item()
        ) * batch_size

    return {
        "val_cls_acc": (
            total_correct
            / max(total_samples, 1)
        ),
        "val_cls_loss": (
            total_loss
            / max(total_samples, 1)
        ),
        "val_num_samples": float(
            total_samples
        ),
    }


@torch.no_grad()
def compute_embeddings_by_label(
    model,
    val_loader,
    device,
) -> Dict[int, List[np.ndarray]]:
    """
    提取验证集embedding并按类别分组。
    """
    model.eval()

    per_label: Dict[
        int,
        List[np.ndarray],
    ] = {}

    for images, labels in tqdm(
        val_loader,
        desc="VAL Embedding (same-mean)",
    ):
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
        ).cpu().numpy()

        labels_numpy = labels.cpu().numpy()

        for index, label in enumerate(
            labels_numpy
        ):
            label = int(label)

            if label not in per_label:
                per_label[label] = []

            per_label[label].append(
                embeddings[index]
            )

    return per_label


def compute_same_normal_mean(
    per_label: Dict[int, List[np.ndarray]],
    max_pairs_per_class: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    对每个类别计算同类pair余弦距离均值，
    再对类别均值进行宏平均。
    """
    rng = np.random.RandomState(seed)

    class_means = []
    total_pairs = 0
    classes_used = 0

    for embeddings in per_label.values():
        if len(embeddings) < 2:
            continue

        array = np.stack(
            embeddings,
            axis=0,
        ).astype(np.float32)

        # embedding已L2归一化。
        similarity_matrix = (
            array @ array.T
        )

        upper_indices = np.triu_indices(
            array.shape[0],
            k=1,
        )

        distances = (
            1.0
            - similarity_matrix[upper_indices]
        )

        if (
            max_pairs_per_class > 0
            and distances.size > max_pairs_per_class
        ):
            sampled_indices = rng.choice(
                distances.size,
                size=max_pairs_per_class,
                replace=False,
            )

            distances = distances[
                sampled_indices
            ]

        if distances.size == 0:
            continue

        class_means.append(
            float(np.mean(distances))
        )

        total_pairs += int(
            distances.size
        )

        classes_used += 1

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
            classes_used
        ),
        "val_same_pairs": float(
            total_pairs
        ),
    }


class ArcFaceTrainer:
    """
    ArcFace训练器。

    日志与checkpoint分离：
        log_run_dir/history.json
        output_run_dir/checkpoints/*.pth
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        logger,
        log_run_dir,
        output_run_dir,
        run_name,
        seed=42,
        return_theta=False,
        lambda_intra=0.02,
        acc_eps=1e-4,
        same_eps=1e-4,
        max_pairs_per_class=2000,
        save_every_epoch=True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device

        self.logger = logger
        self.run_name = str(run_name)

        self.log_run_dir = Path(
            log_run_dir
        )

        self.output_run_dir = Path(
            output_run_dir
        )

        self.log_run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.output_run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.ckpt_dir = (
            self.output_run_dir
            / "checkpoints"
        )

        self.ckpt_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history_path = (
            self.log_run_dir
            / "history.json"
        )

        self.seed = int(seed)
        self.return_theta = bool(
            return_theta
        )

        self.lambda_intra = float(
            lambda_intra
        )

        self.acc_eps = float(acc_eps)
        self.same_eps = float(same_eps)

        self.max_pairs_per_class = int(
            max_pairs_per_class
        )

        self.save_every_epoch = bool(
            save_every_epoch
        )

        self.ce_loss = nn.CrossEntropyLoss()

        self.best_val_cls_acc = -1.0
        self.best_val_same_mean = float(
            "inf"
        )

        self.history = {
            "train": [],
            "val": [],
            "runtime": [],
        }

        self.logger.info(
            "ArcFaceTrainer initialized | "
            f"run_name={self.run_name} | "
            f"history={self.history_path} | "
            f"checkpoints={self.ckpt_dir}"
        )

    def _base_model(self):
        return (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )

    def _get_current_gem_p(self):
        model = self._base_model()

        if hasattr(model, "get_gem_p"):
            return model.get_gem_p()

        return None

    def _get_learning_rates(self):
        return [
            float(group["lr"])
            for group in self.optimizer.param_groups
        ]

    def _get_augmentation_statistics(self):
        dataset = self.train_loader.dataset

        if hasattr(
            dataset,
            "get_red_augmentation_statistics",
        ):
            return dataset.get_red_augmentation_statistics(
                reset=True
            )

        return None

    def _save_history(self):
        with open(
            self.history_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.history,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def save_ckpt(
        self,
        epoch: int,
        tag: str,
    ):
        checkpoint = {
            "run_name": self.run_name,
            "epoch": int(epoch),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history,
            "best_val_cls_acc": self.best_val_cls_acc,
            "best_val_same_mean": self.best_val_same_mean,
            "trainer_state": {
                "return_theta": self.return_theta,
                "lambda_intra": self.lambda_intra,
                "acc_eps": self.acc_eps,
                "same_eps": self.same_eps,
                "max_pairs_per_class": (
                    self.max_pairs_per_class
                ),
            },
        }

        save_path = (
            self.ckpt_dir
            / f"{tag}.pth"
        )

        torch.save(
            checkpoint,
            save_path,
        )

        self.logger.info(
            f"[checkpoint] "
            f"epoch={epoch} | "
            f"tag={tag} | "
            f"path={save_path}"
        )

    def load_state_from_ckpt(
        self,
        checkpoint: dict,
    ):
        if not isinstance(
            checkpoint,
            dict,
        ):
            self.logger.warning(
                "Checkpoint不是dict，"
                "Trainer状态未恢复。"
            )
            return

        self.history = checkpoint.get(
            "history",
            {
                "train": [],
                "val": [],
                "runtime": [],
            },
        )

        self.history.setdefault(
            "runtime",
            [],
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

        self.logger.info(
            "[trainer state restored] "
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

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        total_ce_loss = 0.0
        total_l5_penalty = 0.0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"TRAIN Epoch {epoch}",
        )

        for images, labels in progress_bar:
            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )

            outputs = self.model(
                images,
                labels,
                return_theta=self.return_theta,
            )

            if self.return_theta:
                (
                    _,
                    logits,
                    theta_y,
                ) = outputs

                ce_loss = F.cross_entropy(
                    logits,
                    labels,
                )

                l5_penalty = (
                    theta_y.mean()
                    / math.pi
                )

                loss = (
                    ce_loss
                    + self.lambda_intra
                    * l5_penalty
                )
            else:
                _, logits = outputs[:2]

                ce_loss = self.ce_loss(
                    logits,
                    labels,
                )

                l5_penalty = None
                loss = ce_loss

            self.optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                predictions = logits.argmax(
                    dim=1
                )

                correct = int(
                    (predictions == labels)
                    .sum()
                    .item()
                )

            batch_size = int(
                labels.size(0)
            )

            total_loss += float(
                loss.item()
            ) * batch_size

            total_ce_loss += float(
                ce_loss.item()
            ) * batch_size

            if l5_penalty is not None:
                total_l5_penalty += float(
                    l5_penalty.item()
                ) * batch_size

            total_correct += correct
            total_samples += batch_size

            postfix = {
                "loss": (
                    f"{total_loss / max(total_samples, 1):.4f}"
                ),
                "acc": (
                    f"{total_correct / max(total_samples, 1):.3f}"
                ),
            }

            if self.return_theta:
                postfix["ce"] = (
                    f"{total_ce_loss / max(total_samples, 1):.4f}"
                )

                postfix["l5"] = (
                    f"{total_l5_penalty / max(total_samples, 1):.4f}"
                )

            progress_bar.set_postfix(
                postfix
            )

        metrics = {
            "loss": (
                total_loss
                / max(total_samples, 1)
            ),
            "acc": (
                total_correct
                / max(total_samples, 1)
            ),
            "num_samples": float(
                total_samples
            ),
        }

        if self.return_theta:
            metrics.update({
                "ce_loss": (
                    total_ce_loss
                    / max(total_samples, 1)
                ),
                "l5_penalty": (
                    total_l5_penalty
                    / max(total_samples, 1)
                ),
            })

        augmentation_statistics = (
            self._get_augmentation_statistics()
        )

        if augmentation_statistics is not None:
            metrics.update({
                "red_aug_probability": float(
                    augmentation_statistics[
                        "configured_probability"
                    ]
                ),
                "red_aug_actual_ratio": float(
                    augmentation_statistics[
                        "actual_ratio"
                    ]
                ),
                "red_aug_count": int(
                    augmentation_statistics[
                        "augmented_count"
                    ]
                ),
                "red_aug_missing_masks": int(
                    augmentation_statistics[
                        "missing_mask_count"
                    ]
                ),
                "red_aug_empty_masks": int(
                    augmentation_statistics[
                        "empty_mask_count"
                    ]
                ),
                "red_aug_strength_counts": (
                    augmentation_statistics[
                        "strength_counts"
                    ]
                ),
            })

        history_record = {
            "epoch": int(epoch),
            **metrics,
        }

        self.history["train"].append(
            history_record
        )

        log_metrics(
            logger=self.logger,
            stage="train",
            epoch=epoch,
            metrics=metrics,
        )

        return metrics

    def validate(
        self,
        epoch: int,
    ) -> Dict[str, float]:
        classification_metrics = (
            validate_classification(
                model=self.model,
                val_loader=self.val_loader,
                device=self.device,
            )
        )

        embeddings_by_label = (
            compute_embeddings_by_label(
                model=self.model,
                val_loader=self.val_loader,
                device=self.device,
            )
        )

        same_metrics = (
            compute_same_normal_mean(
                per_label=embeddings_by_label,
                max_pairs_per_class=(
                    self.max_pairs_per_class
                ),
                seed=self.seed,
            )
        )

        metrics = {
            **classification_metrics,
            **same_metrics,
        }

        self.history["val"].append({
            "epoch": int(epoch),
            **metrics,
        })

        log_metrics(
            logger=self.logger,
            stage="val",
            epoch=epoch,
            metrics={
                "cls_acc": float(
                    metrics["val_cls_acc"]
                ),
                "cls_loss": float(
                    metrics["val_cls_loss"]
                ),
                "same_mean": float(
                    metrics["val_same_mean"]
                ),
                "classes_used": int(
                    metrics["val_classes_used"]
                ),
                "same_pairs": int(
                    metrics["val_same_pairs"]
                ),
            },
        )

        return metrics

    def _is_better(
        self,
        validation_metrics: Dict[str, float],
    ) -> bool:
        new_accuracy = validation_metrics[
            "val_cls_acc"
        ]

        new_same_mean = validation_metrics[
            "val_same_mean"
        ]

        if (
            new_accuracy
            > self.best_val_cls_acc
            + self.acc_eps
        ):
            return True

        if (
            abs(
                new_accuracy
                - self.best_val_cls_acc
            )
            <= self.acc_eps
            and new_same_mean
            < self.best_val_same_mean
            - self.same_eps
        ):
            return True

        return False

    def _update_best(
        self,
        validation_metrics: Dict[str, float],
    ):
        self.best_val_cls_acc = float(
            validation_metrics[
                "val_cls_acc"
            ]
        )

        self.best_val_same_mean = float(
            validation_metrics[
                "val_same_mean"
            ]
        )

    def fit(
        self,
        epochs: int,
        start_epoch: int = 1,
    ):
        epochs = int(epochs)
        start_epoch = int(start_epoch)

        if start_epoch > epochs:
            self.logger.info(
                f"start_epoch={start_epoch} "
                f"> epochs={epochs}，无需训练。"
            )
            return

        for epoch in range(
            start_epoch,
            epochs + 1,
        ):
            self.logger.info("=" * 100)
            self.logger.info(
                f"Epoch {epoch}/{epochs}"
            )

            self.train_one_epoch(epoch)
            validation_metrics = self.validate(
                epoch
            )

            runtime_metrics = {
                "learning_rates": (
                    self._get_learning_rates()
                ),
                "gem_p": (
                    self._get_current_gem_p()
                ),
            }

            self.history["runtime"].append({
                "epoch": int(epoch),
                **runtime_metrics,
            })

            log_metrics(
                logger=self.logger,
                stage="runtime",
                epoch=epoch,
                metrics=runtime_metrics,
            )

            is_better = self._is_better(
                validation_metrics
            )

            if is_better:
                self._update_best(
                    validation_metrics
                )

            self._save_history()

            if self.save_every_epoch:
                self.save_ckpt(
                    epoch,
                    tag=f"epoch_{epoch}",
                )

            self.save_ckpt(
                epoch,
                tag="latest",
            )

            if is_better:
                self.save_ckpt(
                    epoch,
                    tag="best",
                )

                self.logger.info(
                    f"[best] epoch={epoch} | "
                    f"best_val_cls_acc="
                    f"{self.best_val_cls_acc:.6f} | "
                    f"best_val_same_mean="
                    f"{self.best_val_same_mean:.6f}"
                )

        self.logger.info(
            "Training done | "
            f"best_val_cls_acc="
            f"{self.best_val_cls_acc:.6f} | "
            f"best_val_same_mean="
            f"{self.best_val_same_mean:.6f}"
        )
