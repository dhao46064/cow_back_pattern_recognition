import json
import logging
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger("cow_arcface")


@torch.no_grad()
def validate_classification(
    model,
    val_loader,
    device: str
    
) -> Dict[str, float]:
    """
    直接用分类头做验证：
    输入一张图，是否分到正确 cow_id
    期望 val_loader 返回: (images, labels)
    """
    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    n_batch = 0

    ce_loss = nn.CrossEntropyLoss()

    for images, labels in tqdm(val_loader, desc="VAL Classification"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        emb, logits = model(images, labels,return_theta = False)
        loss = ce_loss(logits, labels)

        pred = logits.argmax(dim=1)
        correct = (pred == labels).sum().item()

        total_correct += int(correct)
        total_samples += int(labels.size(0))
        total_loss += float(loss.item())
        n_batch += 1

    acc = total_correct / max(total_samples, 1)
    avg_loss = total_loss / max(n_batch, 1)

    return {
        "val_cls_acc": float(acc),
        "val_cls_loss": float(avg_loss),
        "val_num_samples": float(total_samples),
    }


@torch.no_grad()
def compute_embeddings_by_label(
    model,
    val_loader,
    device: str
) -> Dict[int, List[np.ndarray]]:
    """
    从 val_loader 中提 embedding，并按 label 分组。
    期望 val_loader 返回: (images, labels)
    返回:
        per_label[label] = [emb1, emb2, ...]
    """
    model.eval()
    per_label: Dict[int, List[np.ndarray]] = {}

    for images, labels in tqdm(val_loader, desc="VAL Embedding (same-mean)"):
        images = images.to(device, non_blocking=True)
        emb = model.forward_emb(images)
        emb = F.normalize(emb, p=2, dim=1).cpu().numpy()

        labels_np = labels.cpu().numpy()
        for i in range(len(labels_np)):
            lab = int(labels_np[i])
            if lab not in per_label:
                per_label[lab] = []
            per_label[lab].append(emb[i])

    return per_label


def compute_same_normal_mean(
    per_label: Dict[int, List[np.ndarray]],
    max_pairs_per_class: int = 2000,
    seed: int = 42
) -> Dict[str, float]:
    """
    计算 same-normal mean：
    对每个类别内部，计算同类 pair 的 cosine distance 平均值；
    最后对所有类别求平均。

    cosine distance = 1 - cosine similarity
    embedding 已经 L2 normalize
    """
    rng = np.random.RandomState(seed)

    class_means = []
    total_pairs = 0
    classes_used = 0

    for lab, embs in per_label.items():
        if len(embs) < 2:
            continue

        arr = np.stack(embs, axis=0).astype(np.float32)  # [N, D]
        n = arr.shape[0]

        # 全部两两组合可能很多，做随机采样
        all_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                all_pairs.append((i, j))

        if len(all_pairs) > max_pairs_per_class:
            idx = rng.choice(len(all_pairs), size=max_pairs_per_class, replace=False)
            sampled_pairs = [all_pairs[k] for k in idx]
        else:
            sampled_pairs = all_pairs

        dists = []
        for i, j in sampled_pairs:
            sim = float(np.dot(arr[i], arr[j]))   # normalized embedding
            dist = 1.0 - sim
            dists.append(dist)

        if len(dists) == 0:
            continue

        class_means.append(float(np.mean(dists)))
        total_pairs += len(dists)
        classes_used += 1

    if len(class_means) == 0:
        return {
            "val_same_mean": float("inf"),
            "val_classes_used": 0.0,
            "val_same_pairs": 0.0,
        }

    return {
        "val_same_mean": float(np.mean(class_means)),
        "val_classes_used": float(classes_used),
        "val_same_pairs": float(total_pairs),
    }


class ArcFaceTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        log_dir="./logs",
        exp_name="arcface_obb",
        early_patience=10,
        seed=42,
        ##添加L5约束
        return_theta=False,
        lambda_intra=0.02,
        ##
        acc_eps=1e-4,
        same_eps=1e-4,
        max_pairs_per_class=2000,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device

        self.seed = seed
        self.return_theta = return_theta
        self.lambda_intra = lambda_intra

        self.acc_eps = acc_eps
        self.same_eps = same_eps
        self.max_pairs_per_class = max_pairs_per_class

        self.ce_loss = nn.CrossEntropyLoss()

        self.log_dir = Path(log_dir) / exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt_dir = self.log_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)

        # 主指标
        self.best_val_cls_acc = -1.0
        # 辅助指标（越小越好）
        self.best_val_same_mean = float("inf")

        self.early_patience = early_patience
        self.early_counter = 0

        self.history = {
            "train": [],
            "val": []
        }

    def save_ckpt(self, epoch: int, tag: str):
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history,
            "best_val_cls_acc": self.best_val_cls_acc,
            "best_val_same_mean": self.best_val_same_mean,
            "early_counter": self.early_counter,
        }
        path = self.ckpt_dir / f"{tag}.pth"
        torch.save(ckpt, path)
        logger.info(f"[ckpt] saved: {path}")

    def load_state_from_ckpt(self, ckpt: dict):
        if not isinstance(ckpt, dict):
            logger.warning("Checkpoint is not a dict, skip restoring trainer states.")
            return

        self.history = ckpt.get("history", {"train": [], "val": []})
        self.best_val_cls_acc = float(ckpt.get("best_val_cls_acc", -1.0))
        self.best_val_same_mean = float(ckpt.get("best_val_same_mean", float("inf")))
        self.early_counter = int(ckpt.get("early_counter", 0))

        logger.info(
            f"[trainer state restored] "
            f"best_val_cls_acc={self.best_val_cls_acc:.6f}, "
            f"best_val_same_mean={self.best_val_same_mean:.6f}, "
            f"early_counter={self.early_counter}"
        )

    def train_one_epoch(self, epoch: int):
        self.model.train()

        total_loss = 0.0
        total_acc = 0.0
        n = 0

        pbar = tqdm(self.train_loader, desc=f"TRAIN Epoch {epoch}")
        # for images, labels in pbar:
        #     images = images.to(self.device, non_blocking=True)
        #     labels = labels.to(self.device, non_blocking=True)

        #     if not self.return_theta:
        #         emb, logits = self.model(images, labels)
        #         loss = self.ce_loss(logits, labels)
        #     else:
        #         emb, logits, theta_y = self.model(images, labels, return_theta=True)

        #         ce_loss = F.cross_entropy(logits, labels)
        #         l5_penalty = theta_y.mean() / math.pi
        #         loss = ce_loss + self.lambda_intra * l5_penalty

        #     self.optimizer.zero_grad(set_to_none=True)
        #     loss.backward()
        #     self.optimizer.step()

        #     with torch.no_grad():
        #         pred = logits.argmax(dim=1)
        #         acc = (pred == labels).float().mean().item()
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images, labels, return_theta=self.return_theta)

            if self.return_theta:
                emb, logits, theta_y = outputs

                ce_loss = F.cross_entropy(logits, labels)
                l5_penalty = theta_y.mean() / math.pi
                loss = ce_loss + self.lambda_intra * l5_penalty
            else:
                emb, logits = outputs[:2]
                loss = self.ce_loss(logits, labels)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == labels).float().mean().item()

            total_loss += float(loss.item())
            total_acc += float(acc)
            n += 1

            # pbar.set_postfix({
            #     "loss": f"{total_loss / n:.4f}",
            #     "acc": f"{total_acc / n:.3f}",
            # })
            if self.return_theta:
                pbar.set_postfix({
                    "loss": f"{total_loss / n:.4f}",
                    "acc": f"{total_acc / n:.3f}",
                    "ce": f"{ce_loss.item():.4f}",
                    "l5": f"{l5_penalty.item():.4f}",
                })
            else:
                pbar.set_postfix({
                    "loss": f"{total_loss / n:.4f}",
                    "acc": f"{total_acc / n:.3f}",
                })

        m = {
            "epoch": epoch,
            "loss": total_loss / max(n, 1),
            "acc": total_acc / max(n, 1),
        }
        self.history["train"].append(m)
        logger.info(f"[train] epoch={epoch} loss={m['loss']:.4f} acc={m['acc']:.4f}")

    def validate(self, epoch: int):
        # 1) 主指标：val classification accuracy
        cls_metrics = validate_classification(
            self.model,
            self.val_loader,
            self.device
        )

        # 2) 辅助指标：same-normal mean
        per_label = compute_embeddings_by_label(
            self.model,
            self.val_loader,
            self.device
        )

        same_metrics = compute_same_normal_mean(
            per_label,
            max_pairs_per_class=self.max_pairs_per_class,
            seed=self.seed
        )

        metrics = {
            "epoch": epoch,
            **cls_metrics,
            **same_metrics,
        }

        self.history["val"].append(metrics)

        logger.info(
            f"[val] epoch={epoch} "
            f"cls_acc={metrics['val_cls_acc']:.6f} "
            f"cls_loss={metrics['val_cls_loss']:.6f} "
            f"same_mean={metrics['val_same_mean']:.6f} "
            f"classes_used={int(metrics['val_classes_used'])} "
            f"same_pairs={int(metrics['val_same_pairs'])}"
        )
        return metrics

    def _is_better(self, valm: Dict[str, float]) -> bool:
        """
        早停判断逻辑：
        1. 先看 val_cls_acc 是否明显提升
        2. 若 val_cls_acc 基本持平，再看 val_same_mean 是否明显下降
        """
        new_acc = valm["val_cls_acc"]
        new_same = valm["val_same_mean"]

        # 第一优先级：分类准确率显著提升
        if new_acc > self.best_val_cls_acc + self.acc_eps:
            return True

        # 第二优先级：分类准确率基本持平时，看 same_mean 是否显著更小
        if abs(new_acc - self.best_val_cls_acc) <= self.acc_eps:
            if new_same < self.best_val_same_mean - self.same_eps:
                return True

        return False

    def _update_best(self, valm: Dict[str, float]):
        self.best_val_cls_acc = valm["val_cls_acc"]
        self.best_val_same_mean = valm["val_same_mean"]

    def fit(self, epochs: int, start_epoch: int = 1):
        if start_epoch > epochs:
            logger.info(
                f"start_epoch ({start_epoch}) > epochs ({epochs}), nothing to train."
            )
            logger.info(
                f"Training done. "
                f"Best val cls_acc={self.best_val_cls_acc:.6f}, "
                f"best val same_mean={self.best_val_same_mean:.6f}"
            )
            return

        for ep in range(start_epoch, epochs + 1):
            logger.info("=" * 70)
            logger.info(f"Epoch {ep}/{epochs}")

            self.train_one_epoch(ep)
            valm = self.validate(ep)

            self.save_ckpt(ep, tag=f"epoch_{ep}")
            self.save_ckpt(ep, tag="latest")

            if self._is_better(valm):
                self._update_best(valm)
                self.early_counter = 0
                self.save_ckpt(ep, tag="best")
                logger.info(
                    f"[best] epoch={ep}, "
                    f"best_val_cls_acc={self.best_val_cls_acc:.6f}, "
                    f"best_val_same_mean={self.best_val_same_mean:.6f}"
                )
            else:
                self.early_counter += 1
                logger.info(f"EarlyStop counter: {self.early_counter}/{self.early_patience}")
                if self.early_counter >= self.early_patience:
                    logger.info("Early stopping triggered.")
                    break

            with open(self.log_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Training done. "
            f"Best val cls_acc={self.best_val_cls_acc:.6f}, "
            f"best val same_mean={self.best_val_same_mean:.6f}"
        )