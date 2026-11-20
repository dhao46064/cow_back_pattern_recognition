import logging
from pathlib import Path

import torch

from src.sup.engine.validator import (
    compute_embeddings_normal_and_abnormal,
    validate_outlier_by_abnormal,
)
from src.sup.utils.common import save_json

logger = logging.getLogger("cow_newid_outlier")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader_normal,
        val_loader_abnormal,
        triplet_loss,
        ce_loss,
        ce_weight,
        optimizer,
        device,
        log_dir="./logs",
        exp_name="cow_newid_outlier",
        early_patience=10,
        clean_calib_ratio=0.5,
        seed=42,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader_normal = val_loader_normal
        self.val_loader_abnormal = val_loader_abnormal

        self.triplet_loss = triplet_loss
        self.ce_loss = ce_loss
        self.ce_weight = ce_weight
        self.optimizer = optimizer
        self.device = device

        self.clean_calib_ratio = clean_calib_ratio
        self.seed = seed

        self.log_dir = Path(log_dir) / exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.log_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)

        self.best_f1 = -1.0
        self.early_patience = early_patience
        self.early_counter = 0

        self.history = {"train": [], "val_outlier": []}

    def save_ckpt(self, epoch: int, tag: str):
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history,
        }
        path = self.ckpt_dir / f"{tag}.pth"
        torch.save(ckpt, path)
        logger.info(f"[ckpt] saved: {path}")

    def train_one_epoch(self, epoch: int):
        from tqdm import tqdm

        self.model.train()
        total_loss = total_tri = total_ce = total_acc = 0.0
        n = 0

        pbar = tqdm(self.train_loader, desc=f"TRAIN Epoch {epoch}")
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            emb, logits = self.model(images)

            loss_tri = self.triplet_loss(emb, labels)
            loss_ce = self.ce_loss(logits, labels)
            loss = loss_tri + self.ce_weight * loss_ce

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == labels).float().mean().item()

            total_loss += float(loss.item())
            total_tri += float(loss_tri.item())
            total_ce += float(loss_ce.item())
            total_acc += float(acc)
            n += 1

            pbar.set_postfix({
                "loss": f"{total_loss / n:.4f}",
                "tri": f"{total_tri / n:.4f}",
                "ce": f"{total_ce / n:.4f}",
                "acc": f"{total_acc / n:.3f}",
            })

        m = {
            "loss": total_loss / n,
            "tri": total_tri / n,
            "ce": total_ce / n,
            "acc": total_acc / n,
        }
        self.history["train"].append(m)
        logger.info(f"[train] loss={m['loss']:.4f} tri={m['tri']:.4f} ce={m['ce']:.4f} acc={m['acc']:.3f}")

    def validate_outlier(self, epoch: int, q_threshold: float):
        per_cow = compute_embeddings_normal_and_abnormal(
            self.model, self.val_loader_normal, self.val_loader_abnormal, self.device
        )
        metrics = validate_outlier_by_abnormal(
            per_cow,
            q_threshold=q_threshold,
            clean_calib_ratio=self.clean_calib_ratio,
            seed=self.seed,
        )

        self.history["val_outlier"].append(metrics)
        logger.info(
            f"[val_abnormal] epoch={epoch} cows_used={int(metrics['val_cows_used'])} "
            f"P={metrics['precision_abnormal']:.3f} R={metrics['recall_abnormal']:.3f} "
            f"F1={metrics['f1_abnormal']:.3f} tp={int(metrics['tp'])} "
            f"fp={int(metrics['fp'])} fn={int(metrics['fn'])}"
        )
        return metrics

    def fit(self, epochs: int, q_threshold: float):
        for ep in range(1, epochs + 1):
            logger.info("=" * 70)
            logger.info(f"Epoch {ep}/{epochs}")

            self.train_one_epoch(ep)
            valm = self.validate_outlier(ep, q_threshold=q_threshold)

            self.save_ckpt(ep, tag=f"epoch_{ep}")

            f1 = valm["f1_abnormal"]
            if f1 > self.best_f1 + 1e-6:
                self.best_f1 = f1
                self.save_ckpt(ep, tag="best")
                self.early_counter = 0
            else:
                self.early_counter += 1
                logger.info(f"EarlyStop counter: {self.early_counter}/{self.early_patience}")
                if self.early_counter >= self.early_patience:
                    logger.info("Early stopping triggered.")
                    break

            save_json(self.history, self.log_dir / "history.json")

        logger.info(f"Training done. Best val F1(abnormal): {self.best_f1:.3f}")