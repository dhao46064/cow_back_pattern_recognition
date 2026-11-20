import os


import json
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

logger = logging.getLogger("cow_newid_outlier")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =========================
# 1) Model (Embedding + optional CE head)
# =========================
class TripletCowModel(nn.Module):
    """
    Output: L2-normalized embedding [B, D]
    """
    def __init__(self, embedding_dim: int = 512,
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 hidden_dims=(1024, 512),
                 dropout: float = 0.3):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        feature_dim = 2048

        layers = []
        in_dim = feature_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = h

        layers += [
            nn.Linear(in_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        ]
        self.head = nn.Sequential(*layers)

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        pooled = self.pool(feat).flatten(1)
        emb = self.head(pooled)
        emb = F.normalize(emb, p=2, dim=1)
        return emb


class TripletCowModelWithCE(nn.Module):
    """
    Training: return (emb, logits). Validation: use emb only.
    """
    def __init__(self, num_train_classes: int,
                 embedding_dim: int = 512,
                 pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()
        self.embedder = TripletCowModel(
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        self.classifier = nn.Linear(embedding_dim, num_train_classes)

    def forward(self, x: torch.Tensor):
        emb = self.embedder(x)
        logits = self.classifier(emb)
        return emb, logits

    def forward_emb(self, x: torch.Tensor):
        return self.embedder(x)


# =========================
# 2) Loss (Batch-hard Triplet)
# =========================
class BatchHardTripletLoss(nn.Module):
    """
    embeddings: [B, D] (L2-normalized)
    labels: [B]
    """
    def __init__(self, margin=0.7):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(embeddings, embeddings, p=2)  # [B,B]
        B = labels.size(0)
        lab = labels.view(B, 1)

        is_pos = (lab == lab.t())
        is_neg = (lab != lab.t())

        diag = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        is_pos = is_pos & (~diag)

        dist_pos = dist.clone()
        dist_pos[~is_pos] = -1e9
        hardest_pos, _ = dist_pos.max(dim=1)

        dist_neg = dist.clone()
        dist_neg[~is_neg] = 1e9
        hardest_neg, _ = dist_neg.min(dim=1)

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        # print(f"hardest_pos:{hardest_pos},hardest_neg:{hardest_neg},total:{hardest_pos - hardest_neg + self.margin}")

        valid = hardest_pos > -1e8  # ensure K>=2
        return loss[valid].mean() if valid.any() else loss.mean() * 0.0


# =========================
# 3) Dataset helpers
# =========================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_cow_ids(root_dir: str) -> List[str]:
    root = Path(root_dir)
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def list_images_in_bucket(root_dir: str, cow_id: str, bucket: str) -> List[str]:
    d = Path(root_dir) / cow_id / bucket
    if not d.exists():
        return []
    paths = []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(str(p))
    return paths



class CowTrainDataset(Dataset):
    """
    Train dataset over train_cows, only buckets: clean/hard
    Returns: (img, label_train_class)
    """
    def __init__(self, root_dir: str, train_cow_ids: List[str], transform=None):
        self.root = root_dir
        self.transform = transform

        self.cow_ids = train_cow_ids
        self.id2label = {cid: i for i, cid in enumerate(self.cow_ids)}

        self.samples = []  # (path, label, bucket)
        for cid in self.cow_ids:
            for bucket in ("clean", "hard"):
                for p in list_images_in_bucket(self.root, cid, bucket):
                    self.samples.append((p, self.id2label[cid], bucket))

        # label -> {"clean":[], "hard":[]}
        self.label_to_bucket_paths: Dict[int, Dict[str, List[str]]] = {}
        for path, lab, bucket in self.samples:
            if lab not in self.label_to_bucket_paths:
                self.label_to_bucket_paths[lab] = {"clean": [], "hard": []}
            if bucket in self.label_to_bucket_paths[lab]:
                self.label_to_bucket_paths[lab][bucket].append(path)

        self.path_to_index = {path: i for i, (path, _, _) in enumerate(self.samples)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _bucket = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def list_images_in_normal(root_dir: str, cow_id: str) -> List[str]:
    """
    root_dir/cow_id/normal 下所有图片。
    兼容 normal 下直接是图片、或者 normal/clean|hard|noisy 的子结构。
    """
    d = Path(root_dir) / cow_id / "normal"
    if not d.exists():
        return []
    paths = []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(str(p))
    return paths
class CowTrainDatasetNormal(Dataset):
    """
    Train dataset over train_cows, only from normal/ (no bucket bias).
    Returns: (img, label_train_class)
    """
    def __init__(self, root_dir: str, train_cow_ids: List[str], transform=None):
        self.root = root_dir
        self.transform = transform

        self.cow_ids = train_cow_ids
        self.id2label = {cid: i for i, cid in enumerate(self.cow_ids)}

        self.samples = []  # (path, label)
        for cid in self.cow_ids:
            for p in list_images_in_normal(self.root, cid):
                self.samples.append((p, self.id2label[cid]))

        # label -> [paths...]
        self.label_to_paths: Dict[int, List[str]] = {}
        for path, lab in self.samples:
            self.label_to_paths.setdefault(lab, []).append(path)

        self.path_to_index = {path: i for i, (path, _) in enumerate(self.samples)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
class CowValAllBucketsDataset(Dataset):
    """
    Val dataset reading clean/hard/noisy from data_root (ALL treated as normal difficulty levels).
    Returns: (img, cow_id_str, bucket_str)
    """
    def __init__(self, root_dir: str, val_cow_ids: List[str], transform=None):
        self.root = root_dir
        self.transform = transform
        self.val_cow_ids = val_cow_ids

        self.samples = []  # (path, cow_id, bucket)
        for cid in self.val_cow_ids:
            for bucket in ("clean", "hard", "noisy"):
                for p in list_images_in_bucket(self.root, cid, bucket):
                    self.samples.append((p, cid, bucket))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cid, bucket = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, cid, bucket


class CowValAbnormalDataset(Dataset):
    """
    Read abnormal images from dataset_OBB/<cid>/abnormal
    Returns: (img, cid)
    """
    def __init__(self, obb_root: str, val_cow_ids: List[str], transform=None):
        self.root = obb_root
        self.transform = transform
        self.val_cow_ids = val_cow_ids

        self.samples = []  # (path, cid)
        for cid in self.val_cow_ids:
            d = Path(self.root) / cid / "abnormal"
            if not d.exists():
                continue
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    self.samples.append((str(p), cid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, cid


# =========================
# 4) PK Sampler for train (batch=8 => P=4,K=2)
# =========================
class PKBatchSampler(Sampler):
    """
    PK sampler supporting both:
    - CowTrainDataset (clean/hard buckets): dataset.label_to_bucket_paths
    - CowTrainDatasetNormal (normal only): dataset.label_to_paths
    """
    def __init__(self, dataset: Dataset, P=4, K=2, steps_per_epoch=2500, seed=42, prefer_clean_hard_pair=True):
        self.ds = dataset
        self.P = P
        self.K = K
        self.steps = steps_per_epoch
        self.rng = random.Random(seed)
        self.prefer_clean_hard_pair = prefer_clean_hard_pair

        self.mode = None
        if hasattr(self.ds, "label_to_bucket_paths"):
            self.mode = "bucket"
            # labels that have >=K total images
            self.labels = []
            for lab, buckets in self.ds.label_to_bucket_paths.items():
                n = len(buckets.get("clean", [])) + len(buckets.get("hard", []))
                if n >= K:
                    self.labels.append(lab)
        elif hasattr(self.ds, "label_to_paths"):
            self.mode = "normal"
            self.labels = [lab for lab, paths in self.ds.label_to_paths.items() if len(paths) >= K]
        else:
            raise ValueError("Dataset must have label_to_bucket_paths or label_to_paths.")

        assert len(self.labels) >= P, f"Not enough train IDs for P={P}, need at least {P}."

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            chosen = self.rng.sample(self.labels, self.P)
            batch_indices = []

            for lab in chosen:
                if self.mode == "bucket":
                    clean = self.ds.label_to_bucket_paths[lab]["clean"]
                    hard = self.ds.label_to_bucket_paths[lab]["hard"]

                    # 旧逻辑：K=2 时强制 1 clean + 1 hard（可用 prefer_clean_hard_pair 控制）
                    if self.prefer_clean_hard_pair and self.K == 2 and len(clean) >= 1 and len(hard) >= 1:
                        pick_paths = [self.rng.choice(clean), self.rng.choice(hard)]  #p=2 全部进如采样
                    else:
                        pool = clean + hard
                        pick_paths = self.rng.sample(pool, self.K)

                else:  # normal
                    pool = self.ds.label_to_paths[lab]
                    pick_paths = self.rng.sample(pool, self.K)

                for p in pick_paths:
                    batch_indices.append(self.ds.path_to_index[p])

            yield batch_indices


# =========================
# 5) Split cows (new-id eval)
# =========================
def split_cow_ids(root_dir: str, val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    cows = list_cow_ids(root_dir)
    rng = random.Random(seed)
    rng.shuffle(cows)
    n_val = max(1, int(round(len(cows) * val_ratio)))
    val_ids = cows[:n_val]
    train_ids = cows[n_val:]
    return train_ids, val_ids


# =========================
# 6) Validation: prototype from clean; abnormal from dataset_OBB
# =========================
@torch.no_grad()
def compute_embeddings_normal_and_abnormal(
    model: TripletCowModelWithCE,
    normal_loader: DataLoader,
    abnormal_loader: DataLoader,
    device: str
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    per_cow[cid] = {
        "clean": [emb...],
        "hard": [emb...],
        "noisy": [emb...],    # still normal (harder)
        "abnormal": [emb...]  # true abnormal from dataset_OBB
    }
    """
    model.eval()
    per_cow: Dict[str, Dict[str, List[np.ndarray]]] = {}

    # normal buckets from data_root

    ###为了生成类别，就是很单纯的得到normal以及abnormal，生成了字典
    for images, cow_ids, buckets in tqdm(normal_loader, desc="VAL Embedding (normal buckets)"):
        images = images.to(device, non_blocking=True)
        emb = model.forward_emb(images).cpu().numpy()
        for i in range(len(cow_ids)):
            cid = cow_ids[i]
            b = buckets[i]
            if cid not in per_cow:
                per_cow[cid] = {"clean": [], "hard": [], "noisy": [], "abnormal": []}
            if b in per_cow[cid]:
                per_cow[cid][b].append(emb[i])

    # abnormal from dataset_OBB
    for images, cow_ids in tqdm(abnormal_loader, desc="VAL Embedding (abnormal)"):
        images = images.to(device, non_blocking=True)
        emb = model.forward_emb(images).cpu().numpy()
        for i in range(len(cow_ids)):
            cid = cow_ids[i]
            if cid not in per_cow:
                per_cow[cid] = {"clean": [], "hard": [], "noisy": [], "abnormal": []}
            per_cow[cid]["abnormal"].append(emb[i])

    return per_cow


def prf_from_counts(tp, fp, fn) -> Tuple[float, float, float]:
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(prec), float(rec), float(f1)


@torch.no_grad()
def validate_outlier_by_abnormal(
    per_cow: Dict[str, Dict[str, List[np.ndarray]]],
    q_threshold: float = 0.90,
    clean_calib_ratio: float = 0.5,
    seed: int = 42
) -> Dict[str, float]:
    """
    Prototype center computed ONLY from clean (calib subset).
    normal test = (remaining clean) + hard + noisy  [GT=0]
    abnormal test = abnormal from dataset_OBB       [GT=1]

    threshold = quantile(dist(calib_clean), q_threshold)
    dist = 1 - cosine(emb, center)   (both normalized)
    """
    rng = np.random.RandomState(seed)
    total_tp = total_fp = total_fn = total_tn = 0
    cows_used = 0

    for cid, buckets in per_cow.items():
        clean = buckets.get("clean", [])
        hard = buckets.get("hard", [])
        noisy = buckets.get("noisy", [])
        abnormal = buckets.get("abnormal", [])

        # clean 太少，center+thr不稳
        if len(clean) < 4:
            continue

        cows_used += 1

        clean_arr = np.stack(clean, axis=0)  # [Nc, D]
        idx = np.arange(len(clean_arr))
        rng.shuffle(idx)

        n_calib = max(2, int(len(idx) * clean_calib_ratio))
        calib_idx = idx[:n_calib]
        eval_idx = idx[n_calib:]

        calib = clean_arr[calib_idx]  # [Ncalib, D]
        eval_clean = clean_arr[eval_idx] if len(eval_idx) > 0 else np.zeros((0, clean_arr.shape[1]), dtype=np.float32)

        # 特征中心：prototype = mean(calib clean)
        center = calib.mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-12)

        def dist_of_X(X: np.ndarray) -> np.ndarray:
            """
            X: [N,D] normalized; center: [D] normalized
            return dist: [N]
            """
            if X.shape[0] == 0:
                return np.array([], dtype=np.float32)
            cos = (X @ center).clip(-1, 1)
            return (1.0 - cos).astype(np.float32)

        # threshold from calib clean distances
        d_calib = dist_of_X(calib)
        thr = float(np.quantile(d_calib, q_threshold))

        # normal test: eval_clean + hard + noisy  (all GT=0)
        normal_embs = []
        if eval_clean.shape[0] > 0:
            normal_embs.append(eval_clean)
        if len(hard) > 0:
            normal_embs.append(np.stack(hard, axis=0))
        if len(noisy) > 0:
            normal_embs.append(np.stack(noisy, axis=0))

        Xn = np.concatenate(normal_embs, axis=0) if len(normal_embs) > 0 else np.zeros((0, clean_arr.shape[1]), dtype=np.float32)
        d_normal = dist_of_X(Xn)
        pred_normal = d_normal > thr
        fp = int(pred_normal.sum())
        tn = int((~pred_normal).sum())

        # abnormal test: GT=1
        Xa = np.stack(abnormal, axis=0) if len(abnormal) > 0 else np.zeros((0, clean_arr.shape[1]), dtype=np.float32)
        d_ab = dist_of_X(Xa)
        pred_ab = d_ab > thr
        tp = int(pred_ab.sum())
        fn = int((~pred_ab).sum())

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    prec, rec, f1 = prf_from_counts(total_tp, total_fp, total_fn)
    return {
        "val_cows_used": float(cows_used),
        "precision_abnormal": prec,
        "recall_abnormal": rec,
        "f1_abnormal": f1,
        "tp": float(total_tp),
        "fp": float(total_fp),
        "fn": float(total_fn),
        "tn": float(total_tn),
    }


# =========================
# 7) Trainer
# =========================
class Trainer:
    def __init__(self,
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
                 seed=42):
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
                "loss": f"{total_loss/n:.4f}",
                "tri": f"{total_tri/n:.4f}",
                "ce": f"{total_ce/n:.4f}",
                "acc": f"{total_acc/n:.3f}",
            })

        m = {"loss": total_loss/n, "tri": total_tri/n, "ce": total_ce/n, "acc": total_acc/n}
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
            seed=self.seed
        )

        self.history["val_outlier"].append(metrics)
        logger.info(
            f"[val_abnormal] epoch={epoch} cows_used={int(metrics['val_cows_used'])} "
            f"P={metrics['precision_abnormal']:.3f} R={metrics['recall_abnormal']:.3f} "
            f"F1={metrics['f1_abnormal']:.3f} tp={int(metrics['tp'])} fp={int(metrics['fp'])} fn={int(metrics['fn'])}"
        )
        return metrics

    def fit(self, epochs: int, q_threshold: float):
        for ep in range(1, epochs + 1):
            logger.info("=" * 70)
            logger.info(f"Epoch {ep}/{epochs}")

            self.train_one_epoch(ep)
            valm = self.validate_outlier(ep, q_threshold=q_threshold)

            # save every epoch
            self.save_ckpt(ep, tag=f"epoch_{ep}")

            # early stop by F1(abnormal)
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

            # save history
            with open(self.log_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)

        logger.info(f"Training done. Best val F1(abnormal): {self.best_f1:.3f}")


# =========================
# 8) Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default=r"D:\code_project\code_cp\data_supervised_output_cleanhardnoisy")
    parser.add_argument("--obb_root", type=str,
                        default=r"D:\code_project\code_cp\dataset_OBB")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--val_batch_size", type=int, default=64)

    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--ce_weight", type=float, default=1)
    parser.add_argument("--train_source", type=str, default="normal", choices=["cleanhard", "normal"])
    parser.add_argument("--train_root_normal", type=str, default=r"D:\code_project\code_cp\dataset_OBB")
    parser.add_argument("--prefer_clean_hard_pair", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--exp_name", type=str, default="cow_3.3_outlier_3.1")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)  # Windows建议先0，稳定

    parser.add_argument("--q_threshold", type=float, default=0.9)  # calib clean dist quantile；决定abnormal为pre_abnormal的阈值
    parser.add_argument("--clean_calib_ratio", type=float, default=0.5)  # split clean into calib/eval
    parser.add_argument("--early_patience", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU.")
        device = "cpu"

    # split cows for NEW-ID evaluation (based on data_root cow list)
    train_cows, val_cows = split_cow_ids(args.data_root, val_ratio=args.val_ratio, seed=args.seed)
    logger.info(f"Total cows: {len(train_cows) + len(val_cows)} | train cows: {len(train_cows)} | val cows: {len(val_cows)}")

    # transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # datasets
    train_ds = CowTrainDataset(args.data_root, train_cows, transform=transform)
    val_ds_normal = CowValAllBucketsDataset(args.data_root, val_cows, transform=transform)     # clean/hard/noisy (all normal)
    val_ds_ab = CowValAbnormalDataset(args.obb_root, val_cows, transform=transform)            # abnormal

    # loaders
    P, K = 4, 2  # batch=8
    if args.train_source == "cleanhard":
        train_ds = CowTrainDataset(args.data_root, train_cows, transform=transform)
        train_sampler = PKBatchSampler(
            train_ds, P=P, K=K,
            steps_per_epoch=args.steps_per_epoch,
            seed=args.seed,
            prefer_clean_hard_pair=args.prefer_clean_hard_pair
        )
    else:
        train_ds = CowTrainDatasetNormal(args.train_root_normal, train_cows, transform=transform)
        train_sampler = PKBatchSampler(
            train_ds, P=P, K=K,
            steps_per_epoch=args.steps_per_epoch,
            seed=args.seed,
            prefer_clean_hard_pair=False  # normal 模式不用这个
        )
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True)

    val_loader_normal = DataLoader(val_ds_normal, batch_size=args.val_batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=True)

    val_loader_abnormal = DataLoader(val_ds_ab, batch_size=args.val_batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)

    # model (CE head only for train cows)
    num_train_classes = len(train_cows)
    model = TripletCowModelWithCE(
        num_train_classes=num_train_classes,
        embedding_dim=args.embedding_dim,
        pretrained=True,
        freeze_backbone=args.freeze_backbone
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
        seed=args.seed
    )

    trainer.fit(args.epochs, q_threshold=args.q_threshold)


if __name__ == "__main__":
    main()