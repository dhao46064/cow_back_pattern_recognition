import os
import sys
import gc
import time
import random
import hashlib
from pathlib import Path
from typing import List
import json

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# =========================================================
# 0. 基础配置
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fresh_start():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    plt.close("all")


def tensor_sha16(t: torch.Tensor) -> str:
    arr = t.detach().cpu().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()[:16]


# =========================================================
# 1. 路径设置
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.sup.models.arcface_model import ArcFaceCowModel
from src.sup.datasets.split_cow import get_cow_ids


# =========================================================
# 2. 变换（固定推理版，不用随机增强）
# =========================================================
def build_eval_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# =========================================================
# 3. 数据扫描
# =========================================================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def list_images_for_cow(root_dir: str, cow_id: str):
    """
    优先读:
        root/cow_id/normal/*
    如果没有 normal 目录，就读:
        root/cow_id/*
    """
    cow_dir = Path(root_dir) / cow_id
    normal_dir = cow_dir / "normal"

    files = []
    if normal_dir.exists() and normal_dir.is_dir():
        for p in normal_dir.iterdir():
            if p.is_file() and is_image_file(p):
                files.append(str(p))
    else:
        for p in cow_dir.iterdir():
            if p.is_file() and is_image_file(p):
                files.append(str(p))

    return sorted(files)


def print_most_confused_centers(report: dict, topn: int = 20):
    rows = []
    for cid, info in report.items():
        rows.append((cid, info["num_wrong"]))

    rows = sorted(rows, key=lambda x: x[1], reverse=True)

    print("\n[INFO] Most confused centers (by wrong samples in topk-nearest):")
    for cid, num_wrong in rows[:topn]:
        print(f"  center={cid} | wrong_in_topk={num_wrong}")


def plot_tsne_with_centers(X_2d, labels, cids, save_path: str, title_suffix=""):
    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        cmap="tab20",
        s=12,
        alpha=0.75,
    )

    uniq_cids = sorted(set(cids))
    for cid in uniq_cids:
        idx = [i for i, x in enumerate(cids) if x == cid]
        pts = X_2d[idx]
        if len(pts) == 0:
            continue
        center = pts.mean(axis=0)

        plt.text(
            center[0],
            center[1],
            str(cid),
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="gray",
                alpha=0.7
            )
        )

    plt.colorbar(scatter)
    plt.title(f"t-SNE of Cow Embeddings {title_suffix}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] figure saved to: {save_path}")
    plt.show()
    plt.close()


def save_wrong_paths_near_each_center(
    X: np.ndarray,
    cids: List[str],
    paths: List[str],
    save_path: str,
    topk: int = 190
):
    """
    对每个 cow 的高维 center：
    - 找全体样本里最近的 topk 张
    - 只保存其中“不是本 cow_id”的错误样本路径
    """
    X = X.astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    uniq_cids = sorted(set(cids))
    result = {}

    cids_arr = np.array(cids)

    for cid in uniq_cids:
        idx = np.where(cids_arr == cid)[0]
        if len(idx) == 0:
            continue

        feats = X[idx]
        center = feats.mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-12)

        sims = X @ center
        dists = 1.0 - sims

        order = np.argsort(dists)[:topk]

        wrong_samples = []
        for rank, j in enumerate(order, start=1):
            sample_cid = cids[j]
            if sample_cid != cid:
                wrong_samples.append({
                    "rank": int(rank),
                    "path": paths[j],
                    "cid": sample_cid,
                    "dist": float(dists[j]),
                })

        result[cid] = {
            "num_wrong": len(wrong_samples),
            "wrong_samples": wrong_samples
        }

        print(f"[INFO] center={cid} | top{topk} wrong={len(wrong_samples)}")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[INFO] wrong-path report saved to: {save_path}")
    return result


class CowTSNEDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        cow_ids,
        transform=None,
        max_per_cow=50,
        seed=42,
    ):
        self.transform = transform
        self.samples = []
        self.id2label = {cid: i for i, cid in enumerate(cow_ids)}

        rng = random.Random(seed)

        for cid in cow_ids:
            paths = list_images_for_cow(root_dir, cid)

            if len(paths) == 0:
                print(f"[WARN] no images for cow_id={cid}")
                continue

            if len(paths) > max_per_cow:
                paths = rng.sample(paths, max_per_cow)

            for p in paths:
                self.samples.append((p, self.id2label[cid], cid))

        rng.shuffle(self.samples)

        print(f"[INFO] total samples = {len(self.samples)}")
        print(f"[INFO] cows used = {len(cow_ids)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, cid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, cid, path


# =========================================================
# 4. 从 checkpoint 自动推断模型配置
# =========================================================
def load_checkpoint(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    return ckpt, state_dict


def infer_arcface_config(state_dict: dict):
    if "arcface.weight" not in state_dict:
        raise KeyError("checkpoint 中没有 arcface.weight，无法自动推断 num_classes / embedding_dim")

    arc_w = state_dict["arcface.weight"]
    num_classes = arc_w.shape[0]
    embedding_dim = arc_w.shape[1]
    return num_classes, embedding_dim


def build_model_from_ckpt(model_path: str, device: str):
    ckpt, state_dict = load_checkpoint(model_path)
    num_classes, embedding_dim = infer_arcface_config(state_dict)

    print("=" * 80)
    print(f"[INFO] model_path  : {model_path}")
    print(f"[INFO] exists      : {os.path.exists(model_path)}")
    print(f"[INFO] file size   : {os.path.getsize(model_path)} bytes")
    print(f"[INFO] file mtime  : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(model_path)))}")
    print(f"[INFO] num_classes : {num_classes}")
    print(f"[INFO] emb_dim     : {embedding_dim}")
    print("=" * 80)

    model = ArcFaceCowModel(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained=False,
        freeze_backbone=False,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("[INFO] missing keys   :", len(missing))
    print("[INFO] unexpected keys:", len(unexpected))

    if len(missing) > 0:
        print("[INFO] first missing:", missing[:10])
    if len(unexpected) > 0:
        print("[INFO] first unexpected:", unexpected[:10])

    if len(missing) > 0 or len(unexpected) > 0:
        print("\n[WARN] 当前本地模型源码与 checkpoint 结构不完全一致。")
        print("[WARN] 这种情况下 t-SNE 结果不可信。")
        print("[WARN] 你需要确保 src/sup/models 下的源码就是训练该 checkpoint 时的版本。\n")

    model = model.to(device)
    model.eval()

    print("[INFO] model fingerprint:")
    shown = 0
    for name, param in model.state_dict().items():
        print(
            f"  {name:45s} shape={tuple(param.shape)!s:20s} sha16={tensor_sha16(param)}"
        )
        shown += 1
        if shown >= 8:
            break

    return model, ckpt


# =========================================================
# 5. 提 embedding
# =========================================================
@torch.no_grad()
def extract_embeddings(model, dataloader, device: str):
    all_emb = []
    all_labels = []
    all_cids = []
    all_paths = []

    for imgs, labels, cids, paths in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        emb = model.forward_emb(imgs)

        all_emb.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())
        all_cids.extend(list(cids))
        all_paths.extend(list(paths))

    X = np.concatenate(all_emb, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print("\n[INFO] Embedding summary:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X[:3, :8] =\n", X[:3, :8])
    print("embedding mean =", float(X.mean()))
    print("embedding std  =", float(X.std()))
    print("embedding sha16 =", hashlib.sha256(X[:20].tobytes()).hexdigest()[:16])

    return X, y, all_cids, all_paths


# =========================================================
# 6. t-SNE
# =========================================================
def run_tsne(X, seed=42, use_pca=True, pca_dim=50, perplexity=30):
    X_in = X

    if use_pca and X.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        X_in = pca.fit_transform(X)
        print(f"[INFO] PCA -> {X_in.shape}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        n_iter=1000,
        init="pca",
    )
    X_2d = tsne.fit_transform(X_in)
    print("[INFO] TSNE done. X_2d[:5] =\n", X_2d[:5])
    return X_2d


# =========================================================
# 7. 主逻辑
# =========================================================
def run_tsne_analysis(
    root_dir: str,
    model_path: str,
    split_mode: str = "seen",   # "seen" or "unseen"
    max_per_cow: int = 200,
    batch_size: int = 64,
    seed: int = 42,
):
    set_seed(seed)
    fresh_start()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if split_mode == "seen":
        cow_ids = get_cow_ids(root_dir, split="seen_cows")
    elif split_mode == "unseen":
        cow_ids = get_cow_ids(root_dir, split="unseen_cows")
    else:
        raise ValueError("split_mode must be 'seen' or 'unseen'")

    print(f"[INFO] split_mode = {split_mode}")
    print(f"[INFO] num cows   = {len(cow_ids)}")
    print(f"[INFO] cows       = {cow_ids[:10]} ...")

    transform = build_eval_transforms(img_size=224)

    dataset = CowTSNEDataset(
        root_dir=root_dir,
        cow_ids=cow_ids,
        transform=transform,
        max_per_cow=max_per_cow,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model, ckpt = build_model_from_ckpt(model_path, device=device)

    X, y, cids, paths = extract_embeddings(model, dataloader, device=device)
    X_2d = run_tsne(X, seed=seed, use_pca=True, pca_dim=50, perplexity=30)

    exp_name = Path(model_path).parents[1].name

    # 方案一：带 cow_id 中心编号的 t-SNE 图
    tsne_save_path = str(PROJECT_ROOT / f"tsne_{split_mode}_{exp_name}_{split_mode}.png")
    plot_tsne_with_centers(
        X_2d, y, cids,
        save_path=tsne_save_path,
        title_suffix=f"({split_mode})"
    )

    # 方案二：每个 center 最近 topk 张图里，只保存错误地址
    wrong_save_path = str(PROJECT_ROOT / f"tsne_overlap_wrong_paths_{split_mode}_{exp_name}_{split_mode}.json")
    report = save_wrong_paths_near_each_center(
        X=X,
        cids=cids,
        paths=paths,
        save_path=wrong_save_path,
        topk=190
    )

    print_most_confused_centers(report, topn=20)


# =========================================================
# 8. main
# =========================================================
def main():
    root_dir = r"D:\code_project\triplet_code\data\dataset_OBB"
    model_path = r"D:\code_project\triplet_code\logs\exp003_7.22_proxyanchor_baseline\checkpoints\latest.pth"

    # 二选一：
    # split_mode = "seen"
    split_mode = "unseen"

    run_tsne_analysis(
        root_dir=root_dir,
        model_path=model_path,
        split_mode=split_mode,
        max_per_cow=200,
        batch_size=64,
        seed=42,
    )


if __name__ == "__main__":
    main()