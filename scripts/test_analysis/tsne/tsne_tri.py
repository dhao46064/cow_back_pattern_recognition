import os
import sys
import gc
import time
import random
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
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

from src.sup.models.triplet_model import CowModel_resnet50


# =========================================================
# 2. 变换
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


def list_valid_cow_ids(root_dir: str):
    root = Path(root_dir)
    cow_ids = []
    for p in root.iterdir():
        if p.is_dir():
            cow_ids.append(p.name)
    return sorted(cow_ids)


def list_images_for_cow(root_dir: str, cow_id: str):
    """
    优先读取:
        root/cow_id/normal/*
    否则读取:
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
# 4. checkpoint 处理（Triplet 专用）
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


def infer_embedding_dim_from_triplet_state_dict(state_dict: dict) -> int:
    """
    自动推断 embedding_dim

    优先级：
    1) classifier.weight 的输入维度（TripletCowModelWithCE）
    2) 最后一个 head BN / Linear 的输出维度
    """
    # 情况1：TripletCowModelWithCE
    if "classifier.weight" in state_dict:
        return int(state_dict["classifier.weight"].shape[1])

    # 情况2：embedder + classifier 结构
    if "embedder.classifier.weight" in state_dict:
        return int(state_dict["embedder.classifier.weight"].shape[1])

    # 情况3：从 head 最后 BN 推断
    candidate_keys = [
        "head.5.weight",
        "head.9.weight",
        "embedder.head.5.weight",
        "embedder.head.9.weight",
    ]
    for k in candidate_keys:
        if k in state_dict:
            return int(state_dict[k].shape[0])

    # 情况4：从最后 linear 推断
    candidate_keys = [
        "head.4.weight",
        "head.8.weight",
        "embedder.head.4.weight",
        "embedder.head.8.weight",
    ]
    for k in candidate_keys:
        if k in state_dict:
            return int(state_dict[k].shape[0])

    raise KeyError("无法从 Triplet checkpoint 自动推断 embedding_dim，请手动指定。")


def extract_embedder_state_dict(state_dict: dict) -> dict:
    """
    把 checkpoint 统一整理成 CowModel_resnet50 可直接加载的格式。

    支持：
    - backbone.xxx / head.xxx
    - embedder.backbone.xxx / embedder.head.xxx
    - 跳过 classifier.xxx / arcface.xxx
    """
    new_sd = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]

        # 跳过分类头
        if k.startswith("classifier.") or k.startswith("arcface."):
            continue

        # TripletCowModelWithCE 保存出来的 embedder.*
        if k.startswith("embedder."):
            k = k[len("embedder."):]

        # 只保留 CowModel_resnet50 自己的参数
        if k.startswith("backbone.") or k.startswith("head.") or k.startswith("pool."):
            new_sd[k] = v

    return new_sd


def build_triplet_model_from_ckpt(model_path: str, device: str, embedding_dim: int = None):
    ckpt, raw_state_dict = load_checkpoint(model_path)

    if embedding_dim is None:
        embedding_dim = infer_embedding_dim_from_triplet_state_dict(raw_state_dict)

    state_dict = extract_embedder_state_dict(raw_state_dict)

    print("=" * 80)
    print(f"[INFO] model_path  : {model_path}")
    print(f"[INFO] exists      : {os.path.exists(model_path)}")
    print(f"[INFO] file size   : {os.path.getsize(model_path)} bytes")
    print(f"[INFO] file mtime  : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(model_path)))}")
    print(f"[INFO] emb_dim     : {embedding_dim}")
    print(f"[INFO] extracted embedder keys: {len(state_dict)}")
    print("=" * 80)

    model = CowModel_resnet50(
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

    assert len(missing) == 0 and len(unexpected) == 0, "Triplet 模型权重没有正确加载"

    model = model.to(device)
    model.eval()

    print("[INFO] model fingerprint:")
    shown = 0
    for name, param in model.state_dict().items():
        print(f"  {name:35s} shape={tuple(param.shape)!s:20s} sha16={tensor_sha16(param)}")
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
        emb = model(imgs)

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
# 7. 画图
# =========================================================
def plot_tsne(X_2d, labels, save_path: str, title="t-SNE of Triplet Embeddings"):
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        cmap="tab20",
        s=12,
        alpha=0.85,
    )
    plt.colorbar(scatter)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] figure saved to: {save_path}")
    plt.show()
    plt.close()


# =========================================================
# 8. 主函数
# =========================================================
def main():
    set_seed(42)
    fresh_start()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # 这里改成你的路径
    # =========================
    root_dir = r"D:\code_project\triplet_code\data\dataset_OBB"
    model_path = r"D:\code_project\triplet_code\logs\triLoss_obb_emb128_Noenhence_m0.3\checkpoints\best.pth"

    max_cows = 30
    max_per_cow = 200
    batch_size = 64

    cow_ids = list_valid_cow_ids(root_dir)[:max_cows]
    print("[INFO] cow_ids =", cow_ids)

    transform = build_eval_transforms(img_size=224)

    dataset = CowTSNEDataset(
        root_dir=root_dir,
        cow_ids=cow_ids,
        transform=transform,
        max_per_cow=max_per_cow,
        seed=42,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model, ckpt = build_triplet_model_from_ckpt(model_path, device=device, embedding_dim=None)

    X, y, cids, paths = extract_embeddings(model, dataloader, device=device)
    X_2d = run_tsne(X, seed=42, use_pca=True, pca_dim=50, perplexity=30)

    exp_name = Path(model_path).parents[1].name
    save_path = str(PROJECT_ROOT / f"tsne_triplet_{exp_name}.png")
    plot_tsne(X_2d, y, save_path=save_path, title="t-SNE of Triplet Embeddings")


if __name__ == "__main__":
    main()