import os
import shutil
import yaml
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


class CattleFeatureExtractor:
    def __init__(self, model_name="resnet50", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = self._load_pretrained_model()
        self.transform = self._get_transform()

    def _load_pretrained_model(self):
        if self.model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])  # output: [B,2048,1,1]
        elif self.model_name == "vgg16":
            model = models.vgg16(pretrained=True)
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        elif self.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=True)
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        model = model.to(self.device)
        model.eval()
        return model

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract_features(self, image_paths, batch_size=32):
        """
        Return: np.ndarray shape [N, D] with L2-normalized features
        """
        features = []
        valid_paths = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            keep_paths = []

            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = self.transform(img)
                    batch_images.append(img)
                    keep_paths.append(img_path)
                except Exception as e:
                    print(f"[WARN] Failed to load: {img_path} ({e})")

            if not batch_images:
                continue

            batch_tensor = torch.stack(batch_images).to(self.device)

            feats = self.model(batch_tensor)
            feats = feats.view(feats.size(0), -1)        # [B,D]
            feats = F.normalize(feats, p=2, dim=1)       # L2 normalize for cosine stability

            features.append(feats.cpu().numpy())
            valid_paths.extend(keep_paths)

        if not features:
            return np.array([]), []

        return np.vstack(features), valid_paths


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


from pathlib import Path

def list_normal_images(cow_dir: Path):
    """
    Only return NORMAL images for this cow_id.

    Supported input folder structures:
    A) cow_dir/normal/*.jpg  (and maybe cow_dir/abnormal/*.jpg)
    B) cow_dir/*.jpg         (folder contains only normal images)
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    # Case A: has a 'normal' subfolder -> ONLY use it
    normal_dir = cow_dir / "normal"
    if normal_dir.exists() and normal_dir.is_dir():
        paths = []
        for p in normal_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in image_exts:
                paths.append(str(p))
        return sorted(paths)

    # Case B: no 'normal' folder -> use images directly under cow_dir,
    # but DO NOT go into subfolders like abnormal/clean/hard/noisy
    exclude_subdirs = {"abnormal", "clean", "hard", "noisy", "normal"}  # just in case
    paths = []
    for p in cow_dir.iterdir():
        if p.is_dir() and p.name.lower() in exclude_subdirs:
            continue
        if p.is_file() and p.suffix.lower() in image_exts:
            paths.append(str(p))
    return sorted(paths)


def cosine_distance_to_center(features: np.ndarray, robust_center: bool, robust_drop_ratio: float):
    """
    features: [N,D] already L2-normalized
    return:
      center: [D] L2-normalized
      dist: [N] cosine distance = 1 - cos_sim
      sim:  [N] cosine similarity
    """
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("Need at least 2 feature vectors [N,D].")

    # initial center
    center = features.mean(axis=0, keepdims=True)
    center = center / (np.linalg.norm(center, axis=1, keepdims=True) + 1e-12)  # normalize

    sim = (features @ center.T).squeeze(1)     # [N]
    dist = 1.0 - sim                           # [N]

    if robust_center:
        # drop farthest top ratio, recompute center
        n = features.shape[0]
        drop_n = int(round(n * robust_drop_ratio))
        drop_n = max(0, min(drop_n, n - 2))    # keep at least 2 samples

        if drop_n > 0:
            keep_idx = np.argsort(dist)[: n - drop_n]
            f_keep = features[keep_idx]
            center = f_keep.mean(axis=0, keepdims=True)
            center = center / (np.linalg.norm(center, axis=1, keepdims=True) + 1e-12)

            sim = (features @ center.T).squeeze(1)
            dist = 1.0 - sim

    return center.squeeze(0), dist, sim


def bucket_indices_by_quantile(
        dist: np.ndarray,
        q_clean=0.70,
        q_hard=0.90,
        clean_max_dist=0.07,
        noisy_min_dist=0.16
):
    """
    Hybrid bucket strategy:
    - clean : dist <= min(Q70 , clean_max_dist)
    - hard  : clean < dist <= min(Q90 , noisy_min_dist)
    - noisy : dist > noisy_min_dist
    """

    if not (0.0 < q_clean < q_hard < 1.0):
        raise ValueError("Require 0 < q_clean < q_hard < 1")

    t1_q = float(np.quantile(dist, q_clean))
    t2_q = float(np.quantile(dist, q_hard))

    # hybrid thresholds
    t_clean = min(t1_q, clean_max_dist)
    t_noisy = max(t2_q, noisy_min_dist)

    clean_idx = np.where(dist <= t_clean)[0]

    hard_idx = np.where(
        (dist > t_clean) & (dist <= t_noisy)
    )[0]

    noisy_idx = np.where(dist > t_noisy)[0]

    return t_clean, t_noisy, clean_idx, hard_idx, noisy_idx

def write_bucket_folders(cow_id: str, image_paths, dist, sim,
                         clean_idx, hard_idx, noisy_idx,
                         out_root: Path, write_mode="copy", save_dist_csv=True):
    cow_out = out_root / cow_id
    clean_dir = cow_out / "clean"
    hard_dir = cow_out / "hard"
    noisy_dir = cow_out / "noisy"
    clean_dir.mkdir(parents=True, exist_ok=True)
    hard_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    def do_write(src, dst):
        if write_mode == "move":
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)

    def dump(idx_list, target_dir: Path):
        for i in idx_list:
            src = image_paths[i]
            name = os.path.basename(src)
            dst = str(target_dir / name)
            do_write(src, dst)

    dump(clean_idx, clean_dir)
    dump(hard_idx, hard_dir)
    dump(noisy_idx, noisy_dir)

    if save_dist_csv:
        import pandas as pd
        rows = []
        for i, p in enumerate(image_paths):
            bucket = "clean" if i in set(clean_idx) else ("hard" if i in set(hard_idx) else "noisy")
            rows.append({
                "image_path": p,
                "bucket": bucket,
                "cos_sim_to_center": float(sim[i]),
                "cos_dist_to_center": float(dist[i]),
            })
        df = pd.DataFrame(rows).sort_values("cos_dist_to_center")
        df.to_csv(cow_out / "dist_to_center.csv", index=False, encoding="utf-8-sig")


def bucket_single_cow_folder(cow_dir: Path, feature_extractor: CattleFeatureExtractor, config: dict):
    cow_id = cow_dir.name
    print(f"\n[PROCESS] {cow_id}")

    # ✅ only normal
    image_paths = list_normal_images(cow_dir)
    if len(image_paths) < 2:
        print(f"[SKIP] {cow_id}: only {len(image_paths)} NORMAL images")
        return None

    batch_size = int(config["processing"].get("batch_size", 32))
    feats, valid_paths = feature_extractor.extract_features(image_paths, batch_size=batch_size)
    if feats.size == 0 or len(valid_paths) < 2:
        print(f"[ERROR] {cow_id}: feature extraction failed")
        return None

    # 注意：valid_paths 可能比 image_paths 少（有坏图读不了）
    image_paths = valid_paths

    bcfg = config.get("bucket", {})
    q_clean = float(bcfg.get("q_clean", 0.70))
    q_hard = float(bcfg.get("q_hard", 0.90))
    robust_center = bool(bcfg.get("robust_center", True))
    robust_drop_ratio = float(bcfg.get("robust_drop_ratio", 0.10))
    write_mode = str(bcfg.get("write_mode", "copy")).lower()
    save_dist_csv = bool(bcfg.get("save_dist_csv", True))

    center, dist, sim = cosine_distance_to_center(
        feats,
        robust_center=robust_center,
        robust_drop_ratio=robust_drop_ratio
    )

    t1, t2, clean_idx, hard_idx, noisy_idx = bucket_indices_by_quantile(
        dist, q_clean=q_clean, q_hard=q_hard
    )

    out_root = Path(config["paths"]["output_dir"])
    write_bucket_folders(
        cow_id=cow_id,
        image_paths=image_paths,
        dist=dist,
        sim=sim,
        clean_idx=clean_idx,
        hard_idx=hard_idx,
        noisy_idx=noisy_idx,
        out_root=out_root,
        write_mode=write_mode,
        save_dist_csv=save_dist_csv
    )

    stats = {
        "cow_id": cow_id,
        "num_images_input": len(image_paths),
        "q_clean": q_clean,
        "q_hard": q_hard,
        "t_clean": t1,
        "t_noisy": t2,
        "num_clean": int(len(clean_idx)),
        "num_hard": int(len(hard_idx)),
        "num_noisy": int(len(noisy_idx)),
        "robust_center": robust_center,
        "robust_drop_ratio": robust_drop_ratio,
        "write_mode": write_mode,
    }

    print(f"[OK] {cow_id}: clean={stats['num_clean']} hard={stats['num_hard']} noisy={stats['num_noisy']}"
          f" | t1={t1:.4f} t2={t2:.4f}")
    return stats