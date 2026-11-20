from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset

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


def list_images_in_normal(root_dir: str, cow_id: str) -> List[str]:
    d = Path(root_dir) / cow_id / "normal"
    if not d.exists():
        return []
    paths = []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(str(p))
    return paths


def split_cow_ids(root_dir: str, val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    import random
    cows = list_cow_ids(root_dir)
    rng = random.Random(seed)
    rng.shuffle(cows)
    n_val = max(1, int(round(len(cows) * val_ratio)))
    val_ids = cows[:n_val]
    train_ids = cows[n_val:]
    return train_ids, val_ids


class CowTrainDataset(Dataset):
    def __init__(self, root_dir: str, train_cow_ids: List[str], transform=None):
        self.root = root_dir
        self.transform = transform
        self.cow_ids = train_cow_ids
        self.id2label = {cid: i for i, cid in enumerate(self.cow_ids)}

        self.samples = []
        for cid in self.cow_ids:
            for bucket in ("clean", "hard"):
                for p in list_images_in_bucket(self.root, cid, bucket):
                    self.samples.append((p, self.id2label[cid], bucket))

        self.label_to_bucket_paths: Dict[int, Dict[str, List[str]]] = {}
        for path, lab, bucket in self.samples:
            if lab not in self.label_to_bucket_paths:
                self.label_to_bucket_paths[lab] = {"clean": [], "hard": []}
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


class CowTrainDatasetNormal(Dataset):
    def __init__(self, root_dir: str, train_cow_ids: List[str], transform=None):
        self.root = root_dir
        self.transform = transform
        self.cow_ids = train_cow_ids
        self.id2label = {cid: i for i, cid in enumerate(self.cow_ids)}

        self.samples = []
        for cid in self.cow_ids:
            for p in list_images_in_normal(self.root, cid):
                self.samples.append((p, self.id2label[cid]))

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
    def __init__(self, root_dir: str, val_cow_ids: List[str], transform=None):
        self.root = root_dir
        self.transform = transform
        self.val_cow_ids = val_cow_ids

        self.samples = []
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
    def __init__(self, obb_root: str, val_cow_ids: List[str], transform=None):
        self.root = obb_root
        self.transform = transform
        self.val_cow_ids = val_cow_ids

        self.samples = []
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