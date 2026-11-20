from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from src.sup.utils.metrics import prf_from_counts


@torch.no_grad()
def compute_embeddings_normal_and_abnormal(
    model,
    normal_loader,
    abnormal_loader,
    device: str
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    model.eval()
    per_cow: Dict[str, Dict[str, List[np.ndarray]]] = {}

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

    for images, cow_ids in tqdm(abnormal_loader, desc="VAL Embedding (abnormal)"):
        images = images.to(device, non_blocking=True)
        emb = model.forward_emb(images).cpu().numpy()
        for i in range(len(cow_ids)):
            cid = cow_ids[i]
            if cid not in per_cow:
                per_cow[cid] = {"clean": [], "hard": [], "noisy": [], "abnormal": []}
            per_cow[cid]["abnormal"].append(emb[i])

    return per_cow


@torch.no_grad()
def validate_outlier_by_abnormal(
    per_cow: Dict[str, Dict[str, List[np.ndarray]]],
    q_threshold: float = 0.90,
    clean_calib_ratio: float = 0.5,
    seed: int = 42
) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    total_tp = total_fp = total_fn = total_tn = 0
    cows_used = 0

    for cid, buckets in per_cow.items():
        clean = buckets.get("clean", [])
        hard = buckets.get("hard", [])
        noisy = buckets.get("noisy", [])
        abnormal = buckets.get("abnormal", [])

        if len(clean) < 4:
            continue

        cows_used += 1
        clean_arr = np.stack(clean, axis=0)
        idx = np.arange(len(clean_arr))
        rng.shuffle(idx)

        n_calib = max(2, int(len(idx) * clean_calib_ratio))
        calib_idx = idx[:n_calib]
        eval_idx = idx[n_calib:]

        calib = clean_arr[calib_idx]
        eval_clean = clean_arr[eval_idx] if len(eval_idx) > 0 else np.zeros((0, clean_arr.shape[1]), dtype=np.float32)

        center = calib.mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-12)

        def dist_of_X(X: np.ndarray) -> np.ndarray:
            if X.shape[0] == 0:
                return np.array([], dtype=np.float32)
            cos = (X @ center).clip(-1, 1)
            return (1.0 - cos).astype(np.float32)

        d_calib = dist_of_X(calib)
        thr = float(np.quantile(d_calib, q_threshold))

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