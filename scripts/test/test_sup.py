# # import os
# # import sys
# # import json
# # import csv
# # import random
# # import shutil
# # from pathlib import Path
# # from typing import List, Dict

# # import numpy as np
# # from PIL import Image

# # import torch
# # from torch.utils.data import Dataset, DataLoader
# # from torchvision import transforms


# # # =========================================================
# # # 0. 基础配置
# # # =========================================================
# # def set_seed(seed=42):
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     torch.cuda.manual_seed_all(seed)


# # IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# # def is_image_file(path: Path) -> bool:
# #     return path.suffix.lower() in IMG_EXTS


# # # =========================================================
# # # 1. 路径设置
# # # =========================================================
# # PROJECT_ROOT = Path(__file__).resolve().parents[1]
# # if str(PROJECT_ROOT) not in sys.path:
# #     sys.path.append(str(PROJECT_ROOT))

# # from src.sup.models.arcface_model import ArcFaceCowModel


# # # =========================================================
# # # 2. 图像变换
# # # =========================================================
# # def build_eval_transforms(img_size=224):
# #     return transforms.Compose([
# #         transforms.Resize((img_size, img_size)),
# #         transforms.ToTensor(),
# #         transforms.Normalize(
# #             mean=[0.485, 0.456, 0.406],
# #             std=[0.229, 0.224, 0.225],
# #         ),
# #     ])


# # # =========================================================
# # # 3. checkpoint 加载
# # # =========================================================
# # def load_checkpoint(model_path: str):
# #     ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
# #     if isinstance(ckpt, dict) and "model" in ckpt:
# #         state_dict = ckpt["model"]
# #     elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
# #         state_dict = ckpt["model_state_dict"]
# #     elif isinstance(ckpt, dict) and "state_dict" in ckpt:
# #         state_dict = ckpt["state_dict"]
# #     else:
# #         state_dict = ckpt
# #     return ckpt, state_dict


# # def infer_arcface_config(state_dict: dict):
# #     if "arcface.weight" not in state_dict:
# #         raise KeyError("checkpoint 中没有 arcface.weight，无法自动推断 num_classes / embedding_dim")
# #     arc_w = state_dict["arcface.weight"]
# #     num_classes = arc_w.shape[0]
# #     embedding_dim = arc_w.shape[1]
# #     return num_classes, embedding_dim


# # def build_model_from_ckpt(model_path: str, device: str):
# #     _, state_dict = load_checkpoint(model_path)
# #     num_classes, embedding_dim = infer_arcface_config(state_dict)

# #     print("=" * 80)
# #     print(f"[INFO] model_path  : {model_path}")
# #     print(f"[INFO] num_classes : {num_classes}")
# #     print(f"[INFO] emb_dim     : {embedding_dim}")
# #     print("=" * 80)

# #     model = ArcFaceCowModel(
# #         num_classes=num_classes,
# #         embedding_dim=embedding_dim,
# #         pretrained=False,
# #         freeze_backbone=False,
# #     )

# #     missing, unexpected = model.load_state_dict(state_dict, strict=False)

# #     print("[INFO] missing keys   :", len(missing))
# #     print("[INFO] unexpected keys:", len(unexpected))
# #     if len(missing) > 0:
# #         print("[INFO] first missing:", missing[:10])
# #     if len(unexpected) > 0:
# #         print("[INFO] first unexpected:", unexpected[:10])

# #     if len(missing) > 0 or len(unexpected) > 0:
# #         print("\n[WARN] 当前模型源码与 checkpoint 结构不完全一致，结果可能不可靠。\n")

# #     model = model.to(device)
# #     model.eval()
# #     return model


# # # =========================================================
# # # 4. split.json 读取
# # # =========================================================
# # def load_split_ids(split_json_path: str, split_key: str = "unseen_cows") -> List[str]:
# #     with open(split_json_path, "r", encoding="utf-8") as f:
# #         data = json.load(f)
# #     if split_key not in data:
# #         raise KeyError(f"{split_json_path} 中不存在键: {split_key}")
# #     return [str(x) for x in data[split_key]]


# # # =========================================================
# # # 5. 文件读取
# # # =========================================================
# # def list_images_in_dir(dir_path: Path) -> List[str]:
# #     if not dir_path.exists():
# #         return []
# #     files = []
# #     for p in dir_path.iterdir():
# #         if p.is_file() and is_image_file(p):
# #             files.append(str(p))
# #     return sorted(files)


# # def list_ori_images_for_cow(ori_root: str, cow_id: str) -> List[str]:
# #     return list_images_in_dir(Path(ori_root) / cow_id)


# # def build_gt_dict_for_cow(gt_root: str, cow_id: str) -> Dict[str, int]:
# #     """
# #     normal = 1
# #     abnormal = 0
# #     按文件名建立字典
# #     """
# #     gt = {}
# #     cow_dir = Path(gt_root) / cow_id
# #     normal_dir = cow_dir / "normal"
# #     abnormal_dir = cow_dir / "abnormal"

# #     if normal_dir.exists():
# #         for p in normal_dir.iterdir():
# #             if p.is_file() and is_image_file(p):
# #                 gt[p.name] = 1

# #     if abnormal_dir.exists():
# #         for p in abnormal_dir.iterdir():
# #             if p.is_file() and is_image_file(p):
# #                 gt[p.name] = 0

# #     return gt


# # # =========================================================
# # # 6. Dataset
# # # =========================================================
# # class SimpleImageDataset(Dataset):
# #     def __init__(self, image_paths: List[str], transform=None):
# #         self.image_paths = image_paths
# #         self.transform = transform

# #     def __len__(self):
# #         return len(self.image_paths)

# #     def __getitem__(self, idx):
# #         path = self.image_paths[idx]
# #         img = Image.open(path).convert("RGB")
# #         if self.transform:
# #             img = self.transform(img)
# #         return img, path


# # # =========================================================
# # # 7. embedding 提取
# # # =========================================================
# # @torch.no_grad()
# # def extract_embeddings_for_paths(model, image_paths: List[str], transform, device: str, batch_size=64):
# #     dataset = SimpleImageDataset(image_paths=image_paths, transform=transform)
# #     loader = DataLoader(
# #         dataset,
# #         batch_size=batch_size,
# #         shuffle=False,
# #         num_workers=4,
# #         pin_memory=torch.cuda.is_available(),
# #     )

# #     all_emb = []
# #     all_paths = []

# #     for imgs, paths in loader:
# #         imgs = imgs.to(device, non_blocking=True)
# #         emb = model.forward_emb(imgs)
# #         emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-12)

# #         all_emb.append(emb.cpu().numpy())
# #         all_paths.extend(list(paths))

# #     if len(all_emb) == 0:
# #         return np.zeros((0, 1), dtype=np.float32), []

# #     X = np.concatenate(all_emb, axis=0).astype(np.float32)
# #     return X, all_paths


# # # =========================================================
# # # 8. 无监督分数：ori 内部 top-k 一致性
# # # =========================================================
# # def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
# #     X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
# #     return X @ X.T


# # def calc_internal_topk_scores(X: np.ndarray, topk: int = 3) -> np.ndarray:
# #     """
# #     对同一目录内部每张图：
# #     计算它与其它图的余弦相似度
# #     取 top-k 平均，作为内部一致性分数
# #     """
# #     n = X.shape[0]
# #     if n < 2:
# #         return np.zeros((n,), dtype=np.float32)

# #     sims = cosine_sim_matrix(X)
# #     scores = []

# #     for i in range(n):
# #         row = sims[i].copy()
# #         row[i] = -1.0
# #         k = min(topk, n - 1)
# #         topk_vals = np.partition(row, -k)[-k:]
# #         score = float(topk_vals.mean())
# #         scores.append(score)

# #     return np.asarray(scores, dtype=np.float32)


# # def build_threshold_mean_std(scores: np.ndarray, std_scale: float = 2.0) -> Dict[str, float]:
# #     mu = float(np.mean(scores))
# #     sigma = float(np.std(scores))
# #     thr = mu - std_scale * sigma
# #     return {
# #         "mean": mu,
# #         "std": sigma,
# #         "threshold": float(thr),
# #     }


# # # =========================================================
# # # 9. 监督评估：normal=1 为正类
# # # =========================================================
# # def calc_metrics_binary(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
# #     y_true = np.asarray(y_true).astype(int)
# #     y_pred = np.asarray(y_pred).astype(int)

# #     tp = int(((y_true == 1) & (y_pred == 1)).sum())
# #     tn = int(((y_true == 0) & (y_pred == 0)).sum())
# #     fp = int(((y_true == 0) & (y_pred == 1)).sum())
# #     fn = int(((y_true == 1) & (y_pred == 0)).sum())

# #     accuracy = (tp + tn) / max(len(y_true), 1)
# #     precision = tp / max(tp + fp, 1e-12)
# #     recall = tp / max(tp + fn, 1e-12)
# #     f1 = 2 * precision * recall / max(precision + recall, 1e-12)

# #     return {
# #         "accuracy": float(accuracy),
# #         "precision_normal": float(precision),
# #         "recall_normal": float(recall),
# #         "f1_normal": float(f1),
# #         "tp": tp,
# #         "tn": tn,
# #         "fp": fp,
# #         "fn": fn,
# #     }


# # # =========================================================
# # # 10. macro 平均
# # # =========================================================
# # def compute_macro_average(report: dict) -> Dict[str, float]:
# #     per_cow = report.get("per_cow_results", {})

# #     accs = []
# #     precs = []
# #     recalls = []
# #     f1s = []

# #     for _, info in per_cow.items():
# #         m = info.get("metrics", {})
# #         if len(m) == 0:
# #             continue

# #         accs.append(m.get("accuracy", 0.0))
# #         precs.append(m.get("precision_normal", 0.0))
# #         recalls.append(m.get("recall_normal", 0.0))
# #         f1s.append(m.get("f1_normal", 0.0))

# #     def safe_mean(x):
# #         return float(np.mean(x)) if len(x) > 0 else 0.0

# #     return {
# #         "macro_accuracy": safe_mean(accs),
# #         "macro_precision_normal": safe_mean(precs),
# #         "macro_recall_normal": safe_mean(recalls),
# #         "macro_f1_normal": safe_mean(f1s),
# #         "num_cows": len(accs),
# #     }


# # # =========================================================
# # # 11. 保存 CSV
# # # =========================================================
# # def save_per_cow_metrics_csv(report: dict, csv_path: str):
# #     rows = []
# #     per_cow_results = report.get("per_cow_results", {})

# #     for cow_id, info in per_cow_results.items():
# #         metrics = info.get("metrics", {})
# #         thr_info = info.get("threshold_info", {})

# #         row = {
# #             "cow_id": cow_id,
# #             "num_ori_images": info.get("num_ori_images", 0),
# #             "num_matched_gt": info.get("num_matched_gt", 0),
# #             "num_unmatched_gt": info.get("num_unmatched_gt", 0),
# #             "score_mean": thr_info.get("mean", None),
# #             "score_std": thr_info.get("std", None),
# #             "threshold": thr_info.get("threshold", None),
# #             "accuracy": metrics.get("accuracy", None),
# #             "precision_normal": metrics.get("precision_normal", None),
# #             "recall_normal": metrics.get("recall_normal", None),
# #             "f1_normal": metrics.get("f1_normal", None),
# #             "tp": metrics.get("tp", None),
# #             "tn": metrics.get("tn", None),
# #             "fp": metrics.get("fp", None),
# #             "fn": metrics.get("fn", None),
# #             "num_normal_to_abnormal": len(info.get("normal_to_abnormal", [])),
# #             "num_abnormal_to_normal": len(info.get("abnormal_to_normal", [])),
# #         }
# #         rows.append(row)

# #     fieldnames = [
# #         "cow_id",
# #         "num_ori_images",
# #         "num_matched_gt",
# #         "num_unmatched_gt",
# #         "score_mean",
# #         "score_std",
# #         "threshold",
# #         "accuracy",
# #         "precision_normal",
# #         "recall_normal",
# #         "f1_normal",
# #         "tp",
# #         "tn",
# #         "fp",
# #         "fn",
# #         "num_normal_to_abnormal",
# #         "num_abnormal_to_normal",
# #     ]

# #     with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
# #         writer = csv.DictWriter(f, fieldnames=fieldnames)
# #         writer.writeheader()
# #         writer.writerows(rows)

# #     print(f"[INFO] per-cow metrics csv saved to: {csv_path}")


# # def save_overall_metrics_csv(report: dict, csv_path: str):
# #     overall = report.get("overall_metrics", {})
# #     setting = report.get("setting", {})
# #     macro = report.get("macro_metrics", {})

# #     row = {
# #         "split_json_path": setting.get("split_json_path", ""),
# #         "ori_root": setting.get("ori_root", ""),
# #         "gt_root": setting.get("gt_root", ""),
# #         "model_path": setting.get("model_path", ""),
# #         "topk": setting.get("topk", ""),
# #         "threshold_rule": setting.get("threshold_rule", ""),
# #         "threshold_source": setting.get("threshold_source", ""),
# #         "evaluation_source": setting.get("evaluation_source", ""),

# #         "accuracy_micro": overall.get("accuracy", None),
# #         "precision_micro": overall.get("precision_normal", None),
# #         "recall_micro": overall.get("recall_normal", None),
# #         "f1_micro": overall.get("f1_normal", None),

# #         "accuracy_macro": macro.get("macro_accuracy", None),
# #         "precision_macro": macro.get("macro_precision_normal", None),
# #         "recall_macro": macro.get("macro_recall_normal", None),
# #         "f1_macro": macro.get("macro_f1_normal", None),

# #         "num_cows": macro.get("num_cows", 0),

# #         "tp": overall.get("tp", None),
# #         "tn": overall.get("tn", None),
# #         "fp": overall.get("fp", None),
# #         "fn": overall.get("fn", None),
# #     }

# #     fieldnames = [
# #         "split_json_path",
# #         "ori_root",
# #         "gt_root",
# #         "model_path",
# #         "topk",
# #         "threshold_rule",
# #         "threshold_source",
# #         "evaluation_source",

# #         "accuracy_micro",
# #         "precision_micro",
# #         "recall_micro",
# #         "f1_micro",

# #         "accuracy_macro",
# #         "precision_macro",
# #         "recall_macro",
# #         "f1_macro",

# #         "num_cows",

# #         "tp",
# #         "tn",
# #         "fp",
# #         "fn",
# #     ]

# #     with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
# #         writer = csv.DictWriter(f, fieldnames=fieldnames)
# #         writer.writeheader()
# #         writer.writerow(row)

# #     print(f"[INFO] overall metrics csv saved to: {csv_path}")


# # # =========================================================
# # # 12. 保存错误图片
# # # =========================================================
# # def copy_error_images(report: dict, error_root: str):
# #     """
# #     输出目录结构：
# #     error_root/
# #         cow_id/
# #             normal_to_abnormal/
# #                 xxx.jpg
# #             abnormal_to_normal/
# #                 yyy.jpg
# #     """
# #     error_root = Path(error_root)
# #     error_root.mkdir(parents=True, exist_ok=True)

# #     per_cow_results = report.get("per_cow_results", {})

# #     total_n2a = 0
# #     total_a2n = 0

# #     for cow_id, info in per_cow_results.items():
# #         cow_dir = error_root / str(cow_id)
# #         n2a_dir = cow_dir / "normal_to_abnormal"
# #         a2n_dir = cow_dir / "abnormal_to_normal"

# #         n2a_dir.mkdir(parents=True, exist_ok=True)
# #         a2n_dir.mkdir(parents=True, exist_ok=True)

# #         normal_to_abnormal = info.get("normal_to_abnormal", [])
# #         abnormal_to_normal = info.get("abnormal_to_normal", [])

# #         for row in normal_to_abnormal:
# #             src = Path(row["path"])
# #             if src.exists():
# #                 dst = n2a_dir / src.name
# #                 if dst.exists():
# #                     dst = n2a_dir / f"{src.stem}__dup{src.suffix}"
# #                 shutil.copy2(src, dst)
# #                 total_n2a += 1

# #         for row in abnormal_to_normal:
# #             src = Path(row["path"])
# #             if src.exists():
# #                 dst = a2n_dir / src.name
# #                 if dst.exists():
# #                     dst = a2n_dir / f"{src.stem}__dup{src.suffix}"
# #                 shutil.copy2(src, dst)
# #                 total_a2n += 1

# #     print(f"[INFO] error images copied to: {error_root}")
# #     print(f"[INFO] total normal_to_abnormal copied: {total_n2a}")
# #     print(f"[INFO] total abnormal_to_normal copied: {total_a2n}")


# # # =========================================================
# # # 13. 主流程：阈值只用 ori，GT 只用于评估
# # # =========================================================
# # def evaluate_unseen_ori_with_gt_check(
# #     split_json_path: str,
# #     ori_root: str,
# #     gt_root: str,
# #     model_path: str,
# #     output_json: str,
# #     per_cow_csv: str,
# #     overall_csv: str,
# #     error_root: str,
# #     batch_size: int = 64,
# #     seed: int = 42,
# #     topk: int = 3,
# #     std_scale: float = 2.0,
# # ):
# #     set_seed(seed)

# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     transform = build_eval_transforms(img_size=224)

# #     unseen_cows = load_split_ids(split_json_path, split_key="unseen_cows")
# #     print(f"[INFO] unseen cows = {len(unseen_cows)}")

# #     model = build_model_from_ckpt(model_path, device=device)

# #     report = {
# #         "setting": {
# #             "split_json_path": split_json_path,
# #             "ori_root": ori_root,
# #             "gt_root": gt_root,
# #             "model_path": model_path,
# #             "topk": topk,
# #             "threshold_rule": f"mean - {std_scale} * std",
# #             "threshold_source": "ori_only",
# #             "evaluation_source": "gt_only_after_prediction",
# #             "positive_class": "normal=1",
# #             "negative_class": "abnormal=0",
# #             "error_root": error_root,
# #         },
# #         "per_cow_results": {}
# #     }

# #     all_y_true = []
# #     all_y_pred = []

# #     for cow_id in unseen_cows:
# #         print("\n" + "=" * 80)
# #         print(f"[INFO] Processing cow_id = {cow_id}")

# #         ori_paths = list_ori_images_for_cow(ori_root, cow_id)
# #         gt_dict = build_gt_dict_for_cow(gt_root, cow_id)

# #         if len(ori_paths) < 2:
# #             print(f"[WARN] ori 图片少于 2 张，跳过: {cow_id}")
# #             continue

# #         if len(gt_dict) == 0:
# #             print(f"[WARN] GT 字典为空，跳过: {cow_id}")
# #             continue

# #         X, emb_paths = extract_embeddings_for_paths(
# #             model=model,
# #             image_paths=ori_paths,
# #             transform=transform,
# #             device=device,
# #             batch_size=batch_size,
# #         )

# #         if X.shape[0] < 2:
# #             print(f"[WARN] embedding 数量少于 2，跳过: {cow_id}")
# #             continue

# #         # 1) 只用 ori 自己内部算 score 和 threshold
# #         scores = calc_internal_topk_scores(X, topk=topk)
# #         thr_info = build_threshold_mean_std(scores, std_scale=std_scale)
# #         threshold = thr_info["threshold"]

# #         # 2) 先完成无监督预测
# #         details = []
# #         unmatched = []
# #         y_true = []
# #         y_pred = []

# #         normal_to_abnormal = []
# #         abnormal_to_normal = []

# #         for path, score in zip(emb_paths, scores):
# #             filename = Path(path).name
# #             pred = 1 if float(score) >= threshold else 0

# #             row = {
# #                 "path": path,
# #                 "filename": filename,
# #                 "score": float(score),
# #                 "pred": int(pred),
# #             }

# #             # 3) 再去对 GT，仅用于评估
# #             if filename not in gt_dict:
# #                 unmatched.append({
# #                     "path": path,
# #                     "filename": filename,
# #                     "score": float(score),
# #                     "pred": int(pred),
# #                     "reason": "filename_not_found_in_gt_dict",
# #                 })
# #                 continue

# #             gt = int(gt_dict[filename])
# #             row["gt"] = gt
# #             row["correct"] = bool(gt == pred)

# #             y_true.append(gt)
# #             y_pred.append(pred)
# #             details.append(row)

# #             if gt == 1 and pred == 0:
# #                 normal_to_abnormal.append(row)
# #             elif gt == 0 and pred == 1:
# #                 abnormal_to_normal.append(row)

# #         if len(y_true) == 0:
# #             print(f"[WARN] 没有可对齐 GT 的样本，跳过: {cow_id}")
# #             continue

# #         metrics = calc_metrics_binary(y_true, y_pred)

# #         num_keep = sum(int(x == 1) for x in y_pred)
# #         num_reject = sum(int(x == 0) for x in y_pred)

# #         print(f"[INFO] num_ori_images        = {len(ori_paths)}")
# #         print(f"[INFO] num_matched_gt       = {len(y_true)}")
# #         print(f"[INFO] num_unmatched_gt     = {len(unmatched)}")
# #         print(f"[INFO] score_mean/std       = {thr_info['mean']:.6f} / {thr_info['std']:.6f}")
# #         print(f"[INFO] threshold            = {threshold:.6f}")
# #         print(f"[INFO] keep/reject          = {num_keep}/{num_reject}")
# #         print(f"[INFO] metrics              = {metrics}")
# #         print(f"[INFO] normal_to_abnormal   = {len(normal_to_abnormal)}")
# #         print(f"[INFO] abnormal_to_normal   = {len(abnormal_to_normal)}")

# #         report["per_cow_results"][cow_id] = {
# #             "num_ori_images": int(len(ori_paths)),
# #             "num_matched_gt": int(len(y_true)),
# #             "num_unmatched_gt": int(len(unmatched)),
# #             "threshold_info": thr_info,
# #             "metrics": metrics,
# #             "normal_to_abnormal": normal_to_abnormal,
# #             "abnormal_to_normal": abnormal_to_normal,
# #             "unmatched": unmatched,
# #             "details": details,
# #         }

# #         all_y_true.extend(y_true)
# #         all_y_pred.extend(y_pred)

# #     if len(all_y_true) == 0:
# #         raise ValueError("没有任何有效评测样本，请检查 ori / GT 文件名对应关系。")

# #     overall_metrics = calc_metrics_binary(all_y_true, all_y_pred)
# #     report["overall_metrics"] = overall_metrics

# #     macro_metrics = compute_macro_average(report)
# #     report["macro_metrics"] = macro_metrics

# #     with open(output_json, "w", encoding="utf-8") as f:
# #         json.dump(report, f, ensure_ascii=False, indent=2)

# #     save_per_cow_metrics_csv(report, per_cow_csv)
# #     save_overall_metrics_csv(report, overall_csv)
# #     copy_error_images(report, error_root)

# #     print("\n" + "#" * 80)
# #     print("[INFO] OVERALL METRICS (MICRO)")
# #     print(overall_metrics)
# #     print("[INFO] MACRO METRICS")
# #     print(macro_metrics)
# #     print(f"[INFO] report saved to: {output_json}")
# #     print("#" * 80)

# #     return report


# # # =========================================================
# # # 14. main
# # # =========================================================
# # def main():
# #     split_json_path = r"D:\code_project\triplet_code\data\dataset_OBB\split.json"
# #     ori_root = r"D:\code_project\triplet_code\data\dataset_obb_ori"
# #     gt_root = r"D:\code_project\triplet_code\data\dataset_OBB"
# #     model_path = r"D:\code_project\triplet_code\logs\exp001_3.31_arcface_CHA_GAP\checkpoints\best.pth"

# #     output_json = r"D:\code_project\triplet_code\logs\unseen_ori_eval_with_gtcheck.json"
# #     per_cow_csv = r"D:\code_project\triplet_code\logs\unseen_ori_eval_per_cow_metrics.csv"
# #     overall_csv = r"D:\code_project\triplet_code\logs\unseen_ori_eval_overall_metrics.csv"

# #     # 错误图片输出根目录
# #     error_root = r"D:\code_project\triplet_code\logs\unseen_ori_error_images"

# #     evaluate_unseen_ori_with_gt_check(
# #         split_json_path=split_json_path,
# #         ori_root=ori_root,
# #         gt_root=gt_root,
# #         model_path=model_path,
# #         output_json=output_json,
# #         per_cow_csv=per_cow_csv,
# #         overall_csv=overall_csv,
# #         error_root=error_root,
# #         batch_size=64,
# #         seed=42,
# #         topk=3,
# #         std_scale=2.0,
# #     )


# # if __name__ == "__main__":
# #     main()

# import os
# import sys
# import json
# import csv
# import random
# import shutil
# from pathlib import Path
# from typing import List, Dict

# import numpy as np
# from PIL import Image

# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms


# # =========================================================
# # 0. 基础配置
# # =========================================================
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# def is_image_file(path: Path) -> bool:
#     return path.suffix.lower() in IMG_EXTS


# # =========================================================
# # 1. 路径设置
# # =========================================================
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.append(str(PROJECT_ROOT))

# from src.sup.models.arcface_model import ArcFaceCowModel


# # =========================================================
# # 2. 图像变换
# # =========================================================
# def build_eval_transforms(img_size=224):
#     return transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225],
#         ),
#     ])


# # =========================================================
# # 3. checkpoint 加载
# # =========================================================
# def load_checkpoint(model_path: str):
#     ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
#     if isinstance(ckpt, dict) and "model" in ckpt:
#         state_dict = ckpt["model"]
#     elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
#         state_dict = ckpt["model_state_dict"]
#     elif isinstance(ckpt, dict) and "state_dict" in ckpt:
#         state_dict = ckpt["state_dict"]
#     else:
#         state_dict = ckpt
#     return ckpt, state_dict


# def infer_arcface_config(state_dict: dict):
#     if "arcface.weight" not in state_dict:
#         raise KeyError("checkpoint 中没有 arcface.weight，无法自动推断 num_classes / embedding_dim")
#     arc_w = state_dict["arcface.weight"]
#     num_classes = arc_w.shape[0]
#     embedding_dim = arc_w.shape[1]
#     return num_classes, embedding_dim


# def build_model_from_ckpt(model_path: str, device: str):
#     _, state_dict = load_checkpoint(model_path)
#     num_classes, embedding_dim = infer_arcface_config(state_dict)

#     print("=" * 80)
#     print(f"[INFO] model_path  : {model_path}")
#     print(f"[INFO] num_classes : {num_classes}")
#     print(f"[INFO] emb_dim     : {embedding_dim}")
#     print("=" * 80)

#     model = ArcFaceCowModel(
#         num_classes=num_classes,
#         embedding_dim=embedding_dim,
#         pretrained=False,
#         freeze_backbone=False,
#     )

#     missing, unexpected = model.load_state_dict(state_dict, strict=False)

#     print("[INFO] missing keys   :", len(missing))
#     print("[INFO] unexpected keys:", len(unexpected))
#     if len(missing) > 0:
#         print("[INFO] first missing:", missing[:10])
#     if len(unexpected) > 0:
#         print("[INFO] first unexpected:", unexpected[:10])

#     if len(missing) > 0 or len(unexpected) > 0:
#         print("\n[WARN] 当前模型源码与 checkpoint 结构不完全一致，结果可能不可靠。\n")

#     model = model.to(device)
#     model.eval()
#     return model


# # =========================================================
# # 4. split.json 读取
# # =========================================================
# def load_split_ids(split_json_path: str, split_key: str = "unseen_cows") -> List[str]:
#     with open(split_json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     if split_key not in data:
#         raise KeyError(f"{split_json_path} 中不存在键: {split_key}")
#     return [str(x) for x in data[split_key]]


# # =========================================================
# # 5. 文件读取
# # =========================================================
# def list_images_in_dir(dir_path: Path) -> List[str]:
#     if not dir_path.exists():
#         return []
#     files = []
#     for p in dir_path.iterdir():
#         if p.is_file() and is_image_file(p):
#             files.append(str(p))
#     return sorted(files)


# def list_ori_images_for_cow(ori_root: str, cow_id: str) -> List[str]:
#     return list_images_in_dir(Path(ori_root) / cow_id)


# def build_gt_dict_for_cow(gt_root: str, cow_id: str) -> Dict[str, int]:
#     """
#     normal = 1
#     abnormal = 0
#     按文件名建立字典
#     """
#     gt = {}
#     cow_dir = Path(gt_root) / cow_id
#     normal_dir = cow_dir / "normal"
#     abnormal_dir = cow_dir / "abnormal"

#     if normal_dir.exists():
#         for p in normal_dir.iterdir():
#             if p.is_file() and is_image_file(p):
#                 gt[p.name] = 1

#     if abnormal_dir.exists():
#         for p in abnormal_dir.iterdir():
#             if p.is_file() and is_image_file(p):
#                 gt[p.name] = 0

#     return gt


# # =========================================================
# # 6. Dataset
# # =========================================================
# class SimpleImageDataset(Dataset):
#     def __init__(self, image_paths: List[str], transform=None):
#         self.image_paths = image_paths
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         path = self.image_paths[idx]
#         img = Image.open(path).convert("RGB")
#         if self.transform:
#             img = self.transform(img)
#         return img, path


# # =========================================================
# # 7. embedding 提取
# # =========================================================
# @torch.no_grad()
# def extract_embeddings_for_paths(model, image_paths: List[str], transform, device: str, batch_size=64):
#     dataset = SimpleImageDataset(image_paths=image_paths, transform=transform)
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=torch.cuda.is_available(),
#     )

#     all_emb = []
#     all_paths = []

#     for imgs, paths in loader:
#         imgs = imgs.to(device, non_blocking=True)
#         emb = model.forward_emb(imgs)
#         emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-12)

#         all_emb.append(emb.cpu().numpy())
#         all_paths.extend(list(paths))

#     if len(all_emb) == 0:
#         return np.zeros((0, 1), dtype=np.float32), []

#     X = np.concatenate(all_emb, axis=0).astype(np.float32)
#     return X, all_paths


# # =========================================================
# # 8. 无监督分数：改为到中心的欧氏距离
# # =========================================================
# def calc_center_euclidean_scores(X: np.ndarray) -> np.ndarray:
#     """
#     对同一目录内部每张图：
#     先求所有 embedding 的中心（均值）
#     再计算每张图到中心的欧氏距离

#     返回:
#         scores: [N]
#         数值越大，越异常
#     """
#     n = X.shape[0]
#     if n < 2:
#         return np.zeros((n,), dtype=np.float32)

#     center = np.mean(X, axis=0, keepdims=True)      # [1, D]
#     scores = np.linalg.norm(X - center, axis=1)     # [N]

#     return scores.astype(np.float32)


# def build_threshold_mean_std(scores: np.ndarray, std_scale: float = 2.0) -> Dict[str, float]:
#     mu = float(np.mean(scores))
#     sigma = float(np.std(scores))
#     thr = mu + std_scale * sigma
#     return {
#         "mean": mu,
#         "std": sigma,
#         "threshold": float(thr),
#     }


# # =========================================================
# # 9. 监督评估：normal=1 为正类
# # =========================================================
# def calc_metrics_binary(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
#     y_true = np.asarray(y_true).astype(int)
#     y_pred = np.asarray(y_pred).astype(int)

#     tp = int(((y_true == 1) & (y_pred == 1)).sum())
#     tn = int(((y_true == 0) & (y_pred == 0)).sum())
#     fp = int(((y_true == 0) & (y_pred == 1)).sum())
#     fn = int(((y_true == 1) & (y_pred == 0)).sum())

#     accuracy = (tp + tn) / max(len(y_true), 1)
#     precision = tp / max(tp + fp, 1e-12)
#     recall = tp / max(tp + fn, 1e-12)
#     f1 = 2 * precision * recall / max(precision + recall, 1e-12)

#     return {
#         "accuracy": float(accuracy),
#         "precision_normal": float(precision),
#         "recall_normal": float(recall),
#         "f1_normal": float(f1),
#         "tp": tp,
#         "tn": tn,
#         "fp": fp,
#         "fn": fn,
#     }


# # =========================================================
# # 10. macro 平均
# # =========================================================
# def compute_macro_average(report: dict) -> Dict[str, float]:
#     per_cow = report.get("per_cow_results", {})

#     accs = []
#     precs = []
#     recalls = []
#     f1s = []

#     for _, info in per_cow.items():
#         m = info.get("metrics", {})
#         if len(m) == 0:
#             continue

#         accs.append(m.get("accuracy", 0.0))
#         precs.append(m.get("precision_normal", 0.0))
#         recalls.append(m.get("recall_normal", 0.0))
#         f1s.append(m.get("f1_normal", 0.0))

#     def safe_mean(x):
#         return float(np.mean(x)) if len(x) > 0 else 0.0

#     return {
#         "macro_accuracy": safe_mean(accs),
#         "macro_precision_normal": safe_mean(precs),
#         "macro_recall_normal": safe_mean(recalls),
#         "macro_f1_normal": safe_mean(f1s),
#         "num_cows": len(accs),
#     }


# # =========================================================
# # 11. 保存 CSV
# # =========================================================
# def save_per_cow_metrics_csv(report: dict, csv_path: str):
#     rows = []
#     per_cow_results = report.get("per_cow_results", {})

#     for cow_id, info in per_cow_results.items():
#         metrics = info.get("metrics", {})
#         thr_info = info.get("threshold_info", {})

#         row = {
#             "cow_id": cow_id,
#             "num_ori_images": info.get("num_ori_images", 0),
#             "num_matched_gt": info.get("num_matched_gt", 0),
#             "num_unmatched_gt": info.get("num_unmatched_gt", 0),
#             "score_mean": thr_info.get("mean", None),
#             "score_std": thr_info.get("std", None),
#             "threshold": thr_info.get("threshold", None),
#             "accuracy": metrics.get("accuracy", None),
#             "precision_normal": metrics.get("precision_normal", None),
#             "recall_normal": metrics.get("recall_normal", None),
#             "f1_normal": metrics.get("f1_normal", None),
#             "tp": metrics.get("tp", None),
#             "tn": metrics.get("tn", None),
#             "fp": metrics.get("fp", None),
#             "fn": metrics.get("fn", None),
#             "num_normal_to_abnormal": len(info.get("normal_to_abnormal", [])),
#             "num_abnormal_to_normal": len(info.get("abnormal_to_normal", [])),
#         }
#         rows.append(row)

#     fieldnames = [
#         "cow_id",
#         "num_ori_images",
#         "num_matched_gt",
#         "num_unmatched_gt",
#         "score_mean",
#         "score_std",
#         "threshold",
#         "accuracy",
#         "precision_normal",
#         "recall_normal",
#         "f1_normal",
#         "tp",
#         "tn",
#         "fp",
#         "fn",
#         "num_normal_to_abnormal",
#         "num_abnormal_to_normal",
#     ]

#     with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(rows)

#     print(f"[INFO] per-cow metrics csv saved to: {csv_path}")


# def save_overall_metrics_csv(report: dict, csv_path: str):
#     overall = report.get("overall_metrics", {})
#     setting = report.get("setting", {})
#     macro = report.get("macro_metrics", {})

#     row = {
#         "split_json_path": setting.get("split_json_path", ""),
#         "ori_root": setting.get("ori_root", ""),
#         "gt_root": setting.get("gt_root", ""),
#         "model_path": setting.get("model_path", ""),
#         "topk": setting.get("topk", ""),
#         "threshold_rule": setting.get("threshold_rule", ""),
#         "threshold_source": setting.get("threshold_source", ""),
#         "evaluation_source": setting.get("evaluation_source", ""),

#         "accuracy_micro": overall.get("accuracy", None),
#         "precision_micro": overall.get("precision_normal", None),
#         "recall_micro": overall.get("recall_normal", None),
#         "f1_micro": overall.get("f1_normal", None),

#         "accuracy_macro": macro.get("macro_accuracy", None),
#         "precision_macro": macro.get("macro_precision_normal", None),
#         "recall_macro": macro.get("macro_recall_normal", None),
#         "f1_macro": macro.get("macro_f1_normal", None),

#         "num_cows": macro.get("num_cows", 0),

#         "tp": overall.get("tp", None),
#         "tn": overall.get("tn", None),
#         "fp": overall.get("fp", None),
#         "fn": overall.get("fn", None),
#     }

#     fieldnames = [
#         "split_json_path",
#         "ori_root",
#         "gt_root",
#         "model_path",
#         "topk",
#         "threshold_rule",
#         "threshold_source",
#         "evaluation_source",

#         "accuracy_micro",
#         "precision_micro",
#         "recall_micro",
#         "f1_micro",

#         "accuracy_macro",
#         "precision_macro",
#         "recall_macro",
#         "f1_macro",

#         "num_cows",

#         "tp",
#         "tn",
#         "fp",
#         "fn",
#     ]

#     with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerow(row)

#     print(f"[INFO] overall metrics csv saved to: {csv_path}")


# # =========================================================
# # 12. 保存错误图片
# # =========================================================
# def copy_error_images(report: dict, error_root: str):
#     """
#     输出目录结构：
#     error_root/
#         cow_id/
#             normal_to_abnormal/
#                 xxx.jpg
#             abnormal_to_normal/
#                 yyy.jpg
#     """
#     error_root = Path(error_root)
#     error_root.mkdir(parents=True, exist_ok=True)

#     per_cow_results = report.get("per_cow_results", {})

#     total_n2a = 0
#     total_a2n = 0

#     for cow_id, info in per_cow_results.items():
#         cow_dir = error_root / str(cow_id)
#         n2a_dir = cow_dir / "normal_to_abnormal"
#         a2n_dir = cow_dir / "abnormal_to_normal"

#         n2a_dir.mkdir(parents=True, exist_ok=True)
#         a2n_dir.mkdir(parents=True, exist_ok=True)

#         normal_to_abnormal = info.get("normal_to_abnormal", [])
#         abnormal_to_normal = info.get("abnormal_to_normal", [])

#         for row in normal_to_abnormal:
#             src = Path(row["path"])
#             if src.exists():
#                 dst = n2a_dir / src.name
#                 if dst.exists():
#                     dst = n2a_dir / f"{src.stem}__dup{src.suffix}"
#                 shutil.copy2(src, dst)
#                 total_n2a += 1

#         for row in abnormal_to_normal:
#             src = Path(row["path"])
#             if src.exists():
#                 dst = a2n_dir / src.name
#                 if dst.exists():
#                     dst = a2n_dir / f"{src.stem}__dup{src.suffix}"
#                 shutil.copy2(src, dst)
#                 total_a2n += 1

#     print(f"[INFO] error images copied to: {error_root}")
#     print(f"[INFO] total normal_to_abnormal copied: {total_n2a}")
#     print(f"[INFO] total abnormal_to_normal copied: {total_a2n}")


# # =========================================================
# # 13. 主流程：阈值只用 ori，GT 只用于评估
# # =========================================================
# def evaluate_unseen_ori_with_gt_check(
#     split_json_path: str,
#     ori_root: str,
#     gt_root: str,
#     model_path: str,
#     output_json: str,
#     per_cow_csv: str,
#     overall_csv: str,
#     error_root: str,
#     batch_size: int = 64,
#     seed: int = 42,
#     topk: int = 3,
#     std_scale: float = 2.0,
# ):
#     set_seed(seed)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     transform = build_eval_transforms(img_size=224)

#     unseen_cows = load_split_ids(split_json_path, split_key="unseen_cows")
#     print(f"[INFO] unseen cows = {len(unseen_cows)}")

#     model = build_model_from_ckpt(model_path, device=device)

#     report = {
#         "setting": {
#             "split_json_path": split_json_path,
#             "ori_root": ori_root,
#             "gt_root": gt_root,
#             "model_path": model_path,
#             "topk": topk,
#             "threshold_rule": f"mean + {std_scale} * std",
#             "threshold_source": "ori_only",
#             "evaluation_source": "gt_only_after_prediction",
#             "positive_class": "normal=1",
#             "negative_class": "abnormal=0",
#             "error_root": error_root,
#         },
#         "per_cow_results": {}
#     }

#     all_y_true = []
#     all_y_pred = []

#     for cow_id in unseen_cows:
#         print("\n" + "=" * 80)
#         print(f"[INFO] Processing cow_id = {cow_id}")

#         ori_paths = list_ori_images_for_cow(ori_root, cow_id)
#         gt_dict = build_gt_dict_for_cow(gt_root, cow_id)

#         if len(ori_paths) < 2:
#             print(f"[WARN] ori 图片少于 2 张，跳过: {cow_id}")
#             continue

#         if len(gt_dict) == 0:
#             print(f"[WARN] GT 字典为空，跳过: {cow_id}")
#             continue

#         X, emb_paths = extract_embeddings_for_paths(
#             model=model,
#             image_paths=ori_paths,
#             transform=transform,
#             device=device,
#             batch_size=batch_size,
#         )

#         if X.shape[0] < 2:
#             print(f"[WARN] embedding 数量少于 2，跳过: {cow_id}")
#             continue

#         # 1) 只用 ori 自己内部算 score 和 threshold
#         scores = calc_center_euclidean_scores(X)
#         thr_info = build_threshold_mean_std(scores, std_scale=std_scale)
#         threshold = thr_info["threshold"]

#         # 2) 先完成无监督预测
#         details = []
#         unmatched = []
#         y_true = []
#         y_pred = []

#         normal_to_abnormal = []
#         abnormal_to_normal = []

#         for path, score in zip(emb_paths, scores):
#             filename = Path(path).name
#             pred = 1 if float(score) <= threshold else 0

#             row = {
#                 "path": path,
#                 "filename": filename,
#                 "score": float(score),
#                 "pred": int(pred),
#             }

#             # 3) 再去对 GT，仅用于评估
#             if filename not in gt_dict:
#                 unmatched.append({
#                     "path": path,
#                     "filename": filename,
#                     "score": float(score),
#                     "pred": int(pred),
#                     "reason": "filename_not_found_in_gt_dict",
#                 })
#                 continue

#             gt = int(gt_dict[filename])
#             row["gt"] = gt
#             row["correct"] = bool(gt == pred)

#             y_true.append(gt)
#             y_pred.append(pred)
#             details.append(row)

#             if gt == 1 and pred == 0:
#                 normal_to_abnormal.append(row)
#             elif gt == 0 and pred == 1:
#                 abnormal_to_normal.append(row)

#         if len(y_true) == 0:
#             print(f"[WARN] 没有可对齐 GT 的样本，跳过: {cow_id}")
#             continue

#         metrics = calc_metrics_binary(y_true, y_pred)

#         num_keep = sum(int(x == 1) for x in y_pred)
#         num_reject = sum(int(x == 0) for x in y_pred)

#         print(f"[INFO] num_ori_images        = {len(ori_paths)}")
#         print(f"[INFO] num_matched_gt       = {len(y_true)}")
#         print(f"[INFO] num_unmatched_gt     = {len(unmatched)}")
#         print(f"[INFO] score_mean/std       = {thr_info['mean']:.6f} / {thr_info['std']:.6f}")
#         print(f"[INFO] threshold            = {threshold:.6f}")
#         print(f"[INFO] keep/reject          = {num_keep}/{num_reject}")
#         print(f"[INFO] metrics              = {metrics}")
#         print(f"[INFO] normal_to_abnormal   = {len(normal_to_abnormal)}")
#         print(f"[INFO] abnormal_to_normal   = {len(abnormal_to_normal)}")

#         report["per_cow_results"][cow_id] = {
#             "num_ori_images": int(len(ori_paths)),
#             "num_matched_gt": int(len(y_true)),
#             "num_unmatched_gt": int(len(unmatched)),
#             "threshold_info": thr_info,
#             "metrics": metrics,
#             "normal_to_abnormal": normal_to_abnormal,
#             "abnormal_to_normal": abnormal_to_normal,
#             "unmatched": unmatched,
#             "details": details,
#         }

#         all_y_true.extend(y_true)
#         all_y_pred.extend(y_pred)

#     if len(all_y_true) == 0:
#         raise ValueError("没有任何有效评测样本，请检查 ori / GT 文件名对应关系。")

#     overall_metrics = calc_metrics_binary(all_y_true, all_y_pred)
#     report["overall_metrics"] = overall_metrics

#     macro_metrics = compute_macro_average(report)
#     report["macro_metrics"] = macro_metrics

#     with open(output_json, "w", encoding="utf-8") as f:
#         json.dump(report, f, ensure_ascii=False, indent=2)

#     save_per_cow_metrics_csv(report, per_cow_csv)
#     save_overall_metrics_csv(report, overall_csv)
#     copy_error_images(report, error_root)

#     print("\n" + "#" * 80)
#     print("[INFO] OVERALL METRICS (MICRO)")
#     print(overall_metrics)
#     print("[INFO] MACRO METRICS")
#     print(macro_metrics)
#     print(f"[INFO] report saved to: {output_json}")
#     print("#" * 80)

#     return report


# # =========================================================
# # 14. main
# # =========================================================
# def main():
#     split_json_path = r"D:\code_project\triplet_code\data\dataset_OBB\split.json"
#     ori_root = r"D:\code_project\triplet_code\data\dataset_obb_ori"
#     gt_root = r"D:\code_project\triplet_code\data\dataset_OBB"
#     model_path = r"D:\code_project\triplet_code\logs\exp001_3.31_arcface_CHA_GAP\checkpoints\best.pth"

#     output_json = r"D:\code_project\triplet_code\unseen_ori_eval_with_gtcheck.json"
#     per_cow_csv = r"D:\code_project\triplet_code\unseen_ori_eval_per_cow_metrics.csv"
#     overall_csv = r"D:\code_project\triplet_code\unseen_ori_eval_overall_metrics.csv"

#     # 错误图片输出根目录
#     error_root = r"D:\code_project\triplet_code\unseen_ori_error_images"

#     evaluate_unseen_ori_with_gt_check(
#         split_json_path=split_json_path,
#         ori_root=ori_root,
#         gt_root=gt_root,
#         model_path=model_path,
#         output_json=output_json,
#         per_cow_csv=per_cow_csv,
#         overall_csv=overall_csv,
#         error_root=error_root,
#         batch_size=64,
#         seed=42,
#         topk=3,
#         std_scale=2.0,
#     )


# if __name__ == "__main__":
#     main()
import os
import sys
import csv
import random
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# =========================================================
# 0. 基础配置
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


# =========================================================
# 1. 路径设置
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.sup.models.arcface_model import ArcFaceCowModel


# =========================================================
# 2. 图像变换
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
# 3. checkpoint 加载
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
        raise KeyError("checkpoint 中没有 arcface.weight")

    arc_w = state_dict["arcface.weight"]

    num_classes = arc_w.shape[0]
    embedding_dim = arc_w.shape[1]

    return num_classes, embedding_dim


def build_model_from_ckpt(model_path: str, device: str):
    _, state_dict = load_checkpoint(model_path)

    num_classes, embedding_dim = infer_arcface_config(state_dict)

    print("=" * 80)
    print(f"[INFO] model_path  : {model_path}")
    print(f"[INFO] num_classes : {num_classes}")
    print(f"[INFO] emb_dim     : {embedding_dim}")
    print("=" * 80)

    model = ArcFaceCowModel(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained=False,
        freeze_backbone=False,
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    print("[INFO] missing keys   :", len(missing))
    print("[INFO] unexpected keys:", len(unexpected))

    if len(missing) > 0:
        print("[INFO] first missing:", missing[:10])

    if len(unexpected) > 0:
        print("[INFO] first unexpected:", unexpected[:10])

    model = model.to(device)
    model.eval()

    return model


# =========================================================
# 4. split.json 读取
# =========================================================
def load_split_ids(
    split_json_path: str,
    split_key: str = "unseen_cows"
) -> List[str]:

    import json

    with open(split_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if split_key not in data:
        raise KeyError(f"{split_json_path} 中不存在键: {split_key}")

    return [str(x) for x in data[split_key]]


# =========================================================
# 5. 文件读取
# =========================================================
def list_images_in_dir(dir_path: Path) -> List[str]:
    if not dir_path.exists():
        return []

    files = []

    for p in dir_path.iterdir():
        if p.is_file() and is_image_file(p):
            files.append(str(p))

    return sorted(files)


def list_ori_images_for_cow(
    ori_root: str,
    cow_id: str
) -> List[str]:

    return list_images_in_dir(Path(ori_root) / cow_id)


def build_gt_dict_for_cow(
    gt_root: str,
    cow_id: str
) -> Dict[str, int]:

    gt = {}

    cow_dir = Path(gt_root) / cow_id

    normal_dir = cow_dir / "normal"
    abnormal_dir = cow_dir / "abnormal"

    if normal_dir.exists():
        for p in normal_dir.iterdir():
            if p.is_file() and is_image_file(p):
                gt[p.name] = 1

    if abnormal_dir.exists():
        for p in abnormal_dir.iterdir():
            if p.is_file() and is_image_file(p):
                gt[p.name] = 0

    return gt


# =========================================================
# 6. Dataset
# =========================================================
class SimpleImageDataset(Dataset):

    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        path = self.image_paths[idx]

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, path


# =========================================================
# 7. embedding 提取
# =========================================================
@torch.no_grad()
def extract_embeddings_for_paths(
    model,
    image_paths: List[str],
    transform,
    device: str,
    batch_size=64
):

    dataset = SimpleImageDataset(
        image_paths=image_paths,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    all_emb = []
    all_paths = []

    for imgs, paths in loader:

        imgs = imgs.to(device, non_blocking=True)

        emb = model.forward_emb(imgs)

        emb = emb / (
            emb.norm(dim=1, keepdim=True) + 1e-12
        )

        all_emb.append(emb.cpu().numpy())

        all_paths.extend(list(paths))

    if len(all_emb) == 0:
        return np.zeros((0, 1), dtype=np.float32), []

    X = np.concatenate(all_emb, axis=0).astype(np.float32)

    return X, all_paths


# =========================================================
# 8. 欧氏距离 score
# =========================================================
def calc_center_euclidean_scores(X: np.ndarray):

    n = X.shape[0]

    if n < 2:
        return np.zeros((n,), dtype=np.float32)

    center = np.mean(X, axis=0, keepdims=True)

    scores = np.linalg.norm(
        X - center,
        axis=1
    )

    return scores.astype(np.float32)


def build_threshold_mean_std(
    scores: np.ndarray,
    std_scale: float = 1
):

    mu = float(np.mean(scores))

    sigma = float(np.std(scores))

    thr = mu + std_scale * sigma

    return {
        "mean": mu,
        "std": sigma,
        "threshold": float(thr),
    }


# =========================================================
# 9. metrics
# =========================================================
def calc_metrics_binary(
    y_true: List[int],
    y_pred: List[int]
):

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(len(y_true), 1)

    precision = tp / max(tp + fp, 1e-12)

    recall = tp / max(tp + fn, 1e-12)

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    return {
        "accuracy": float(accuracy),
        "precision_normal": float(precision),
        "recall_normal": float(recall),
        "f1_normal": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# =========================================================
# 10. 保存图片
# =========================================================
def save_error_image_with_text(
    src_path: str,
    dst_path: str,
    row: dict
):

    img = Image.open(src_path).convert("RGB")

    draw = ImageDraw.Draw(img)

    text = (
        f"score: {row['score']:.6f}\n"
        f"pred: {row['pred']}\n"
        f"gt: {row['gt']}"
    )

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    draw.text(
        (10, 10),
        text,
        fill=(255, 0, 0),
        font=font
    )

    dst_path = Path(dst_path)

    dst_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    img.save(dst_path)


def save_wrong_images_for_cow(
    error_root: str,
    cow_id: str,
    normal_to_abnormal: list,
    abnormal_to_normal: list,
):

    error_root = Path(error_root)

    n2a_dir = (
        error_root
        / str(cow_id)
        / "normal_to_abnormal"
    )

    a2n_dir = (
        error_root
        / str(cow_id)
        / "abnormal_to_normal"
    )

    n2a_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    a2n_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for row in normal_to_abnormal:

        src = Path(row["path"])

        if src.exists():

            dst = n2a_dir / src.name

            save_error_image_with_text(
                str(src),
                str(dst),
                row
            )

    for row in abnormal_to_normal:

        src = Path(row["path"])

        if src.exists():

            dst = a2n_dir / src.name

            save_error_image_with_text(
                str(src),
                str(dst),
                row
            )


# =========================================================
# 11. 主流程
# =========================================================
def evaluate_unseen_ori_with_gt_check(
    split_json_path: str,
    ori_root: str,
    gt_root: str,
    model_path: str,
    overall_csv: str,
    error_root: str,
    batch_size: int = 64,
    seed: int = 42,
    std_scale: float = 1.0,
):

    set_seed(seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    transform = build_eval_transforms(
        img_size=224
    )

    unseen_cows = load_split_ids(
        split_json_path,
        split_key="unseen_cows"
    )

    print(f"[INFO] unseen cows = {len(unseen_cows)}")

    model = build_model_from_ckpt(
        model_path,
        device=device
    )

    all_y_true = []
    all_y_pred = []

    rows = []

    for cow_id in unseen_cows:

        print("\n" + "=" * 80)
        print(f"[INFO] Processing cow_id = {cow_id}")

        ori_paths = list_ori_images_for_cow(
            ori_root,
            cow_id
        )

        gt_dict = build_gt_dict_for_cow(
            gt_root,
            cow_id
        )

        if len(ori_paths) < 2:
            print(f"[WARN] ori 图片少于2张，跳过: {cow_id}")
            continue

        if len(gt_dict) == 0:
            print(f"[WARN] GT为空，跳过: {cow_id}")
            continue

        X, emb_paths = extract_embeddings_for_paths(
            model=model,
            image_paths=ori_paths,
            transform=transform,
            device=device,
            batch_size=batch_size,
        )

        if X.shape[0] < 2:
            continue

        scores = calc_center_euclidean_scores(X)

        thr_info = build_threshold_mean_std(
            scores,
            std_scale=std_scale
        )

        threshold = thr_info["threshold"]
        mean = thr_info["mean"]
        std = thr_info["std"]

        y_true = []
        y_pred = []

        normal_to_abnormal = []
        abnormal_to_normal = []

        for path, score in zip(emb_paths, scores):

            filename = Path(path).name

            pred = 1 if float(score) <= threshold else 0

            if filename not in gt_dict:
                continue

            gt = int(gt_dict[filename])

            row = {
                "path": path,
                "filename": filename,
                "score": float(score),
                "pred": int(pred),
                "gt": int(gt),
            }

            y_true.append(gt)
            y_pred.append(pred)

            if gt == 1 and pred == 0:
                normal_to_abnormal.append(row)

            elif gt == 0 and pred == 1:
                abnormal_to_normal.append(row)

        if len(y_true) == 0:
            continue

        metrics = calc_metrics_binary(
            y_true,
            y_pred
        )

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        print(f"[INFO] threshold            = {threshold:.6f}")
        print(f"[INFO] metrics              = {metrics}")
        print(f"[INFO] normal_to_abnormal   = {len(normal_to_abnormal)}")
        print(f"[INFO] abnormal_to_normal   = {len(abnormal_to_normal)}")

        save_wrong_images_for_cow(
            error_root=error_root,
            cow_id=cow_id,
            normal_to_abnormal=normal_to_abnormal,
            abnormal_to_normal=abnormal_to_normal,
        )

        rows.append({
            "cow_id": cow_id,
           "mean":mean,
           "std":std,
            "threshold": threshold,
            "accuracy": metrics["accuracy"],
            "precision_normal": metrics["precision_normal"],
            "recall_normal": metrics["recall_normal"],
            "f1_normal": metrics["f1_normal"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "num_normal_to_abnormal": len(normal_to_abnormal),
            "num_abnormal_to_normal": len(abnormal_to_normal),
        })

    overall_metrics = calc_metrics_binary(
        all_y_true,
        all_y_pred
    )

    print("\n" + "#" * 80)
    print("[INFO] OVERALL")
    print(overall_metrics)
    print("#" * 80)

    fieldnames = list(rows[0].keys())

    with open(
        overall_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)

    print(f"[INFO] csv saved to: {overall_csv}")


# =========================================================
# 12. main
# =========================================================
def main():
    model_name = "exp001_3.31_arcface_CHA_GAP"

    split_json_path = (
        r"D:\code_project\triplet_code\data\dataset_OBB\split.json"
    )

    ori_root = (
        r"D:\code_project\triplet_code\data\dataset_obb_ori"
    )

    gt_root = (
        r"D:\code_project\triplet_code\data\dataset_OBB"
    )

    model_path = (
        r"D:\code_project\triplet_code\logs\{}\checkpoints\latest.pth".format(model_name)
    )

    overall_csv = (
        r"D:\code_project\triplet_code\logs\{}\unseen_ori_eval_overall_metrics.csv".format(model_name)
    )

    error_root = (
        r"D:\code_project\triplet_code\logs\{}\unseen_ori_error_images".format(model_name)
    )

    evaluate_unseen_ori_with_gt_check(
        split_json_path=split_json_path,
        ori_root=ori_root,
        gt_root=gt_root,
        model_path=model_path,
        overall_csv=overall_csv,
        error_root=error_root,
        batch_size=64,
        seed=42,
        std_scale=1.0,
    )


if __name__ == "__main__":
    main()