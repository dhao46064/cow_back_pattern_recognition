# import os
# import sys
# import gc
# import json
# import random
# from pathlib import Path
# from itertools import combinations



# PROJECT_ROOT = os.path.abspath(
#     os.path.join(
#         os.path.dirname(__file__),
#         "../.."
#     )
# )


# sys.path.append(
#     PROJECT_ROOT
# )
# # import numpy as np
# # from PIL import Image

# # import torch
# # import torch.nn.functional as F
# # from torch.utils.data import Dataset, DataLoader
# # from torchvision import transforms
# # import matplotlib.pyplot as plt



# # # =========================================================
# # # 0. 基础配置
# # # =========================================================
# # def set_seed(seed=42):
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     torch.cuda.manual_seed_all(seed)


# # def fresh_start():
# #     gc.collect()
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()
# #     plt.close("all")


# # # =========================================================
# # # 1. 路径设置
# # # =========================================================
# # PROJECT_ROOT = Path(__file__).resolve().parents[1]
# # if str(PROJECT_ROOT) not in sys.path:
# #     sys.path.append(str(PROJECT_ROOT))

# # from src.sup.models.dolg_arcface_model import DOLGArcFaceCowModel
# # from src.sup.datasets.split_cow import get_cow_ids


# # # =========================================================
# # # 2. 变换
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
# # # 3. 数据扫描
# # # =========================================================
# # IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# # def is_image_file(path: Path) -> bool:
# #     return path.suffix.lower() in IMG_EXTS


# # def list_normal_images_for_cow(root_dir: str, cow_id: str):
# #     """
# #     只读 root/cow_id/normal/*
# #     """
# #     normal_dir = Path(root_dir) / cow_id / "normal"
# #     files = []
# #     if normal_dir.exists() and normal_dir.is_dir():
# #         for p in normal_dir.iterdir():
# #             if p.is_file() and is_image_file(p):
# #                 files.append(str(p))
# #     return sorted(files)


# # # =========================================================
# # # 4. checkpoint / model
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
# #     ckpt, state_dict = load_checkpoint(model_path)
# #     num_classes, embedding_dim = infer_arcface_config(state_dict)

# #     print(f"[INFO] model_path  : {model_path}")
# #     print(f"[INFO] num_classes : {num_classes}")
# #     print(f"[INFO] emb_dim     : {embedding_dim}")

# #     model = DOLGArcFaceCowModel(
# #         num_classes=num_classes,
# #         embedding_dim=embedding_dim,
# #         local_dim=512,
# #         fusion_hidden_dim=512,
# #         s=30.0,
# #         m=0.25,
# #         pretrained=False,
# #         freeze_backbone=False,
# #         gem_p=3.0,
# #         gem_learn_p=True,
# #         dilation_rates=(1, 2, 3),
# #         dropout=0.3,
# #     )

# #     missing, unexpected = model.load_state_dict(state_dict, strict=False)
# #     print("[INFO] missing keys   :", len(missing))
# #     print("[INFO] unexpected keys:", len(unexpected))

# #     if len(missing) > 0 or len(unexpected) > 0:
# #         print("[WARN] 当前本地模型源码与 checkpoint 结构不完全一致，结果可能不可信。")

# #     model = model.to(device)
# #     model.eval()
# #     return model


# # # =========================================================
# # # 5. 单牛数据集
# # # =========================================================
# # class ImageListDataset(Dataset):
# #     def __init__(self, img_paths, transform=None):
# #         self.img_paths = img_paths
# #         self.transform = transform

# #     def __len__(self):
# #         return len(self.img_paths)

# #     def __getitem__(self, idx):
# #         path = self.img_paths[idx]
# #         img = Image.open(path).convert("RGB")
# #         if self.transform:
# #             img = self.transform(img)
# #         return img, path


# # @torch.no_grad()
# # def extract_embeddings_for_paths(model, img_paths, transform, device, batch_size=64):
# #     dataset = ImageListDataset(img_paths, transform=transform)
# #     dataloader = DataLoader(
# #         dataset,
# #         batch_size=batch_size,
# #         shuffle=False,
# #         num_workers=4,
# #         pin_memory=torch.cuda.is_available(),
# #     )

# #     all_emb = []
# #     all_paths = []

# #     for imgs, paths in dataloader:
# #         imgs = imgs.to(device, non_blocking=True)
# #         emb = model.forward_emb(imgs)
# #         emb = F.normalize(emb, p=2, dim=1)
# #         all_emb.append(emb.cpu())
# #         all_paths.extend(list(paths))

# #     all_emb = torch.cat(all_emb, dim=0)
# #     return all_emb, all_paths


# # # =========================================================
# # # 6. pair 统计
# # # =========================================================
# # def sample_same_pairs(emb: torch.Tensor, max_pairs=2000, seed=42):
# #     rng = random.Random(seed)
# #     n = emb.shape[0]
# #     if n < 2:
# #         return []

# #     all_idx = list(combinations(range(n), 2))
# #     if len(all_idx) > max_pairs:
# #         all_idx = rng.sample(all_idx, max_pairs)

# #     dists = []
# #     for i, j in all_idx:
# #         d = 1.0 - torch.sum(emb[i] * emb[j]).item()
# #         dists.append(d)
# #     return dists


# # def sample_diff_pairs(emb_by_cow: dict, max_pairs=5000, seed=42):
# #     rng = random.Random(seed)
# #     cow_ids = list(emb_by_cow.keys())
# #     dists = []

# #     if len(cow_ids) < 2:
# #         return dists

# #     while len(dists) < max_pairs:
# #         cid1, cid2 = rng.sample(cow_ids, 2)
# #         e1 = emb_by_cow[cid1]
# #         e2 = emb_by_cow[cid2]

# #         i = rng.randrange(e1.shape[0])
# #         j = rng.randrange(e2.shape[0])

# #         d = 1.0 - torch.sum(e1[i] * e2[j]).item()
# #         dists.append(d)

# #     return dists


# # def summarize_dist(name, arr):
# #     arr = np.asarray(arr, dtype=np.float32)
# #     print(f"\n[{name}]")
# #     print(f"count = {len(arr)}")
# #     print(f"mean  = {arr.mean():.6f}")
# #     print(f"std   = {arr.std():.6f}")
# #     print(f"min   = {arr.min():.6f}")
# #     print(f"p25   = {np.percentile(arr, 25):.6f}")
# #     print(f"p50   = {np.percentile(arr, 50):.6f}")
# #     print(f"p75   = {np.percentile(arr, 75):.6f}")
# #     print(f"max   = {arr.max():.6f}")


# # def plot_histogram(same_dists, diff_dists, save_path, title_suffix=""):
# #     plt.figure(figsize=(8, 6))
# #     plt.hist(same_dists, bins=50, alpha=0.6, label="same-normal", density=True)
# #     plt.hist(diff_dists, bins=50, alpha=0.6, label="diff-normal", density=True)
# #     plt.xlabel("Cosine Distance")
# #     plt.ylabel("Density")
# #     plt.title(f"Same-normal vs Diff-normal Distance Histogram {title_suffix}")
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.savefig(save_path, dpi=300, bbox_inches="tight")
# #     print(f"[INFO] histogram saved to: {save_path}")
# #     plt.show()
# #     plt.close()


# # # =========================================================
# # # 7. 主逻辑
# # # =========================================================
# # def run_pair_analysis(
# #     save_path,
# #     root_dir: str,
# #     model_path: str,
# #     split_mode: str = "seen",   # "seen" or "unseen"
    
# #     max_per_cow: int = 150,
# #     batch_size: int = 64,
# #     seed: int = 42,
# # ):
# #     set_seed(seed)
# #     fresh_start()

# #     device = "cuda" if torch.cuda.is_available() else "cpu"

# #     if split_mode == "seen":
# #         cow_ids = get_cow_ids(root_dir, split="seen_cows")
# #     elif split_mode == "unseen":
# #         cow_ids = get_cow_ids(root_dir, split="unseen_cows")
# #     else:
# #         raise ValueError("split_mode must be 'seen' or 'unseen'")

# #     print(f"[INFO] split_mode = {split_mode}")
# #     print(f"[INFO] num cows   = {len(cow_ids)}")
# #     print(f"[INFO] cows       = {cow_ids[:10]} ...")

# #     transform = build_eval_transforms(img_size=224)
# #     model = build_model_from_ckpt(model_path, device=device)

# #     emb_by_cow = {}

# #     for cid in cow_ids:
# #         img_paths = list_normal_images_for_cow(root_dir, cid)
# #         if len(img_paths) < 2:
# #             print(f"[WARN] cow {cid} normal images < 2, skip")
# #             continue

# #         if len(img_paths) > max_per_cow:
# #             img_paths = random.sample(img_paths, max_per_cow)

# #         emb, paths = extract_embeddings_for_paths(
# #             model=model,
# #             img_paths=img_paths,
# #             transform=transform,
# #             device=device,
# #             batch_size=batch_size,
# #         )
# #         emb_by_cow[cid] = emb
# #         print(f"[INFO] {cid}: {emb.shape[0]} embeddings")

# #     if len(emb_by_cow) < 2:
# #         print("[WARN] valid cows < 2, cannot compute diff-normal")
# #         return

# #     same_dists = []
# #     for cid, emb in emb_by_cow.items():
# #         same_dists.extend(sample_same_pairs(emb, max_pairs=1000, seed=seed))

# #     diff_dists = sample_diff_pairs(emb_by_cow, max_pairs=10000, seed=seed)

# #     summarize_dist(f"{split_mode}-same-normal", same_dists)
# #     summarize_dist(f"{split_mode}-diff-normal", diff_dists)

# #     same_mean = np.mean(same_dists)
# #     diff_mean = np.mean(diff_dists)

# #     print(f"\n[INFO] split_mode = {split_mode}")
# #     print(f"[INFO] inter/intra ratio = {diff_mean / (same_mean + 1e-12):.6f}")
# #     print(f"[INFO] hardest positive (max same) = {np.max(same_dists):.6f}")
# #     print(f"[INFO] hardest negative (min diff) = {np.min(diff_dists):.6f}")

# #     exp_name = Path(model_path).parents[1].name
# #     # save_path = str(PROJECT_ROOT / f"pair_hist_{split_mode}_{exp_name}.png")
# #     plot_histogram(
# #         same_dists,
# #         diff_dists,
# #         save_path=save_path,
# #         title_suffix=f"({split_mode})"
# #     )

# #     # 结果也存一份 json
# #     result = {
# #         "split_mode": split_mode,
# #         "num_cows_used": len(emb_by_cow),
# #         "same_count": len(same_dists),
# #         "diff_count": len(diff_dists),
# #         "same_mean": float(np.mean(same_dists)),
# #         "same_std": float(np.std(same_dists)),
# #         "same_min": float(np.min(same_dists)),
# #         "same_max": float(np.max(same_dists)),
# #         "diff_mean": float(np.mean(diff_dists)),
# #         "diff_std": float(np.std(diff_dists)),
# #         "diff_min": float(np.min(diff_dists)),
# #         "diff_max": float(np.max(diff_dists)),
# #         "inter_intra_ratio": float(diff_mean / (same_mean + 1e-12)),
# #         "hardest_positive": float(np.max(same_dists)),
# #         "hardest_negative": float(np.min(diff_dists)),
# #     }

# #     json_path = str(PROJECT_ROOT / f"pair_stats_{split_mode}_{exp_name}.json")
# #     with open(json_path, "w", encoding="utf-8") as f:
# #         json.dump(result, f, indent=2, ensure_ascii=False)

# #     print(f"[INFO] stats saved to: {json_path}")


# # # =========================================================
# # # 8. main
# # # =========================================================
# # def main():
# #     root_dir = r"D:\code_project\triplet_code\data\dataset_OBB"
# #     model_name = r"D:\datasets\openset_CowBack\openset_ori_8.2"
# #     model_path = r"D:\code_project\triplet_code\logs\exp003_7.25_arcface_DOLG\checkpoints\epoch_35.pth"
# #     save_path = r"{}/unseen_pair_hist.png".format(model_name)

# #     # 二选一：
# #     # split_mode = "seen"
# #     split_mode = "unseen"

# #     run_pair_analysis(
# #         save_path=save_path,
# #         root_dir=root_dir,
# #         model_path=model_path,
# #         split_mode=split_mode,
# #         max_per_cow=150,
# #         batch_size=64,
# #         seed=42,
# #     )


# # if __name__ == "__main__":
# #     main()
# # -*- coding: utf-8 -*-

# import sys
# import gc
# import random
# from datetime import datetime
# from pathlib import Path

# import numpy as np
# from PIL import Image

# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from tqdm import tqdm


# # =========================================================
# # 0. 基础配置
# # =========================================================
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# SCRIPT_DIR = Path(__file__).resolve().parent

# for candidate in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
#     if (candidate / "src" / "__init__.py").is_file():
#         PROJECT_ROOT = candidate
#         break
# else:
#     raise RuntimeError(
#         f"无法从脚本位置找到项目根目录：{SCRIPT_DIR}"
#     )

# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))




# def fresh_start():
#     gc.collect()

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
# # =========================================================
# # 1. 项目路径与模型导入
# # =========================================================
# # 假设本脚本位于：
# # D:\code_project\triplet_code\scripts\xxx.py
# # 此时 parents[1] 即 D:\code_project\triplet_code
# # PROJECT_ROOT = Path(__file__).resolve().parents[1]

# # if str(PROJECT_ROOT) not in sys.path:
# #     sys.path.append(str(PROJECT_ROOT))

# from src.models.dolg_arcface_model import DOLGArcFaceCowModel
# from src.datasets.split import get_cow_ids


# IMG_EXTS = {
#     ".jpg",
#     ".jpeg",
#     ".png",
#     ".bmp",
#     ".tif",
#     ".tiff",
#     ".webp",
# }


# # =========================================================
# # 2. 单文档日志
# # =========================================================
# class ReportLogger:
#     """
#     同时输出到控制台和一个TXT报告文件。
#     """

#     def __init__(self, report_path):
#         self.report_path = Path(report_path)
#         self.report_path.parent.mkdir(
#             parents=True,
#             exist_ok=True,
#         )

#         self.file = open(
#             self.report_path,
#             "w",
#             encoding="utf-8",
#         )

#     def log(self, text=""):
#         text = str(text)

#         print(text)

#         self.file.write(text + "\n")
#         self.file.flush()

#     def close(self):
#         self.file.close()


# # =========================================================
# # 3. 图像变换
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
# # 4. 数据扫描
# # =========================================================
# def is_image_file(path: Path) -> bool:
#     return path.suffix.lower() in IMG_EXTS


# def list_normal_images_for_cow(root_dir: str, cow_id: str):
#     """
#     只读取：
#         root_dir/cow_id/normal/*
#     """
#     normal_dir = (
#         Path(root_dir)
#         /
#         str(cow_id)
#         /
#         "normal"
#     )

#     if not normal_dir.exists():
#         return []

#     files = [
#         str(path)
#         for path in normal_dir.iterdir()
#         if path.is_file() and is_image_file(path)
#     ]

#     return sorted(files)


# # =========================================================
# # 5. checkpoint与模型
# # =========================================================
# def load_checkpoint(model_path: str):
#     ckpt = torch.load(
#         model_path,
#         map_location="cpu",
#         weights_only=False,
#     )

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
#         raise KeyError(
#             "checkpoint中没有arcface.weight，"
#             "无法自动推断num_classes与embedding_dim。"
#         )

#     arcface_weight = state_dict["arcface.weight"]

#     num_classes = int(
#         arcface_weight.shape[0]
#     )

#     embedding_dim = int(
#         arcface_weight.shape[1]
#     )

#     return num_classes, embedding_dim


# def build_model_from_ckpt(
#     model_path: str,
#     device: str,
#     logger,
# ):
#     _, state_dict = load_checkpoint(
#         model_path
#     )

#     num_classes, embedding_dim = infer_arcface_config(
#         state_dict
#     )

#     logger.log(f"模型路径：{model_path}")
#     logger.log(f"类别数量：{num_classes}")
#     logger.log(f"Embedding维度：{embedding_dim}")

#     model = DOLGArcFaceCowModel(
#         num_classes=num_classes,
#         embedding_dim=embedding_dim,
#         local_dim=512,
#         fusion_hidden_dim=512,
#         s=30.0,
#         m=0.25,
#         pretrained=False,
#         freeze_backbone=False,
#         gem_p=3.0,
#         gem_learn_p=True,
#         dilation_rates=(1, 2, 3),
#         dropout=0.3,
#     )

#     missing, unexpected = model.load_state_dict(
#         state_dict,
#         strict=False,
#     )

#     logger.log(f"缺失权重键数量：{len(missing)}")
#     logger.log(f"额外权重键数量：{len(unexpected)}")

#     if missing or unexpected:
#         logger.log(
#             "警告：模型源码与checkpoint结构不完全一致，"
#             "请检查模型定义和权重。"
#         )

#     model = model.to(device)
#     model.eval()

#     return model


# # =========================================================
# # 6. 数据集与Embedding提取
# # =========================================================
# class ImageListDataset(Dataset):
#     def __init__(
#         self,
#         img_paths,
#         transform=None,
#     ):
#         self.img_paths = img_paths
#         self.transform = transform

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, index):
#         image_path = self.img_paths[index]

#         image = Image.open(
#             image_path
#         ).convert("RGB")

#         if self.transform is not None:
#             image = self.transform(image)

#         return image, image_path


# @torch.no_grad()
# def extract_embeddings_for_paths(
#     model,
#     img_paths,
#     transform,
#     device,
#     batch_size=64,
#     num_workers=4,
# ):
#     dataset = ImageListDataset(
#         img_paths=img_paths,
#         transform=transform,
#     )

#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=torch.cuda.is_available(),
#     )

#     all_embeddings = []
#     all_paths = []

#     for images, paths in dataloader:
#         images = images.to(
#             device,
#             non_blocking=True,
#         )

#         embeddings = model.forward_emb(
#             images
#         )

#         embeddings = F.normalize(
#             embeddings,
#             p=2,
#             dim=1,
#         )

#         all_embeddings.append(
#             embeddings.cpu()
#         )

#         all_paths.extend(
#             list(paths)
#         )

#     if not all_embeddings:
#         return torch.empty((0, 0)), []

#     all_embeddings = torch.cat(
#         all_embeddings,
#         dim=0,
#     )

#     return all_embeddings, all_paths


# # =========================================================
# # 7. 类内全部配对
# # =========================================================
# def compute_all_same_pairs(
#     embeddings: torch.Tensor,
# ):
#     """
#     计算同一头牛内部所有不重复图片对。

#     n张图片的Pair数量：
#         n * (n - 1) / 2

#     embeddings已经完成L2归一化，因此：
#         cosine_distance = 1 - embedding_i @ embedding_j
#     """
#     image_count = int(
#         embeddings.shape[0]
#     )

#     if image_count < 2:
#         return []

#     similarity_matrix = (
#         embeddings
#         @
#         embeddings.T
#     )

#     row_indices, col_indices = torch.triu_indices(
#         row=image_count,
#         col=image_count,
#         offset=1,
#     )

#     distances = (
#         1.0
#         -
#         similarity_matrix[
#             row_indices,
#             col_indices,
#         ]
#     )

#     return (
#         distances
#         .cpu()
#         .numpy()
#         .astype(np.float64)
#         .tolist()
#     )


# # =========================================================
# # 8. 类间随机采样10000个Pair
# # =========================================================
# def sample_diff_pairs(
#     embeddings_by_cow: dict,
#     max_pairs=10000,
#     seed=42,
# ):
#     """
#     类间Pair采样逻辑：

#     1. 随机选择两个不同牛编号；
#     2. 在两头牛中分别随机选择一张图片；
#     3. 计算余弦距离；
#     4. 重复max_pairs次。

#     当前为有放回随机采样，因此最终Pair数量固定为max_pairs。
#     """
#     rng = random.Random(seed)

#     cow_ids = list(
#         embeddings_by_cow.keys()
#     )

#     if len(cow_ids) < 2:
#         return []

#     distances = []

#     for _ in tqdm(
#         range(max_pairs),
#         desc="计算类间Pair",
#     ):
#         cow_id_1, cow_id_2 = rng.sample(
#             cow_ids,
#             2,
#         )

#         embeddings_1 = embeddings_by_cow[
#             cow_id_1
#         ]

#         embeddings_2 = embeddings_by_cow[
#             cow_id_2
#         ]

#         index_1 = rng.randrange(
#             embeddings_1.shape[0]
#         )

#         index_2 = rng.randrange(
#             embeddings_2.shape[0]
#         )

#         cosine_distance = (
#             1.0
#             -
#             torch.sum(
#                 embeddings_1[index_1]
#                 *
#                 embeddings_2[index_2]
#             ).item()
#         )

#         distances.append(
#             cosine_distance
#         )

#     return distances


# # =========================================================
# # 9. 距离统计
# # =========================================================
# def compute_statistics(values):
#     array = np.asarray(
#         values,
#         dtype=np.float64,
#     )

#     if array.size == 0:
#         return None

#     return {
#         "pair_count": int(array.size),
#         "distance_sum": float(array.sum()),
#         "mean": float(array.mean()),
#         "std": float(array.std()),
#         "min": float(array.min()),
#         "p25": float(np.percentile(array, 25)),
#         "p50": float(np.percentile(array, 50)),
#         "p75": float(np.percentile(array, 75)),
#         "p95": float(np.percentile(array, 95)),
#         "p99": float(np.percentile(array, 99)),
#         "max": float(array.max()),
#     }


# def log_statistics(
#     logger,
#     title,
#     statistics,
# ):
#     logger.log("")
#     logger.log("=" * 60)
#     logger.log(title)
#     logger.log("=" * 60)

#     if statistics is None:
#         logger.log("没有可用数据")
#         return

#     logger.log(
#         f"Pair数量       ："
#         f"{statistics['pair_count']}"
#     )

#     logger.log(
#         f"Distance总和   ："
#         f"{statistics['distance_sum']:.6f}"
#     )

#     logger.log(
#         f"平均Distance   ："
#         f"{statistics['mean']:.6f}"
#     )

#     logger.log(
#         f"标准差         ："
#         f"{statistics['std']:.6f}"
#     )

#     logger.log(
#         f"最小值         ："
#         f"{statistics['min']:.6f}"
#     )

#     logger.log(
#         f"P25            ："
#         f"{statistics['p25']:.6f}"
#     )

#     logger.log(
#         f"P50/中位数     ："
#         f"{statistics['p50']:.6f}"
#     )

#     logger.log(
#         f"P75            ："
#         f"{statistics['p75']:.6f}"
#     )

#     logger.log(
#         f"P95            ："
#         f"{statistics['p95']:.6f}"
#     )

#     logger.log(
#         f"P99            ："
#         f"{statistics['p99']:.6f}"
#     )

#     logger.log(
#         f"最大值         ："
#         f"{statistics['max']:.6f}"
#     )


# # =========================================================
# # 10. 主分析
# # =========================================================
# def run_pair_analysis(
#     report_path: str,
#     root_dir: str,
#     model_path: str,
#     split_mode: str = "unseen",
#     max_per_cow: int = 120,
#     max_diff_pairs: int = 10000,
#     batch_size: int = 64,
#     num_workers: int = 4,
#     seed: int = 42,
# ):
#     logger = ReportLogger(
#         report_path
#     )

#     try:
#         set_seed(seed)
#         fresh_start()

#         device = (
#             "cuda"
#             if torch.cuda.is_available()
#             else "cpu"
#         )

#         logger.log("=" * 70)
#         logger.log("DOLG类内/类间余弦距离统计报告")
#         logger.log("=" * 70)

#         logger.log(
#             "运行时间："
#             + datetime.now().strftime(
#                 "%Y-%m-%d %H:%M:%S"
#             )
#         )

#         logger.log(f"数据目录：{root_dir}")
#         logger.log(f"模型路径：{model_path}")
#         logger.log(f"划分模式：{split_mode}")
#         logger.log(f"计算设备：{device}")
#         logger.log(f"随机种子：{seed}")
#         logger.log(
#             f"每头牛最多使用图片数：{max_per_cow}"
#         )
#         logger.log("类内Pair策略：每头牛全部配对")
#         logger.log(
#             f"类间Pair策略：随机采样{max_diff_pairs}对"
#         )
#         logger.log(f"Batch size：{batch_size}")
#         logger.log(
#             f"DataLoader workers：{num_workers}"
#         )
#         logger.log("")

#         if split_mode == "seen":
#             cow_ids = get_cow_ids(
#                 root_dir,
#                 split="seen_cows",
#             )

#         elif split_mode == "unseen":
#             cow_ids = get_cow_ids(
#                 root_dir,
#                 split="unseen_cows",
#             )

#         else:
#             raise ValueError(
#                 "split_mode必须是'seen'或'unseen'"
#             )

#         cow_ids = [
#             str(cow_id)
#             for cow_id in cow_ids
#         ]

#         logger.log(
#             f"划分中的牛编号数量：{len(cow_ids)}"
#         )

#         transform = build_eval_transforms(
#             img_size=224
#         )

#         model = build_model_from_ckpt(
#             model_path=model_path,
#             device=device,
#             logger=logger,
#         )

#         embeddings_by_cow = {}
#         images_used_by_cow = {}
#         skipped_cows = []

#         logger.log("")
#         logger.log("=" * 70)
#         logger.log("各牛编号图片采集情况")
#         logger.log("=" * 70)

#         for cow_index, cow_id in enumerate(
#             tqdm(
#                 cow_ids,
#                 desc="提取各牛Embedding",
#             ),
#             start=1,
#         ):
#             image_paths = list_normal_images_for_cow(
#                 root_dir=root_dir,
#                 cow_id=cow_id,
#             )

#             original_image_count = len(
#                 image_paths
#             )

#             if original_image_count < 2:
#                 skipped_cows.append(
#                     cow_id
#                 )

#                 logger.log(
#                     f"[{cow_index}/{len(cow_ids)}] "
#                     f"{cow_id}："
#                     f"原始图片={original_image_count}，"
#                     f"少于2张，跳过"
#                 )

#                 continue

#             if original_image_count > max_per_cow:
#                 image_paths = random.sample(
#                     image_paths,
#                     max_per_cow,
#                 )

#             embeddings, _ = extract_embeddings_for_paths(
#                 model=model,
#                 img_paths=image_paths,
#                 transform=transform,
#                 device=device,
#                 batch_size=batch_size,
#                 num_workers=num_workers,
#             )

#             if embeddings.shape[0] < 2:
#                 skipped_cows.append(
#                     cow_id
#                 )

#                 logger.log(
#                     f"[{cow_index}/{len(cow_ids)}] "
#                     f"{cow_id}："
#                     f"Embedding少于2，跳过"
#                 )

#                 continue

#             embeddings_by_cow[
#                 cow_id
#             ] = embeddings

#             images_used_by_cow[
#                 cow_id
#             ] = int(
#                 embeddings.shape[0]
#             )

#             logger.log(
#                 f"[{cow_index}/{len(cow_ids)}] "
#                 f"{cow_id}："
#                 f"原始图片={original_image_count}，"
#                 f"实际使用={embeddings.shape[0]}"
#             )

#         if len(embeddings_by_cow) < 2:
#             raise RuntimeError(
#                 "有效牛编号少于2，"
#                 "无法计算类间距离。"
#             )

#         # -------------------------------------------------
#         # 类内：每头牛所有不重复Pair
#         # -------------------------------------------------
#         all_same_distances = []
#         same_pair_count_by_cow = {}

#         logger.log("")
#         logger.log("开始计算全部类内Pair……")

#         for cow_id, embeddings in tqdm(
#             embeddings_by_cow.items(),
#             desc="计算全部类内Pair",
#         ):
#             same_distances = compute_all_same_pairs(
#                 embeddings
#             )

#             all_same_distances.extend(
#                 same_distances
#             )

#             same_pair_count_by_cow[
#                 cow_id
#             ] = len(
#                 same_distances
#             )

#         # -------------------------------------------------
#         # 类间：随机采样10000个Pair
#         # -------------------------------------------------
#         all_diff_distances = sample_diff_pairs(
#             embeddings_by_cow=embeddings_by_cow,
#             max_pairs=max_diff_pairs,
#             seed=seed,
#         )

#         same_statistics = compute_statistics(
#             all_same_distances
#         )

#         diff_statistics = compute_statistics(
#             all_diff_distances
#         )

#         # -------------------------------------------------
#         # 汇总
#         # -------------------------------------------------
#         logger.log("")
#         logger.log("=" * 70)
#         logger.log("总体数据规模")
#         logger.log("=" * 70)

#         logger.log(
#             f"参与计算的牛编号数量："
#             f"{len(embeddings_by_cow)}"
#         )

#         logger.log(
#             f"跳过的牛编号数量："
#             f"{len(skipped_cows)}"
#         )

#         logger.log(
#             f"实际使用图片总数："
#             f"{sum(images_used_by_cow.values())}"
#         )

#         logger.log(
#             f"全部类内Pair总数："
#             f"{len(all_same_distances)}"
#         )

#         logger.log(
#             f"随机类间Pair总数："
#             f"{len(all_diff_distances)}"
#         )

#         log_statistics(
#             logger=logger,
#             title=f"{split_mode} 类内距离统计",
#             statistics=same_statistics,
#         )

#         log_statistics(
#             logger=logger,
#             title=f"{split_mode} 类间距离统计",
#             statistics=diff_statistics,
#         )

#         if (
#             same_statistics is not None
#             and
#             diff_statistics is not None
#         ):
#             inter_intra_ratio = (
#                 diff_statistics["mean"]
#                 /
#                 (
#                     same_statistics["mean"]
#                     +
#                     1e-12
#                 )
#             )

#             mean_gap = (
#                 diff_statistics["mean"]
#                 -
#                 same_statistics["mean"]
#             )

#             logger.log("")
#             logger.log("=" * 60)
#             logger.log("核心对比参数")
#             logger.log("=" * 60)

#             logger.log(
#                 f"类间/类内均值比："
#                 f"{inter_intra_ratio:.6f}"
#             )

#             logger.log(
#                 f"类间与类内均值间隔："
#                 f"{mean_gap:.6f}"
#             )

#             logger.log(
#                 f"最难正样本距离："
#                 f"{same_statistics['max']:.6f}"
#             )

#             logger.log(
#                 f"最难负样本距离："
#                 f"{diff_statistics['min']:.6f}"
#             )

#         logger.log("")
#         logger.log("=" * 70)
#         logger.log("每头牛类内Pair数量")
#         logger.log("=" * 70)
#         logger.log(
#             "cow_id\t使用图片数\t全部类内Pair数"
#         )

#         for cow_id in embeddings_by_cow:
#             logger.log(
#                 f"{cow_id}\t"
#                 f"{images_used_by_cow[cow_id]}\t"
#                 f"{same_pair_count_by_cow[cow_id]}"
#             )

#         if skipped_cows:
#             logger.log("")
#             logger.log(
#                 "跳过的牛编号："
#                 +
#                 ", ".join(skipped_cows)
#             )

#         logger.log("")
#         logger.log("=" * 70)
#         logger.log("统计完成")
#         logger.log("=" * 70)

#         logger.log(
#             f"报告保存位置："
#             f"{Path(report_path).resolve()}"
#         )

#     except Exception as error:
#         logger.log("")
#         logger.log("=" * 70)
#         logger.log("运行失败")
#         logger.log("=" * 70)

#         logger.log(
#             f"{type(error).__name__}: {error}"
#         )

#         raise

#     finally:
#         logger.close()


# # =========================================================
# # 11. main
# # =========================================================
# def main():
#     # 数据集目录
#     root_dir = (
#         r"D:\datasets\cow_recognition8.3"
#         r"\openset\openset_ori_8.2"
#     )

#     # 报告输出目录
#     # output_dir = (
#     #      r"D:\code_project\triplet_code"
#     #     r"\logs\exp003_7.25_arcface_DOLG"
#     # )

#     # DOLG模型权重
#     model_path = (
#         r"D:\code_project\triplet_code"
#         r"\logs\exp003_7.25_arcface_DOLG"
#         r"\checkpoints\epoch_10.pth"
#     )

#     # seen 或 unseen
#     split_mode = "unseen"

#     report_path = str(
#         r"D:\code_project\triplet_code"
#         r"\logs\exp003_7.25_arcface_DOLG"
#         f"\{split_mode}_pair_analysis_report_open_reject_ori.txt"
#     )

#     run_pair_analysis(
#         report_path=report_path,
#         root_dir=root_dir,
#         model_path=model_path,
#         split_mode=split_mode,

#         # 每头牛最多120张；
#         # 如果normal本身不超过120张，则全部使用
#         max_per_cow=200,

#         # 类间固定随机采样10000个Pair
#         max_diff_pairs=10000,

#         batch_size=64,
#         num_workers=4,
#         seed=42,
#     )


# if __name__ == "__main__":
#     main()
# -*- coding: utf-8 -*-

"""
DOLG 类内/类间余弦距离分析——优化版

主要优化：

1. 所有牛的图片合并后，只创建一次 DataLoader；
2. 不再为每头牛反复启动 num_workers 子进程；
3. 默认 num_workers=0，更适合 Windows；
4. 所有 embedding 一次性提取，再按牛编号切分；
5. 类内距离使用 NumPy float32 数组保存；
6. 不再把几百万个距离转成 Python float 列表；
7. embedding 提取完成后及时释放模型和 GPU 显存；
8. 保留原来的：
   - seen/unseen 选择
   - 每头牛最多使用指定数量图片
   - 类内全部不重复配对
   - 类间随机采样
   - TXT 日志
   - JSON 结果
"""

import gc
import json
import random
import sys

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# =========================================================
# 0. 项目根目录
# =========================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent


def find_project_root(
    script_dir: Path,
) -> Path:
    """
    向上寻找包含 src 文件夹的项目根目录。
    """

    for candidate in [
        script_dir,
        *script_dir.parents,
    ]:
        if (candidate / "src").is_dir():
            return candidate

    raise RuntimeError(
        "无法找到包含 src 文件夹的项目根目录。\n"
        f"当前脚本位置：{SCRIPT_PATH}"
    )


PROJECT_ROOT = find_project_root(
    SCRIPT_DIR
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.models.dolg_arcface_model import (
    DOLGArcFaceCowModel,
)

from src.datasets.split import (
    get_cow_ids,
)


# =========================================================
# 1. 全局配置
# =========================================================

IMG_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# 是否在 checkpoint 结构不一致时直接停止
STRICT_CHECKPOINT = True


# Windows 下建议为 0
NUM_WORKERS = 0


# 是否启用 pin_memory
# 当前使用 num_workers=0，为稳定起见关闭
PIN_MEMORY = False


# 是否使用 AMP 半精度推理
# 为保持与原实验数值尽量一致，默认关闭
USE_AMP = False


# =========================================================
# 2. 基础配置
# =========================================================

def set_seed(
    seed: int = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clear_memory() -> None:
    """
    主动释放 Python 对象和未使用的 GPU 缓存。

    不在每个 batch 或每头牛后调用，
    避免频繁清空 PyTorch 缓存造成速度下降。
    """

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def configure_torch() -> None:
    """
    设置 PyTorch 推理参数。
    """

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

        try:
            torch.set_float32_matmul_precision(
                "high"
            )
        except Exception:
            pass


# =========================================================
# 3. 日志
# =========================================================

class ReportLogger:
    """
    同时输出到控制台和 TXT 文件。
    """

    def __init__(
        self,
        report_path: str,
    ):
        self.report_path = Path(
            report_path
        )

        self.report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.file = self.report_path.open(
            "w",
            encoding="utf-8",
        )

    def log(
        self,
        text="",
    ) -> None:
        text = str(text)

        print(text)

        self.file.write(
            text + "\n"
        )

        self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()


# =========================================================
# 4. 图像预处理
# =========================================================

def build_eval_transforms(
    img_size: int = 224,
):
    return transforms.Compose([
        transforms.Resize(
            (img_size, img_size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])


# =========================================================
# 5. 数据扫描
# =========================================================

def is_image_file(
    path: Path,
) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in IMG_EXTS
    )


def list_normal_images_for_cow(
    root_dir: str,
    cow_id: str,
) -> List[str]:
    """
    只读取：

        root_dir/cow_id/normal/*
    """

    normal_dir = (
        Path(root_dir)
        / str(cow_id)
        / "normal"
    )

    if not normal_dir.is_dir():
        return []

    image_paths = [
        str(path)
        for path in normal_dir.iterdir()
        if is_image_file(path)
    ]

    return sorted(
        image_paths
    )


def deterministic_sample(
    image_paths: List[str],
    max_count: int,
    seed: int,
    cow_id: str,
) -> List[str]:
    """
    每头牛使用独立固定随机种子。

    即使牛编号遍历顺序发生变化，
    同一头牛抽中的图片仍然保持一致。
    """

    if len(image_paths) <= max_count:
        return list(
            image_paths
        )

    rng = random.Random(
        f"{seed}|{cow_id}"
    )

    selected = rng.sample(
        image_paths,
        max_count,
    )

    return sorted(
        selected
    )


def collect_all_image_paths(
    root_dir: str,
    cow_ids: List[str],
    max_per_cow: int,
    seed: int,
    logger: ReportLogger,
) -> Tuple[
    List[str],
    Dict[str, Tuple[int, int]],
    Dict[str, int],
    Dict[str, int],
    List[str],
]:
    """
    收集所有牛的图片，并记录每头牛在总图片列表中的切片范围。

    返回：

    all_image_paths：
        所有图片组成的单一列表。

    slices_by_cow：
        cow_id -> (start, end)

    original_count_by_cow：
        每头牛原始图片数量。

    used_count_by_cow：
        每头牛实际使用图片数量。

    skipped_cows：
        因图片少于2张而跳过的牛。
    """

    all_image_paths: List[str] = []

    slices_by_cow: Dict[
        str,
        Tuple[int, int]
    ] = {}

    original_count_by_cow: Dict[
        str,
        int
    ] = {}

    used_count_by_cow: Dict[
        str,
        int
    ] = {}

    skipped_cows: List[str] = []

    logger.log("")
    logger.log("=" * 70)
    logger.log("各牛编号图片采集情况")
    logger.log("=" * 70)

    for cow_index, cow_id in enumerate(
        tqdm(
            cow_ids,
            desc="扫描各牛normal图片",
        ),
        start=1,
    ):
        image_paths = (
            list_normal_images_for_cow(
                root_dir=root_dir,
                cow_id=cow_id,
            )
        )

        original_count = len(
            image_paths
        )

        original_count_by_cow[
            cow_id
        ] = original_count

        if original_count < 2:
            skipped_cows.append(
                cow_id
            )

            logger.log(
                f"[{cow_index}/{len(cow_ids)}] "
                f"{cow_id}："
                f"原始图片={original_count}，"
                f"少于2张，跳过"
            )

            continue

        image_paths = deterministic_sample(
            image_paths=image_paths,
            max_count=max_per_cow,
            seed=seed,
            cow_id=cow_id,
        )

        start_index = len(
            all_image_paths
        )

        all_image_paths.extend(
            image_paths
        )

        end_index = len(
            all_image_paths
        )

        slices_by_cow[
            cow_id
        ] = (
            start_index,
            end_index,
        )

        used_count_by_cow[
            cow_id
        ] = len(
            image_paths
        )

        logger.log(
            f"[{cow_index}/{len(cow_ids)}] "
            f"{cow_id}："
            f"原始图片={original_count}，"
            f"实际使用={len(image_paths)}"
        )

    return (
        all_image_paths,
        slices_by_cow,
        original_count_by_cow,
        used_count_by_cow,
        skipped_cows,
    )


# =========================================================
# 6. Dataset
# =========================================================

class ImagePathDataset(Dataset):

    def __init__(
        self,
        image_paths: List[str],
        transform,
    ):
        self.image_paths = [
            str(path)
            for path in image_paths
        ]

        self.transform = transform

    def __len__(self) -> int:
        return len(
            self.image_paths
        )

    def __getitem__(
        self,
        index: int,
    ):
        image_path = self.image_paths[
            index
        ]

        try:
            # 使用 with，确保图像文件句柄及时释放
            with Image.open(
                image_path
            ) as image:
                image = image.convert(
                    "RGB"
                )

                image = self.transform(
                    image
                )

        except Exception as error:
            raise RuntimeError(
                f"图片读取失败：{image_path}"
            ) from error

        return image


# =========================================================
# 7. checkpoint 与模型
# =========================================================

def normalize_state_dict_keys(
    state_dict: dict,
) -> dict:
    """
    去除常见训练包装产生的键名前缀。
    """

    result = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith(
            "module."
        ):
            new_key = new_key[
                len("module.") :
            ]

        if new_key.startswith(
            "_orig_mod."
        ):
            new_key = new_key[
                len("_orig_mod.") :
            ]

        result[new_key] = value

    return result


def load_checkpoint(
    model_path: str,
):
    checkpoint_path = Path(
        model_path
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"模型权重不存在：{checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):
        state_dict = checkpoint[
            "model"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "model_state_dict"
        in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
        ]

    else:
        state_dict = checkpoint

    if not isinstance(
        state_dict,
        dict,
    ):
        raise TypeError(
            "checkpoint中没有找到有效的state_dict"
        )

    state_dict = normalize_state_dict_keys(
        state_dict
    )

    return checkpoint, state_dict


def infer_arcface_config(
    state_dict: dict,
) -> Tuple[int, int]:
    if "arcface.weight" not in state_dict:
        available_keys = list(
            state_dict.keys()
        )[:20]

        raise KeyError(
            "checkpoint中没有arcface.weight，"
            "无法推断num_classes和embedding_dim。\n"
            f"前20个权重键：{available_keys}"
        )

    arcface_weight = state_dict[
        "arcface.weight"
    ]

    num_classes = int(
        arcface_weight.shape[0]
    )

    embedding_dim = int(
        arcface_weight.shape[1]
    )

    return (
        num_classes,
        embedding_dim,
    )


def build_model_from_checkpoint(
    model_path: str,
    device: torch.device,
    logger: ReportLogger,
):
    checkpoint, state_dict = (
        load_checkpoint(
            model_path
        )
    )

    num_classes, embedding_dim = (
        infer_arcface_config(
            state_dict
        )
    )

    epoch = (
        checkpoint.get(
            "epoch",
            "unknown",
        )
        if isinstance(checkpoint, dict)
        else "unknown"
    )

    logger.log(
        f"模型路径：{model_path}"
    )

    logger.log(
        f"Checkpoint epoch：{epoch}"
    )

    logger.log(
        f"类别数量：{num_classes}"
    )

    logger.log(
        f"Embedding维度：{embedding_dim}"
    )

    model = DOLGArcFaceCowModel(
        num_classes=num_classes,
        embedding_dim=embedding_dim,

        local_dim=512,
        fusion_hidden_dim=512,

        s=30.0,
        m=0.25,

        pretrained=False,
        freeze_backbone=False,

        gem_p=3.0,
        gem_learn_p=True,

        dilation_rates=(
            1,
            2,
            3,
        ),

        dropout=0.3,
    )

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    logger.log(
        f"缺失权重键数量："
        f"{len(missing_keys)}"
    )

    logger.log(
        f"额外权重键数量："
        f"{len(unexpected_keys)}"
    )

    if missing_keys:
        logger.log(
            "缺失权重键前20项："
            + str(
                missing_keys[:20]
            )
        )

    if unexpected_keys:
        logger.log(
            "额外权重键前20项："
            + str(
                unexpected_keys[:20]
            )
        )

    if (
        STRICT_CHECKPOINT
        and (
            missing_keys
            or unexpected_keys
        )
    ):
        raise RuntimeError(
            "模型源码与checkpoint结构不一致，"
            "已停止运行，避免产生不可信结果。"
        )

    model = model.to(
        device
    )

    model.eval()

    return (
        model,
        num_classes,
        embedding_dim,
        epoch,
    )


# =========================================================
# 8. 一次性提取全部 Embedding
# =========================================================

@torch.inference_mode()
def extract_all_embeddings(
    model,
    image_paths: List[str],
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    """
    所有图片只创建一个 Dataset 和一个 DataLoader。

    返回CPU上的L2归一化embedding。
    """

    if not image_paths:
        raise ValueError(
            "图片路径列表为空"
        )

    dataset = ImagePathDataset(
        image_paths=image_paths,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,

        num_workers=num_workers,

        pin_memory=PIN_MEMORY,

        persistent_workers=(
            num_workers > 0
        ),

        drop_last=False,
    )

    embedding_chunks = []

    amp_enabled = (
        USE_AMP
        and device.type == "cuda"
    )

    for images in tqdm(
        dataloader,
        desc="一次性提取全部Embedding",
    ):
        images = images.to(
            device,
            non_blocking=PIN_MEMORY,
        )

        if amp_enabled:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                embeddings = model.forward_emb(
                    images
                )
        else:
            embeddings = model.forward_emb(
                images
            )

        embeddings = F.normalize(
            embeddings.float(),
            p=2,
            dim=1,
        )

        embedding_chunks.append(
            embeddings
            .cpu()
            .contiguous()
        )

        del images
        del embeddings

    if not embedding_chunks:
        raise RuntimeError(
            "没有成功提取任何Embedding"
        )

    all_embeddings = torch.cat(
        embedding_chunks,
        dim=0,
    )

    del embedding_chunks
    del dataloader
    del dataset

    return all_embeddings


def split_embeddings_by_cow(
    all_embeddings: torch.Tensor,
    slices_by_cow: Dict[
        str,
        Tuple[int, int]
    ],
) -> Dict[str, torch.Tensor]:
    """
    根据收集图片时记录的索引范围，
    将总embedding切分回不同牛编号。
    """

    embeddings_by_cow = {}

    for cow_id, (
        start_index,
        end_index,
    ) in slices_by_cow.items():
        embeddings_by_cow[
            cow_id
        ] = all_embeddings[
            start_index:end_index
        ]

    return embeddings_by_cow


# =========================================================
# 9. 类内全部 Pair
# =========================================================

def compute_all_same_pairs(
    embeddings: torch.Tensor,
) -> np.ndarray:
    """
    计算同一头牛内部所有不重复图片对。

    输入embedding已经L2归一化：

        cosine_distance = 1 - e1 @ e2

    返回：

        np.ndarray，dtype=float32

    不转换成Python float列表。
    """

    image_count = int(
        embeddings.shape[0]
    )

    if image_count < 2:
        return np.empty(
            0,
            dtype=np.float32,
        )

    similarity_matrix = (
        embeddings
        @ embeddings.T
    )

    row_indices, col_indices = (
        torch.triu_indices(
            row=image_count,
            col=image_count,
            offset=1,
            device=(
                similarity_matrix.device
            ),
        )
    )

    distances = (
        1.0
        - similarity_matrix[
            row_indices,
            col_indices
        ]
    )

    result = (
        distances
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    del similarity_matrix
    del row_indices
    del col_indices
    del distances

    return result


def compute_all_same_distances(
    embeddings_by_cow: Dict[
        str,
        torch.Tensor
    ],
) -> Tuple[
    np.ndarray,
    Dict[str, int],
]:
    """
    逐牛计算全部类内距离。

    每头牛产生一个较小的NumPy数组，
    最后只进行一次concatenate。
    """

    distance_chunks: List[
        np.ndarray
    ] = []

    pair_count_by_cow: Dict[
        str,
        int
    ] = {}

    for cow_id, embeddings in tqdm(
        embeddings_by_cow.items(),
        desc="计算全部类内Pair",
    ):
        cow_distances = (
            compute_all_same_pairs(
                embeddings
            )
        )

        pair_count_by_cow[
            cow_id
        ] = int(
            cow_distances.size
        )

        if cow_distances.size > 0:
            distance_chunks.append(
                cow_distances
            )

    if not distance_chunks:
        return (
            np.empty(
                0,
                dtype=np.float32,
            ),
            pair_count_by_cow,
        )

    all_distances = np.concatenate(
        distance_chunks,
        axis=0,
    )

    del distance_chunks

    return (
        all_distances,
        pair_count_by_cow,
    )


# =========================================================
# 10. 类间随机 Pair
# =========================================================

def sample_diff_pairs(
    embeddings_by_cow: Dict[
        str,
        torch.Tensor
    ],
    max_pairs: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """
    类间Pair有放回随机采样。

    每次：
    1. 随机选两头不同牛；
    2. 每头牛随机选一张图；
    3. 计算余弦距离。

    返回float32 NumPy数组。
    """

    rng = random.Random(
        seed
    )

    cow_ids = list(
        embeddings_by_cow.keys()
    )

    if len(cow_ids) < 2:
        return np.empty(
            0,
            dtype=np.float32,
        )

    distances = np.empty(
        max_pairs,
        dtype=np.float32,
    )

    for pair_index in tqdm(
        range(max_pairs),
        desc="随机采样类间Pair",
    ):
        cow_id_1, cow_id_2 = rng.sample(
            cow_ids,
            2,
        )

        embeddings_1 = (
            embeddings_by_cow[
                cow_id_1
            ]
        )

        embeddings_2 = (
            embeddings_by_cow[
                cow_id_2
            ]
        )

        index_1 = rng.randrange(
            embeddings_1.shape[0]
        )

        index_2 = rng.randrange(
            embeddings_2.shape[0]
        )

        similarity = torch.dot(
            embeddings_1[index_1],
            embeddings_2[index_2],
        ).item()

        distances[
            pair_index
        ] = (
            1.0 - similarity
        )

    return distances


# =========================================================
# 11. 距离统计
# =========================================================

def compute_statistics(
    values: np.ndarray,
):
    array = np.asarray(
        values,
        dtype=np.float32,
    )

    if array.size == 0:
        return None

    return {
        "pair_count": int(
            array.size
        ),

        # 累加使用float64，防止大量float32累加误差
        "distance_sum": float(
            array.sum(
                dtype=np.float64
            )
        ),

        "mean": float(
            array.mean(
                dtype=np.float64
            )
        ),

        "std": float(
            array.std(
                dtype=np.float64
            )
        ),

        "min": float(
            array.min()
        ),

        "p25": float(
            np.percentile(
                array,
                25,
            )
        ),

        "p50": float(
            np.percentile(
                array,
                50,
            )
        ),

        "p75": float(
            np.percentile(
                array,
                75,
            )
        ),

        "p95": float(
            np.percentile(
                array,
                95,
            )
        ),

        "p99": float(
            np.percentile(
                array,
                99,
            )
        ),

        "max": float(
            array.max()
        ),
    }


def log_statistics(
    logger: ReportLogger,
    title: str,
    statistics,
) -> None:
    logger.log("")
    logger.log("=" * 60)
    logger.log(title)
    logger.log("=" * 60)

    if statistics is None:
        logger.log(
            "没有可用数据"
        )
        return

    logger.log(
        f"Pair数量       ："
        f"{statistics['pair_count']}"
    )

    logger.log(
        f"Distance总和   ："
        f"{statistics['distance_sum']:.6f}"
    )

    logger.log(
        f"平均Distance   ："
        f"{statistics['mean']:.6f}"
    )

    logger.log(
        f"标准差         ："
        f"{statistics['std']:.6f}"
    )

    logger.log(
        f"最小值         ："
        f"{statistics['min']:.6f}"
    )

    logger.log(
        f"P25            ："
        f"{statistics['p25']:.6f}"
    )

    logger.log(
        f"P50/中位数     ："
        f"{statistics['p50']:.6f}"
    )

    logger.log(
        f"P75            ："
        f"{statistics['p75']:.6f}"
    )

    logger.log(
        f"P95            ："
        f"{statistics['p95']:.6f}"
    )

    logger.log(
        f"P99            ："
        f"{statistics['p99']:.6f}"
    )

    logger.log(
        f"最大值         ："
        f"{statistics['max']:.6f}"
    )


# =========================================================
# 12. 保存 JSON
# =========================================================

def save_json_result(
    report_path: str,
    result: dict,
) -> Path:
    report_file = Path(
        report_path
    )

    json_path = report_file.with_suffix(
        ".json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return json_path


# =========================================================
# 13. 主分析
# =========================================================

def run_pair_analysis(
    report_path: str,
    root_dir: str,
    model_path: str,
    split_mode: str = "unseen",
    max_per_cow: int = 200,
    max_diff_pairs: int = 10000,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
) -> None:
    logger = ReportLogger(
        report_path
    )

    try:
        set_seed(
            seed
        )

        configure_torch()
        clear_memory()

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.log("=" * 70)
        logger.log(
            "DOLG类内/类间余弦距离统计报告"
        )
        logger.log("=" * 70)

        logger.log(
            "运行时间："
            + datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        logger.log(
            f"项目根目录：{PROJECT_ROOT}"
        )

        logger.log(
            f"数据目录：{root_dir}"
        )

        logger.log(
            f"模型路径：{model_path}"
        )

        logger.log(
            f"划分模式：{split_mode}"
        )

        logger.log(
            f"计算设备：{device}"
        )

        if torch.cuda.is_available():
            logger.log(
                "GPU型号："
                + torch.cuda.get_device_name(
                    0
                )
            )

        logger.log(
            f"随机种子：{seed}"
        )

        logger.log(
            f"每头牛最多使用图片数："
            f"{max_per_cow}"
        )

        logger.log(
            "Embedding提取策略："
            "所有图片共用一个DataLoader"
        )

        logger.log(
            "类内Pair策略："
            "每头牛全部不重复配对"
        )

        logger.log(
            "类间Pair策略："
            f"随机采样{max_diff_pairs}对"
        )

        logger.log(
            f"Batch size：{batch_size}"
        )

        logger.log(
            f"DataLoader workers："
            f"{num_workers}"
        )

        logger.log(
            f"AMP：{USE_AMP}"
        )

        logger.log("")

        root_path = Path(
            root_dir
        )

        if not root_path.is_dir():
            raise FileNotFoundError(
                f"数据集目录不存在：{root_path}"
            )

        # -------------------------------------------------
        # 获取牛编号
        # -------------------------------------------------

        if split_mode == "seen":
            cow_ids = get_cow_ids(
                root_dir,
                split="seen_cows",
            )

        elif split_mode == "unseen":
            cow_ids = get_cow_ids(
                root_dir,
                split="unseen_cows",
            )

        else:
            raise ValueError(
                "split_mode必须为"
                "'seen'或'unseen'"
            )

        cow_ids = [
            str(cow_id)
            for cow_id in cow_ids
        ]

        logger.log(
            f"划分中的牛编号数量："
            f"{len(cow_ids)}"
        )

        if len(cow_ids) < 2:
            raise RuntimeError(
                "划分中的牛编号少于2"
            )

        # -------------------------------------------------
        # 收集全部图片
        # -------------------------------------------------

        (
            all_image_paths,
            slices_by_cow,
            original_count_by_cow,
            used_count_by_cow,
            skipped_cows,
        ) = collect_all_image_paths(
            root_dir=root_dir,
            cow_ids=cow_ids,
            max_per_cow=max_per_cow,
            seed=seed,
            logger=logger,
        )

        if len(slices_by_cow) < 2:
            raise RuntimeError(
                "有效牛编号少于2，"
                "无法计算类间距离"
            )

        logger.log("")
        logger.log("=" * 70)
        logger.log("Embedding提取规模")
        logger.log("=" * 70)

        logger.log(
            f"有效牛编号数量："
            f"{len(slices_by_cow)}"
        )

        logger.log(
            f"图片总数："
            f"{len(all_image_paths)}"
        )

        # -------------------------------------------------
        # 模型
        # -------------------------------------------------

        transform = build_eval_transforms(
            img_size=224
        )

        (
            model,
            num_classes,
            embedding_dim,
            checkpoint_epoch,
        ) = build_model_from_checkpoint(
            model_path=model_path,
            device=device,
            logger=logger,
        )

        # -------------------------------------------------
        # 所有图片一次性提取Embedding
        # -------------------------------------------------

        all_embeddings = extract_all_embeddings(
            model=model,
            image_paths=all_image_paths,
            transform=transform,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        if (
            all_embeddings.shape[0]
            != len(all_image_paths)
        ):
            raise RuntimeError(
                "Embedding数量与图片数量不一致："
                f"{all_embeddings.shape[0]} "
                f"!= {len(all_image_paths)}"
            )

        logger.log(
            f"Embedding矩阵形状："
            f"{tuple(all_embeddings.shape)}"
        )

        embeddings_by_cow = (
            split_embeddings_by_cow(
                all_embeddings=all_embeddings,
                slices_by_cow=slices_by_cow,
            )
        )

        # 模型后面不再需要，及时释放GPU
        del model
        del transform

        clear_memory()

        # -------------------------------------------------
        # 类内全部Pair
        # -------------------------------------------------

        logger.log("")
        logger.log(
            "开始计算全部类内Pair……"
        )

        (
            all_same_distances,
            same_pair_count_by_cow,
        ) = compute_all_same_distances(
            embeddings_by_cow
        )

        # -------------------------------------------------
        # 类间随机Pair
        # -------------------------------------------------

        logger.log("")
        logger.log(
            "开始随机采样类间Pair……"
        )

        all_diff_distances = sample_diff_pairs(
            embeddings_by_cow=(
                embeddings_by_cow
            ),
            max_pairs=max_diff_pairs,
            seed=seed,
        )

        # -------------------------------------------------
        # 统计
        # -------------------------------------------------

        same_statistics = compute_statistics(
            all_same_distances
        )

        diff_statistics = compute_statistics(
            all_diff_distances
        )

        logger.log("")
        logger.log("=" * 70)
        logger.log("总体数据规模")
        logger.log("=" * 70)

        logger.log(
            f"参与计算的牛编号数量："
            f"{len(embeddings_by_cow)}"
        )

        logger.log(
            f"跳过的牛编号数量："
            f"{len(skipped_cows)}"
        )

        logger.log(
            f"实际使用图片总数："
            f"{sum(used_count_by_cow.values())}"
        )

        logger.log(
            f"全部类内Pair总数："
            f"{all_same_distances.size}"
        )

        logger.log(
            f"随机类间Pair总数："
            f"{all_diff_distances.size}"
        )

        logger.log(
            f"类内距离数组内存："
            f"{all_same_distances.nbytes / 1024 / 1024:.2f} MB"
        )

        logger.log(
            f"类间距离数组内存："
            f"{all_diff_distances.nbytes / 1024 / 1024:.2f} MB"
        )

        log_statistics(
            logger=logger,
            title=(
                f"{split_mode} 类内距离统计"
            ),
            statistics=same_statistics,
        )

        log_statistics(
            logger=logger,
            title=(
                f"{split_mode} 类间距离统计"
            ),
            statistics=diff_statistics,
        )

        # -------------------------------------------------
        # 核心指标
        # -------------------------------------------------

        inter_intra_ratio = None
        mean_gap = None
        hardest_positive = None
        hardest_negative = None

        if (
            same_statistics is not None
            and diff_statistics is not None
        ):
            inter_intra_ratio = (
                diff_statistics["mean"]
                /
                (
                    same_statistics["mean"]
                    + 1e-12
                )
            )

            mean_gap = (
                diff_statistics["mean"]
                - same_statistics["mean"]
            )

            hardest_positive = (
                same_statistics["max"]
            )

            hardest_negative = (
                diff_statistics["min"]
            )

            logger.log("")
            logger.log("=" * 60)
            logger.log("核心对比参数")
            logger.log("=" * 60)

            logger.log(
                f"类间/类内均值比："
                f"{inter_intra_ratio:.6f}"
            )

            logger.log(
                f"类间与类内均值间隔："
                f"{mean_gap:.6f}"
            )

            logger.log(
                f"最难正样本距离："
                f"{hardest_positive:.6f}"
            )

            logger.log(
                f"最难负样本距离："
                f"{hardest_negative:.6f}"
            )

        # -------------------------------------------------
        # 每头牛信息
        # -------------------------------------------------

        logger.log("")
        logger.log("=" * 70)
        logger.log("每头牛类内Pair数量")
        logger.log("=" * 70)

        logger.log(
            "cow_id\t原始图片数\t"
            "使用图片数\t全部类内Pair数"
        )

        for cow_id in embeddings_by_cow:
            logger.log(
                f"{cow_id}\t"
                f"{original_count_by_cow[cow_id]}\t"
                f"{used_count_by_cow[cow_id]}\t"
                f"{same_pair_count_by_cow[cow_id]}"
            )

        if skipped_cows:
            logger.log("")
            logger.log(
                "跳过的牛编号："
                + ", ".join(
                    skipped_cows
                )
            )

        # -------------------------------------------------
        # JSON结果
        # -------------------------------------------------

        result = {
            "run_time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "project_root": str(
                PROJECT_ROOT
            ),

            "root_dir": str(
                root_dir
            ),

            "model_path": str(
                model_path
            ),

            "checkpoint_epoch": (
                checkpoint_epoch
            ),

            "split_mode": split_mode,

            "device": str(
                device
            ),

            "seed": seed,

            "max_per_cow": (
                max_per_cow
            ),

            "max_diff_pairs": (
                max_diff_pairs
            ),

            "batch_size": (
                batch_size
            ),

            "num_workers": (
                num_workers
            ),

            "use_amp": USE_AMP,

            "num_classes": (
                num_classes
            ),

            "embedding_dim": (
                embedding_dim
            ),

            "num_cows_in_split": len(
                cow_ids
            ),

            "num_cows_used": len(
                embeddings_by_cow
            ),

            "num_cows_skipped": len(
                skipped_cows
            ),

            "skipped_cows": (
                skipped_cows
            ),

            "total_images_used": int(
                sum(
                    used_count_by_cow.values()
                )
            ),

            "same_statistics": (
                same_statistics
            ),

            "diff_statistics": (
                diff_statistics
            ),

            "inter_intra_ratio": (
                float(inter_intra_ratio)
                if inter_intra_ratio is not None
                else None
            ),

            "mean_gap": (
                float(mean_gap)
                if mean_gap is not None
                else None
            ),

            "hardest_positive": (
                float(hardest_positive)
                if hardest_positive is not None
                else None
            ),

            "hardest_negative": (
                float(hardest_negative)
                if hardest_negative is not None
                else None
            ),

            "per_cow": {
                cow_id: {
                    "original_image_count": int(
                        original_count_by_cow[
                            cow_id
                        ]
                    ),

                    "used_image_count": int(
                        used_count_by_cow[
                            cow_id
                        ]
                    ),

                    "same_pair_count": int(
                        same_pair_count_by_cow[
                            cow_id
                        ]
                    ),
                }

                for cow_id in embeddings_by_cow
            },
        }

        json_path = save_json_result(
            report_path=report_path,
            result=result,
        )

        logger.log("")
        logger.log("=" * 70)
        logger.log("统计完成")
        logger.log("=" * 70)

        logger.log(
            f"TXT报告："
            f"{Path(report_path).resolve()}"
        )

        logger.log(
            f"JSON报告："
            f"{json_path.resolve()}"
        )

        # -------------------------------------------------
        # 最后释放内存
        # -------------------------------------------------

        del all_embeddings
        del embeddings_by_cow
        del all_same_distances
        del all_diff_distances

        clear_memory()

    except Exception as error:
        logger.log("")
        logger.log("=" * 70)
        logger.log("运行失败")
        logger.log("=" * 70)

        logger.log(
            f"{type(error).__name__}: "
            f"{error}"
        )

        raise

    finally:
        logger.close()


# =========================================================
# 14. main
# =========================================================

def main():
    # -----------------------------------------------------
    # 数据集
    # -----------------------------------------------------

    root_dir = (
        r"D:\datasets\cow_recognition8.3"
        r"\openset\openset_ori_8.2"
    )

    # -----------------------------------------------------
    # DOLG权重
    # -----------------------------------------------------

    model_path = (
        r"D:\code_project\triplet_code"
        r"\logs\exp003_7.25_arcface_DOLG"
        r"\checkpoints\epoch_10.pth"
    )

    # -----------------------------------------------------
    # seen 或 unseen
    # -----------------------------------------------------

    split_mode = "unseen"

    # -----------------------------------------------------
    # 报告输出
    # -----------------------------------------------------

    output_dir = (
        Path(
            r"D:\code_project\triplet_code"
            r"\logs\exp003_7.25_arcface_DOLG"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = str(
        output_dir
        / (
            f"{split_mode}"
            "_pair_analysis_report"
            "_open_reject_ori_fast.txt"
        )
    )

    # -----------------------------------------------------
    # 开始运行
    # -----------------------------------------------------

    run_pair_analysis(
        report_path=report_path,
        root_dir=root_dir,
        model_path=model_path,
        split_mode=split_mode,

        # 每头牛最多使用200张normal
        max_per_cow=200,

        # 类间固定随机采样10000对
        max_diff_pairs=10000,

        batch_size=64,

        # Windows建议保持0
        num_workers=NUM_WORKERS,

        seed=42,
    )


if __name__ == "__main__":
    main()