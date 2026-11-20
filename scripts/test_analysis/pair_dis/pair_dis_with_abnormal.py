# -*- coding: utf-8 -*-
"""
开放集Pair距离分析

最终只输出三类距离的总体平均值：
1. normal-normal：同一牛只内部的类内距离；
2. random inter-cow：随机不同牛只normal之间的距离；
3. normal-abnormal：同一ID文件夹中normal与abnormal之间的距离。

注意：当前检查点来自旧版模型，权重键名为backbone.*和head.*，
而当前模型使用embedder.backbone.*和embedder.head.*，因此加载时需要映射。
旧检查点的embedding head为2048 -> 1024 -> 128，对应hidden_dims=(1024,)。
"""

import gc
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def find_project_root():
    current = Path(__file__).resolve()
    for path in current.parents:
        if (path / "src").exists():
            return path
    raise RuntimeError("找不到包含src目录的项目根目录。请把脚本放在项目目录内运行。")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.split import get_cow_ids
from src.models.arcface_model import ArcFaceCowModel


IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fresh_start():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ReportLogger:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8")

    def log(self, message=""):
        print(message)
        self.file.write(str(message) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def build_eval_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def is_image_file(path):
    return path.suffix.lower() in IMG_EXTS


def list_images_for_cow(root, cow_id, label):
    """读取root/cow_id/normal或root/cow_id/abnormal中的图片。"""
    folder = Path(root) / str(cow_id) / label
    if not folder.exists():
        return []
    return sorted(
        str(path)
        for path in folder.iterdir()
        if path.is_file() and is_image_file(path)
    )


def deterministic_sample(paths, max_count, seed, cow_id, label):
    """按牛只和类别确定性抽样，保证重复运行选中相同索引。"""
    if max_count is None or len(paths) <= max_count:
        return paths
    rng = random.Random(f"{seed}:{cow_id}:{label}")
    selected_indices = sorted(rng.sample(range(len(paths)), max_count))
    return [paths[index] for index in selected_indices]


def load_checkpoint(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if "model" in checkpoint:
        return checkpoint["model"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def remap_checkpoint_keys(state_dict):
    mapped = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            new_key = "embedder.backbone." + key[len("backbone."):]
        elif key.startswith("head."):
            new_key = "embedder.head." + key[len("head."):]
        else:
            new_key = key
        mapped[new_key] = value
    return mapped


def build_model(model_path, device, logger):
    old_state_dict = load_checkpoint(model_path)

    if "arcface.weight" not in old_state_dict:
        raise KeyError("检查点中没有arcface.weight，无法推断类别数和embedding维度。")

    num_classes = old_state_dict["arcface.weight"].shape[0]
    embedding_dim = old_state_dict["arcface.weight"].shape[1]

    logger.log(f"classes={num_classes}")
    logger.log(f"embedding_dim={embedding_dim}")

    state_dict = remap_checkpoint_keys(old_state_dict)

    model = ArcFaceCowModel(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained=False,
        hidden_dims=(1024,),
    )

    # 严格加载：任何结构或键名不匹配都立即报错，不能静默继续。
    model.load_state_dict(state_dict, strict=True)
    logger.log("checkpoint加载成功：missing=0, unexpected=0")

    model = model.to(device)
    model.eval()
    return model


class ImageListDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        path = self.img_paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, path


@torch.no_grad()
def extract_embeddings_for_paths(
    model,
    img_paths,
    transform,
    device,
    batch_size=128,
):
    if not img_paths:
        return torch.empty((0, 0), dtype=torch.float32), []

    dataset = ImageListDataset(img_paths, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    embeddings = []
    returned_paths = []

    for images, paths_batch in dataloader:
        images = images.to(device, non_blocking=True)
        embedding = model.forward_emb(images)
        embedding = F.normalize(embedding, p=2, dim=1)
        embeddings.append(embedding.cpu())
        returned_paths.extend(paths_batch)

    result = torch.cat(embeddings, dim=0)
    del dataloader, dataset
    gc.collect()
    return result, returned_paths


def compute_same_pairs(embeddings):
    """计算同一牛只normal内部所有不重复pair的余弦距离。"""
    sample_count = embeddings.shape[0]
    if sample_count < 2:
        return np.empty(0, dtype=np.float32)

    similarity = embeddings @ embeddings.T
    row_index, col_index = torch.triu_indices(
        sample_count,
        sample_count,
        offset=1,
    )
    distances = 1.0 - similarity[row_index, col_index]
    # 消除浮点计算产生的-0.000000。
    distances = torch.clamp(distances, min=0.0, max=2.0)
    return distances.numpy()


def compute_cross_pairs(embeddings_a, embeddings_b):
    """计算两个集合之间全部笛卡尔积pair的余弦距离。"""
    if embeddings_a.numel() == 0 or embeddings_b.numel() == 0:
        return np.empty(0, dtype=np.float32)

    similarity = embeddings_a @ embeddings_b.T
    distances = 1.0 - similarity
    distances = torch.clamp(distances, min=0.0, max=2.0)
    return distances.reshape(-1).numpy()


def sample_random_inter_cow_pairs(
    normal_embeddings_by_cow,
    max_pairs=20000,
    seed=42,
):
    """随机抽取不同牛只normal之间的pair。"""
    rng = random.Random(seed)
    cows = sorted(normal_embeddings_by_cow)
    if len(cows) < 2:
        return np.empty(0, dtype=np.float32)

    result = []
    for _ in tqdm(range(max_pairs), desc="随机异牛normal pair"):
        cow_a, cow_b = rng.sample(cows, 2)
        emb_a = normal_embeddings_by_cow[cow_a]
        emb_b = normal_embeddings_by_cow[cow_b]
        index_a = rng.randrange(emb_a.shape[0])
        index_b = rng.randrange(emb_b.shape[0])
        distance = 1.0 - torch.sum(
            emb_a[index_a] * emb_b[index_b]
        ).item()
        result.append(max(0.0, min(2.0, distance)))

    return np.asarray(result, dtype=np.float32)


def print_mean(logger, name, values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        logger.log(f"{name}=nan（没有可计算的样本）")
        return
    logger.log(f"{name}={values.mean():.6f}")


def run_analysis(
    root_dir,
    model_path,
    split_mode,
    report_path,
    max_normal_per_cow=200,
    max_abnormal_per_cow=None,
    batch_size=128,
    random_inter_cow_pairs=20000,
    seed=42,
):
    set_seed(seed)
    fresh_start()
    logger = ReportLogger(report_path)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.log(f"设备={device}")
        logger.log(f"数据集={Path(root_dir).resolve()}")
        logger.log(f"检查点={Path(model_path).resolve()}")
        logger.log(f"seed={seed}")

        split_name = "seen_cows" if split_mode == "seen" else "unseen_cows"
        cows = get_cow_ids(root_dir, split_name)
        logger.log(f"划分中的牛数量={len(cows)}")

        transform = build_eval_transforms()
        model = build_model(model_path, device, logger)

        normal_by_cow = {}
        abnormal_by_cow = {}

        for index, cow_id in enumerate(tqdm(cows, desc="提取embedding"), 1):
            normal_paths = list_images_for_cow(root_dir, cow_id, "normal")
            abnormal_paths = list_images_for_cow(root_dir, cow_id, "abnormal")

            normal_paths = deterministic_sample(
                normal_paths,
                max_normal_per_cow,
                seed,
                cow_id,
                "normal",
            )
            abnormal_paths = deterministic_sample(
                abnormal_paths,
                max_abnormal_per_cow,
                seed,
                cow_id,
                "abnormal",
            )

            if len(normal_paths) < 2:
                logger.log(
                    f"{index}/{len(cows)} {cow_id}: "
                    f"normal={len(normal_paths)}，跳过"
                )
                continue

            normal_embedding, _ = extract_embeddings_for_paths(
                model,
                normal_paths,
                transform,
                device,
                batch_size,
            )
            normal_by_cow[cow_id] = normal_embedding

            if abnormal_paths:
                abnormal_embedding, _ = extract_embeddings_for_paths(
                    model,
                    abnormal_paths,
                    transform,
                    device,
                    batch_size,
                )
                abnormal_by_cow[cow_id] = abnormal_embedding

            logger.log(
                f"{index}/{len(cows)} {cow_id}: "
                f"normal={len(normal_paths)}, "
                f"abnormal={len(abnormal_paths)}"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.log(f"有效normal牛数量={len(normal_by_cow)}")
        logger.log(f"有效abnormal牛数量={len(abnormal_by_cow)}")

        # 1. 同牛normal-normal距离。
        same_all = []
        for cow_id, normal_embedding in tqdm(
            normal_by_cow.items(),
            desc="normal-normal pair",
        ):
            distances = compute_same_pairs(normal_embedding)
            if distances.size:
                same_all.append(distances)

        same_all = (
            np.concatenate(same_all)
            if same_all
            else np.empty(0, dtype=np.float32)
        )

        # 2. 随机不同牛只normal-normal距离。
        random_diff = sample_random_inter_cow_pairs(
            normal_by_cow,
            max_pairs=random_inter_cow_pairs,
            seed=seed,
        )

        # 3. 同一ID文件夹normal-abnormal距离。
        normal_abnormal_all = []
        for cow_id, abnormal_embedding in tqdm(
            abnormal_by_cow.items(),
            desc="normal-abnormal pair",
        ):
            normal_embedding = normal_by_cow.get(cow_id)
            if normal_embedding is None:
                continue
            distances = compute_cross_pairs(
                normal_embedding,
                abnormal_embedding,
            )
            if distances.size:
                normal_abnormal_all.append(distances)

        normal_abnormal_all = (
            np.concatenate(normal_abnormal_all)
            if normal_abnormal_all
            else np.empty(0, dtype=np.float32)
        )

        logger.log("")
        logger.log("=" * 60)
        logger.log("distance mean summary")
        logger.log("=" * 60)
        print_mean(logger, "same_mean", same_all)
        print_mean(logger, "diff_mean", random_diff)
        print_mean(
            logger,
            "normal_abnormal_mean",
            normal_abnormal_all,
        )

    finally:
        logger.close()


if __name__ == "__main__":
    # 原始开放集：
    # root_dir = r"D:\datasets\cow_recognition8.3\openset\openset_ori_8.2"
    # report_path = (
    #     r"D:\code_project\cow_recognition\outputs"
    #     r"\exp_ablation_ResNet50\pair_analysis_ori_with_abnormal.txt"
    # )

    # Hard开放集：
    root_dir = (
        r"D:\datasets\cow_recognition8.3\openset"
        r"\openset_consensus_hard_11to1_v2"
    )

    model_path = (
        r"D:\code_project\cow_recognition\outputs"
        r"\exp_ablation_ResNet50\checkpoints\epoch_10.pth"
    )

    report_path = (
        r"D:\code_project\cow_recognition\outputs"
        r"\exp_ablation_ResNet50\pair_analysis_hard_with_abnormal.txt"
    )

    run_analysis(
        root_dir=root_dir,
        model_path=model_path,
        split_mode="unseen",
        report_path=report_path,
        max_normal_per_cow=200,
        max_abnormal_per_cow=None,
        batch_size=128,
        random_inter_cow_pairs=20000,
        seed=42,
    )
