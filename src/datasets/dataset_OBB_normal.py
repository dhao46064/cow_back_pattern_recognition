# -*- coding: utf-8 -*-

from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


class CowTrainDatasetNormal(Dataset):
    """
    seen_train训练数据集。

    处理顺序：
        读取PIL RGB原图
        -> 在线红色增强
        -> 原有transform
        -> 返回Tensor和label
    """

    def __init__(
        self,
        samples: List[Dict],
        transform=None,
        id2label: Optional[Dict[str, int]] = None,
        red_augment=None,
    ):
        self.samples = samples
        self.transform = transform
        self.red_augment = red_augment

        if id2label is None:
            cow_ids = sorted(
                set(str(x["cow_id"]) for x in samples)
            )
            self.id2label = {
                cow_id: index
                for index, cow_id in enumerate(cow_ids)
            }
        else:
            self.id2label = {
                str(cow_id): int(label)
                for cow_id, label in id2label.items()
            }

        self.data = []

        for sample in samples:
            cow_id = str(sample["cow_id"])

            if cow_id not in self.id2label:
                continue

            self.data.append(
                (
                    str(sample["path"]),
                    self.id2label[cow_id],
                )
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        if self.red_augment is not None:
            image = self.red_augment(
                image=image,
                image_path=image_path,
            )

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_red_augmentation_statistics(
        self,
        reset: bool = False,
    ):
        if self.red_augment is None:
            return None

        if not hasattr(self.red_augment, "get_statistics"):
            return None

        return self.red_augment.get_statistics(
            reset=reset
        )


class CowValDatasetFromSplit(Dataset):
    """
    seen_val验证数据集。

    验证集不执行红色增强。
    """

    def __init__(
        self,
        samples: List[Dict],
        id2label: Dict[str, int],
        transform=None,
    ):
        self.samples = samples
        self.transform = transform
        self.id2label = {
            str(cow_id): int(label)
            for cow_id, label in id2label.items()
        }

        self.data = []

        for sample in samples:
            cow_id = str(sample["cow_id"])

            if cow_id not in self.id2label:
                continue

            self.data.append(
                (
                    str(sample["path"]),
                    self.id2label[cow_id],
                )
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
