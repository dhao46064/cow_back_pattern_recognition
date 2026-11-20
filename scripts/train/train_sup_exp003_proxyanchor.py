# import os
# import sys
# import random
# import logging
# import argparse


# import numpy as np

# import torch
# import torch.optim as optim

# from torch.utils.data import DataLoader



# sys.path.append(
#     os.path.abspath(
#         os.path.join(
#             os.path.dirname(__file__),
#             ".."
#         )
#     )
# )


# from src.sup.datasets.split_cow import get_samples


# from src.sup.datasets.dataset_OBB_normal import (
#     CowTrainDatasetNormal,
#     CowValDatasetFromSplit
# )

# from src.sup.sampler.pk_sampler import PKSampler
# from src.sup.models.proxy_model import ProxyAnchorCowModel


# from src.sup.losses.proxy_anchor_loss import ProxyAnchorLoss


# from src.sup.engine.proxyanchor_trainer import ProxyanchorTrainer


# from src.sup.utils.transforms import build_transforms

# from src.sup.utils.common import set_seed



# logging.basicConfig(

# level=logging.INFO,

# format="%(asctime)s %(message)s"

# )


# logger=logging.getLogger(
#     "proxyanchor"
# )



# def main():


#     parser=argparse.ArgumentParser()



#     parser.add_argument(
#         "--obb_root",
#         default=r"D:\code_project\triplet_code\data\dataset_OBB"
#     )


#     parser.add_argument(
#         "--epochs",
#         type=int,
#         default=150
#     )


#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=64
#     )


#     parser.add_argument(
#         "--embedding_dim",
#         type=int,
#         default=128
#     )


#     parser.add_argument(
#         "--lr",
#         type=float,
#         default=3e-4
#     )


#     parser.add_argument(
#         "--wd",
#         type=float,
#         default=1e-4
#     )


#     parser.add_argument(
#         "--device",
#         default="cuda"
#     )
#     parser.add_argument(
#     "--num_workers",
#     type=int,
#     default=4
# )


#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42
#     )


#     parser.add_argument(
#         "--log_dir",
#         default="./logs"
#     )


#     parser.add_argument(
#         "--exp_name",
#         default="exp003_7.22_proxyanchor_baseline"
#     )



#     args=parser.parse_args()



#     set_seed(
#         args.seed
#     )



#     device=args.device



#     train_samples=get_samples(
#         args.obb_root,
#         "seen_train"
#     )


#     val_samples=get_samples(
#         args.obb_root,
#         "seen_val"
#     )



#     # label mapping

#     cow_ids=sorted(
#         list(
#             set(
#                 x["cow_id"]
#                 for x in train_samples
#             )
#         )
#     )


#     id2label={

#         cid:i

#         for i,cid in enumerate(cow_ids)

#     }



#     num_classes=len(id2label)



#     logger.info(
#         f"classes={num_classes}"
#     )



#     train_ds=CowTrainDatasetNormal(

#         train_samples,

#         transform=build_transforms(train = False),

#         id2label=id2label

#     )



#     val_ds=CowValDatasetFromSplit(

#         val_samples,

#         id2label=id2label,

#         transform=build_transforms(train = False)

#     )


#     train_labels = [
#         label
#         for _,label in train_ds.data
#     ]


#     pk_sampler = PKSampler(
#         labels=train_labels,
#         P=16,
#         K=4
#     )


#     train_loader = DataLoader(
#         train_ds,
#         batch_sampler=pk_sampler,
#         num_workers=args.num_workers,
#         pin_memory=True
#     )



#     val_loader=DataLoader(

#         val_ds,

#         batch_size=32,

#         shuffle=False,

#         num_workers=0

#     )




#     model=ProxyAnchorCowModel(

#         embedding_dim=args.embedding_dim,

#         pretrained=True

#     )



#     proxy_loss=ProxyAnchorLoss(

#         num_classes=num_classes,

#         embedding_dim=args.embedding_dim,

#         margin=0.1,

#         alpha=32

#     )



#     optimizer=optim.AdamW(

#         list(model.parameters())

#         +

#         list(proxy_loss.parameters()),

#         lr=args.lr,

#         weight_decay=args.wd

#     )




#     trainer=ProxyanchorTrainer(

#         model=model,

#         proxy_loss=proxy_loss,

#         train_loader=train_loader,

#         val_loader=val_loader,

#         optimizer=optimizer,

#         device=device,

#         log_dir=args.log_dir,

#         exp_name=args.exp_name

#     )



#     trainer.fit(

#         args.epochs

#     )




# if __name__=="__main__":

#     main()
import os
import sys
import random
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader


# =========================================================
# 项目路径
# =========================================================
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


from src.sup.datasets.split_cow import get_samples

from src.sup.datasets.dataset_OBB_normal import (
    CowTrainDatasetNormal,
    CowValDatasetFromSplit
)

from src.sup.sampler.pk_sampler import PKSampler
from src.sup.models.proxy_model import ProxyAnchorCowModel
from src.sup.losses.proxy_anchor_loss import ProxyAnchorLoss
from src.sup.engine.proxyanchor_trainer import ProxyanchorTrainer
from src.sup.utils.transforms import build_transforms
from src.sup.utils.common import set_seed


# =========================================================
# 日志配置
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)

logger = logging.getLogger("proxyanchor")


# =========================================================
# 将 optimizer 状态移动到指定设备
# =========================================================
def move_optimizer_to_device(optimizer, device):
    """
    checkpoint 使用 CPU 加载后，将 AdamW 的动量状态移动到 GPU。
    """
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


# =========================================================
# 查找第150轮的 checkpoint
# =========================================================
def find_resume_checkpoint(exp_dir):
    """
    自动按照优先级查找续训权重。
    """

    checkpoint_dir = Path(exp_dir) / "checkpoints"

    candidates = [
         checkpoint_dir / "epoch_150.pth",

    ]

    for path in candidates:
        if path.exists():
            logger.info(f"找到续训 checkpoint：{path}")
            return path

    available_files = []

    if checkpoint_dir.exists():
        available_files = sorted(
            list(checkpoint_dir.glob("*.pth"))
            + list(checkpoint_dir.glob("*.pt"))
        )

    available_text = "\n".join(
        str(path)
        for path in available_files
    )

    raise FileNotFoundError(
        "\n没有找到可用于续训的 checkpoint。\n"
        f"检查目录：{checkpoint_dir}\n"
        f"当前目录中的权重文件：\n{available_text}"
    )


# =========================================================
# 加载 checkpoint
# =========================================================
def load_checkpoint(
    checkpoint_path,
    model,
    proxy_loss,
    optimizer,
    device
):
    """
    恢复：
    1. backbone / embedding模型参数
    2. ProxyAnchor可学习Proxy参数
    3. AdamW优化器状态
    4. 当前epoch
    5. history和trainer状态
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint不存在：{checkpoint_path}"
        )

    logger.info("=" * 80)
    logger.info(f"正在加载 checkpoint：{checkpoint_path}")

    # 为了兼容不同PyTorch版本，先加载到CPU
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "当前 checkpoint 不是完整训练字典，"
            "无法恢复 optimizer、proxy_loss 和 epoch。"
        )

    logger.info(
        f"checkpoint keys：{list(checkpoint.keys())}"
    )

    # -----------------------------------------------------
    # 1. 加载模型参数
    # -----------------------------------------------------
    if "model" in checkpoint:
        model_state = checkpoint["model"]

    elif "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        model_state = checkpoint["state_dict"]

    else:
        raise KeyError(
            "checkpoint中没有找到模型参数。\n"
            "需要存在以下字段之一："
            "model、model_state_dict、state_dict"
        )

    model.load_state_dict(
        model_state,
        strict=True
    )

    logger.info("模型参数恢复成功。")

    # -----------------------------------------------------
    # 2. 加载ProxyAnchor的可学习Proxy参数
    # -----------------------------------------------------
    proxy_state = None

    if "proxy_loss" in checkpoint:
        proxy_state = checkpoint["proxy_loss"]

    elif "proxy_loss_state_dict" in checkpoint:
        proxy_state = checkpoint["proxy_loss_state_dict"]

    elif "criterion" in checkpoint:
        proxy_state = checkpoint["criterion"]

    elif "criterion_state_dict" in checkpoint:
        proxy_state = checkpoint["criterion_state_dict"]

    if proxy_state is not None:
        proxy_loss.load_state_dict(
            proxy_state,
            strict=True
        )

        logger.info("ProxyAnchor Proxy参数恢复成功。")

    else:
        logger.warning(
            "checkpoint中没有找到proxy_loss参数。"
        )
        logger.warning(
            "模型backbone可以恢复，但Proxy将重新初始化，"
            "这不属于完全续训。"
        )

    # -----------------------------------------------------
    # 3. 加载优化器
    # -----------------------------------------------------
    optimizer_state = None

    if "optimizer" in checkpoint:
        optimizer_state = checkpoint["optimizer"]

    elif "optimizer_state_dict" in checkpoint:
        optimizer_state = checkpoint["optimizer_state_dict"]

    if optimizer_state is not None:
        optimizer.load_state_dict(
            optimizer_state
        )

        move_optimizer_to_device(
            optimizer,
            device
        )

        logger.info("AdamW优化器状态恢复成功。")

    else:
        logger.warning(
            "checkpoint中没有optimizer状态，"
            "将使用新的AdamW状态继续训练。"
        )

    # -----------------------------------------------------
    # 4. 获取起始epoch
    # -----------------------------------------------------
    saved_epoch = int(
        checkpoint.get(
            "epoch",
            150
        )
    )

    start_epoch = saved_epoch + 1

    logger.info(
        f"checkpoint epoch = {saved_epoch}"
    )

    logger.info(
        f"续训将从 epoch {start_epoch} 开始"
    )

    logger.info("=" * 80)

    return checkpoint, start_epoch


# =========================================================
# 恢复Trainer内部状态
# =========================================================
def restore_trainer_state(
    trainer,
    checkpoint
):
    """
    恢复history、最佳指标等状态。
    """

    if hasattr(
        trainer,
        "load_state_from_ckpt"
    ):
        trainer.load_state_from_ckpt(
            checkpoint
        )

        logger.info(
            "已通过 trainer.load_state_from_ckpt 恢复训练状态。"
        )

        return

    # trainer没有专门的加载函数时，手动恢复
    if "history" in checkpoint:
        trainer.history = checkpoint["history"]

        logger.info(
            f"history恢复成功："
            f"train={len(trainer.history.get('train', []))}，"
            f"val={len(trainer.history.get('val', []))}"
        )

    restore_attributes = [
        "best_val_same_mean",
        "best_val_cls_acc",
        "best_metric",
        "early_counter"
    ]

    for attr_name in restore_attributes:
        if (
            attr_name in checkpoint
            and hasattr(trainer, attr_name)
        ):
            setattr(
                trainer,
                attr_name,
                checkpoint[attr_name]
            )

            logger.info(
                f"恢复trainer状态："
                f"{attr_name}={checkpoint[attr_name]}"
            )


# =========================================================
# 主函数
# =========================================================
def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--obb_root",
        default=r"D:\code_project\triplet_code\data\dataset_OBB"
    )


    # 总训练轮数，不是再训练300轮
    # 将从151训练到300
    parser.add_argument(
        "--epochs",
        type=int,
        default=300
    )


    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )


    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128
    )


    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4
    )


    parser.add_argument(
        "--wd",
        type=float,
        default=1e-4
    )


    parser.add_argument(
        "--device",
        default="cuda"
    )


    parser.add_argument(
        "--num_workers",
        type=int,
        default=4
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )


    parser.add_argument(
        "--log_dir",
        default=r"D:\code_project\triplet_code\logs"
    )


    parser.add_argument(
        "--exp_name",
        default="exp003_7.22_proxyanchor_baseline"
    )


    # 可以手动指定checkpoint；
    # 留空时自动寻找latest.pth或epoch_150.pth
    parser.add_argument(
        "--resume",
        default=""
    )


    args = parser.parse_args()


    # =====================================================
    # 固定随机种子
    # =====================================================
    set_seed(
        args.seed
    )

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )


    # =====================================================
    # 设备
    # =====================================================
    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        logger.warning(
            "CUDA不可用，自动切换到CPU。"
        )

        device = "cpu"

    else:
        device = args.device


    logger.info(
        f"device={device}"
    )


    # =====================================================
    # 获取训练集和验证集
    # =====================================================
    train_samples = get_samples(
        args.obb_root,
        "seen_train"
    )


    val_samples = get_samples(
        args.obb_root,
        "seen_val"
    )


    # =====================================================
    # label mapping
    # =====================================================
    cow_ids = sorted(
        list(
            set(
                x["cow_id"]
                for x in train_samples
            )
        )
    )


    id2label = {
        cid: i
        for i, cid in enumerate(cow_ids)
    }


    num_classes = len(
        id2label
    )


    logger.info(
        f"classes={num_classes}"
    )

    logger.info(
        f"train samples={len(train_samples)}"
    )

    logger.info(
        f"val samples={len(val_samples)}"
    )


    # =====================================================
    # Dataset
    # =====================================================
    train_ds = CowTrainDatasetNormal(
        train_samples,
        transform=build_transforms(
            train=False
        ),
        id2label=id2label
    )


    val_ds = CowValDatasetFromSplit(
        val_samples,
        id2label=id2label,
        transform=build_transforms(
            train=False
        )
    )


    # =====================================================
    # PKSampler
    # =====================================================
    train_labels = [
        label
        for _, label in train_ds.data
    ]


    pk_sampler = PKSampler(
        labels=train_labels,
        P=16,
        K=4
    )


    train_loader = DataLoader(
        train_ds,
        batch_sampler=pk_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )


    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )


    # =====================================================
    # 模型
    # =====================================================
    model = ProxyAnchorCowModel(
        embedding_dim=args.embedding_dim,
        pretrained=True
    ).to(device)


    # =====================================================
    # ProxyAnchor Loss
    # =====================================================
    proxy_loss = ProxyAnchorLoss(
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        margin=0.1,
        alpha=32
    ).to(device)


    # =====================================================
    # 优化器
    # =====================================================
    optimizer = optim.AdamW(
        list(model.parameters())
        +
        list(proxy_loss.parameters()),
        lr=args.lr,
        weight_decay=args.wd
    )


    # =====================================================
    # 自动寻找第150轮checkpoint
    # =====================================================
    exp_dir = (
        Path(args.log_dir)
        /
        args.exp_name
    )


    if args.resume:
        resume_path = Path(
            args.resume
        )

    else:
        resume_path = find_resume_checkpoint(
            exp_dir
        )


    # =====================================================
    # 恢复第150轮训练状态
    # =====================================================
    checkpoint, start_epoch = load_checkpoint(
        checkpoint_path=resume_path,
        model=model,
        proxy_loss=proxy_loss,
        optimizer=optimizer,
        device=device
    )


    # 防止错误加载其他epoch
    if start_epoch != 151:
        logger.warning(
            f"当前checkpoint将从epoch {start_epoch}开始，"
            f"并不是预期的epoch 151。"
        )


    # =====================================================
    # Trainer
    # =====================================================
    trainer = ProxyanchorTrainer(
        model=model,
        proxy_loss=proxy_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        log_dir=args.log_dir,
        exp_name=args.exp_name
    )


    # 恢复history和最佳指标
    restore_trainer_state(
        trainer,
        checkpoint
    )


    # =====================================================
    # 从151训练到300
    # =====================================================
    trainer.fit(
        epochs=args.epochs,
        start_epoch=start_epoch
    )


if __name__ == "__main__":
    main()