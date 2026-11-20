# -*- coding: utf-8 -*-
"""
src/sup/models/arcface_model.py

包含：
1. ResNet50 backbone
2. GAP / GeM池化
3. embedding映射头
4. ArcFace分类头

训练：
    emb, logits = model(images, labels)

需要返回目标类别角度：
    emb, logits, theta_y = model(
        images,
        labels,
        return_theta=True,
    )

推理：
    emb = model.forward_emb(images)
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)

# ArcFace分类头继续放在loss文件中
from src.losses.arcface_loss import ArcMarginProduct


# =========================================================
# 1. GeM池化
# =========================================================

class GeM(nn.Module):
    """
    Generalized Mean Pooling。

    p=1时等价于全局平均池化GAP；
    p增大时逐渐接近最大池化。

    输入：
        [B, C, H, W]

    输出：
        [B, C, 1, 1]
    """

    def __init__(
        self,
        p: float = 3.0,
        eps: float = 1e-6,
        learn_p: bool = True,
        p_min: float = 1.0,
        p_max: float = 6.0,
    ):
        super().__init__()

        if p_min <= 0:
            raise ValueError(
                f"p_min必须大于0，当前值为：{p_min}"
            )

        if p_max < p_min:
            raise ValueError(
                f"p_max不能小于p_min，"
                f"当前p_min={p_min}，p_max={p_max}"
            )

        self.eps = float(eps)
        self.p_min = float(p_min)
        self.p_max = float(p_max)

        p_tensor = torch.tensor(
            [float(p)],
            dtype=torch.float32,
        )

        if learn_p:
            self.p = nn.Parameter(p_tensor)
        else:
            self.register_buffer(
                "p",
                p_tensor,
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # 限制p的取值范围，避免训练过程中出现异常
        p = self.p.clamp(
            min=self.p_min,
            max=self.p_max,
        )

        # 避免小于等于0的值参与幂运算
        x = x.clamp(
            min=self.eps
        )

        # GeM:
        # (mean(x^p))^(1/p)
        x = x.pow(p)

        x = F.adaptive_avg_pool2d(
            x,
            output_size=(1, 1),
        )

        x = x.pow(
            1.0 / p
        )

        return x

    def get_p(self) -> float:
        """
        返回当前GeM池化参数p。
        """

        return float(
            self.p
            .detach()
            .clamp(
                min=self.p_min,
                max=self.p_max,
            )
            .cpu()
            .item()
        )


# =========================================================
# 2. ArcFace完整模型
# =========================================================

class ArcFaceCowModel(nn.Module):
    """
    基于ResNet50-ArcFace的奶牛背纹特征编码模型。

    模型结构：

        输入图像
            ↓
        ResNet50 backbone
            ↓
        GAP或GeM池化
            ↓
        embedding head
            ↓
        L2归一化embedding
            ↓
        ArcFace分类头

    训练：
        emb, logits = model(images, labels)

    训练并返回目标类别角度：
        emb, logits, theta_y = model(
            images,
            labels,
            return_theta=True,
        )

    推理：
        emb = model.forward_emb(images)

    不传labels时：
        emb = model(images)
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 128,
        s: float = 30.0,
        m: float = 0.25,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        hidden_dims: Sequence[int] = (1024, 128),
        dropout: float = 0.3,
        use_gem: bool = False,
        gem_p: float = 3.0,
        gem_learn_p: bool = True,
        gem_p_min: float = 1.0,
        gem_p_max: float = 6.0,
    ):
        super().__init__()

        if num_classes <= 0:
            raise ValueError(
                "num_classes必须大于0。"
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim必须大于0。"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout必须位于[0, 1)之间，"
                f"当前值为：{dropout}"
            )

        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.use_gem = bool(use_gem)

        # -------------------------------------------------
        # 使用一个普通nn.Module作为命名容器
        #
        # 它不是第二个模型类，只用于保留旧权重命名：
        # embedder.backbone.*
        # embedder.pool.*
        # embedder.head.*
        # -------------------------------------------------
        self.embedder = nn.Module()

        # -------------------------------------------------
        # ResNet50 backbone
        # -------------------------------------------------
        weights = (
            ResNet50_Weights.IMAGENET1K_V2
            if pretrained
            else None
        )

        resnet = resnet50(
            weights=weights
        )

        # 去掉ResNet50原始avgpool和fc
        self.embedder.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        if freeze_backbone:
            self.freeze_backbone()

        # -------------------------------------------------
        # GAP或GeM池化
        # -------------------------------------------------
        if self.use_gem:
            self.embedder.pool = GeM(
                p=gem_p,
                learn_p=gem_learn_p,
                p_min=gem_p_min,
                p_max=gem_p_max,
            )
        else:
            self.embedder.pool = (
                nn.AdaptiveAvgPool2d(
                    output_size=(1, 1)
                )
            )

        # ResNet50 layer4输出通道数
        backbone_feature_dim = 2048

        # -------------------------------------------------
        # embedding映射头
        # -------------------------------------------------
        head_layers = []
        current_dim = backbone_feature_dim

        for hidden_dim in hidden_dims:

            hidden_dim = int(hidden_dim)

            if hidden_dim <= 0:
                raise ValueError(
                    "hidden_dims中的维度必须大于0。"
                )

            head_layers.extend([
                nn.Linear(
                    current_dim,
                    hidden_dim,
                ),
                nn.BatchNorm1d(
                    hidden_dim
                ),
                nn.ReLU(
                    inplace=True
                ),
                nn.Dropout(
                    p=dropout
                ),
            ])

            current_dim = hidden_dim

        head_layers.extend([
            nn.Linear(
                current_dim,
                embedding_dim,
            ),
            nn.BatchNorm1d(
                embedding_dim
            ),
        ])

        self.embedder.head = nn.Sequential(
            *head_layers
        )

        # -------------------------------------------------
        # ArcFace分类头
        # -------------------------------------------------
        self.arcface = ArcMarginProduct(
            in_features=embedding_dim,
            out_features=num_classes,
            s=s,
            m=m,
        )

    # =====================================================
    # 特征提取
    # =====================================================

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        提取ResNet50最后一层特征图。

        输入：
            [B, 3, H, W]

        输入320×320时，通常输出：
            [B, 2048, 10, 10]
        """

        return self.embedder.backbone(x)

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        完整embedding提取流程。

        返回：
            L2归一化后的embedding
            形状：[B, embedding_dim]
        """

        # ResNet50特征图
        features = self.forward_features(x)

        # GAP或GeM池化
        pooled = self.embedder.pool(
            features
        )

        pooled = pooled.flatten(
            start_dim=1
        )

        # embedding映射
        embedding = self.embedder.head(
            pooled
        )

        # L2归一化
        embedding = F.normalize(
            embedding,
            p=2,
            dim=1,
        )

        return embedding

    # =====================================================
    # 训练与推理
    # =====================================================

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor = None,
        return_theta: bool = False,
    ):
        """
        不传labels：
            返回embedding

        传入labels：
            返回embedding和ArcFace logits

        return_theta=True：
            额外返回目标类别角度theta_y
        """

        embedding = self.encode(x)

        # 推理模式
        if labels is None:
            return embedding

        # 训练模式，返回theta_y
        if return_theta:
            logits, theta_y = self.arcface(
                embedding,
                labels,
                return_theta=True,
            )

            return (
                embedding,
                logits,
                theta_y,
            )

        # 普通训练模式
        logits = self.arcface(
            embedding,
            labels,
            return_theta=False,
        )

        return (
            embedding,
            logits,
        )

    def forward_emb(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        开放集识别和embedding提取接口。
        """

        return self.encode(x)

    # =====================================================
    # Backbone控制
    # =====================================================

    def freeze_backbone(self):
        """
        冻结ResNet50 backbone。
        """

        for parameter in (
            self.embedder.backbone.parameters()
        ):
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        """
        解冻ResNet50 backbone。
        """

        for parameter in (
            self.embedder.backbone.parameters()
        ):
            parameter.requires_grad = True

    # =====================================================
    # GeM参数读取
    # =====================================================

    def get_gem_p(self):
        """
        使用GeM时返回当前p值；
        使用GAP时返回None。
        """

        if isinstance(
            self.embedder.pool,
            GeM,
        ):
            return self.embedder.pool.get_p()

        return None


# =========================================================
# 3. 独立运行测试
# =========================================================

def test_model():
    """
    检查模型能否正常进行训练和推理前向传播。
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = ArcFaceCowModel(
        num_classes=54,
        embedding_dim=128,
        s=30.0,
        m=0.25,
        pretrained=False,
        freeze_backbone=False,
        use_gem=False,
    ).to(device)

    model.train()

    images = torch.randn(
        4,
        3,
        320,
        320,
        device=device,
    )

    labels = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
        device=device,
    )

    embedding, logits, theta_y = model(
        images,
        labels,
        return_theta=True,
    )

    print("=" * 70)
    print("ArcFace模型测试")
    print("=" * 70)

    print(
        f"embedding形状：{embedding.shape}"
    )

    print(
        f"logits形状：{logits.shape}"
    )

    print(
        f"theta_y形状：{theta_y.shape}"
    )

    print(
        "embedding模长：",
        embedding.norm(
            p=2,
            dim=1,
        ),
    )

    print(
        "池化方式：",
        "GeM"
        if model.use_gem
        else "GAP",
    )

    print(
        f"当前GeM p：{model.get_gem_p()}"
    )

    # 推理测试
    model.eval()

    with torch.no_grad():
        inference_embedding = (
            model.forward_emb(images)
        )

    print(
        "推理embedding形状：",
        inference_embedding.shape,
    )

    print("=" * 70)


if __name__ == "__main__":
    test_model()