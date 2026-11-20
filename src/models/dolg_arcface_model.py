"""面向奶牛背纹个体识别的 DOLG-ArcFace 模型。

建议放置位置：
    src/sup/models/dolg_arcface_model.py

该模型保持现有 ArcFaceTrainer 所需接口：
    forward(x, labels, return_theta=False)
    forward_emb(x)
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

from src.losses.arcface_loss import ArcMarginProduct
from src.models.dolg_blocks import (
    GeM,
    LocalSelfAttention,
    MultiAtrousModule,
    OrthogonalFusion,
)


class DOLGArcFaceCowModel(nn.Module):
    """ResNet50 + DOLG 局部/全局正交融合 + ArcFace。"""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 128,
        local_dim: int = 512,
        fusion_hidden_dim: int = 512,
        s: float = 30.0,
        m: float = 0.25,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        gem_p: float = 3.0,
        gem_learn_p: bool = True,
        dilation_rates: Sequence[int] = (1, 2, 3),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes 必须大于 0")
        if embedding_dim <= 0 or local_dim <= 0 or fusion_hidden_dim <= 0:
            raise ValueError("embedding_dim、local_dim、fusion_hidden_dim 必须大于 0")

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        # 保留分层输出：layer3 用于局部分支，layer4 用于全局分支。
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1   # 256 channels
        self.layer2 = backbone.layer2   # 512 channels
        self.layer3 = backbone.layer3   # 1024 channels, C4
        self.layer4 = backbone.layer4   # 2048 channels, C5

        if freeze_backbone:
            for module in (
                self.stem,
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = False

        # 局部分支：C4 高分辨率特征 -> 多尺度局部特征 -> 空间重要性建模。
        self.multi_atrous = MultiAtrousModule(
            in_channels=1024,
            out_channels=local_dim,
            dilation_rates=dilation_rates,
        )
        self.local_attention = LocalSelfAttention(channels=local_dim)

        # 全局分支：C5 -> GeM -> 与局部分支统一到 local_dim 维。
        self.global_pool = GeM(
            p=gem_p,
            learn_p=gem_learn_p,
            p_min=1.0,
            p_max=6.0,
        )
        self.global_projection = nn.Sequential(
            nn.Linear(2048, local_dim, bias=False),
            nn.BatchNorm1d(local_dim),
        )

        self.orthogonal_fusion = OrthogonalFusion(eps=1e-6)

        # 正交融合后为 2 * local_dim。
        self.embedding_head = nn.Sequential(
            nn.Linear(2 * local_dim, fusion_hidden_dim, bias=False),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

        self.arcface = ArcMarginProduct(
            in_features=embedding_dim,
            out_features=num_classes,
            s=s,
            m=m,
        )

        self.embedding_dim = int(embedding_dim)
        self.local_dim = int(local_dim)
        self._initialize_new_layers()

    def _initialize_new_layers(self) -> None:
        """只初始化新增层，不覆盖 torchvision 的预训练 Backbone。"""
        new_modules = (
            self.multi_atrous,
            self.local_attention,
            self.global_projection,
            self.embedding_head,
        )

        for root in new_modules:
            for module in root.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        module.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def extract_backbone_features(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 C4(layer3) 和 C5(layer4) 特征。"""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c4 = self.layer3(x)
        c5 = self.layer4(c4)
        return c4, c5

    def extract_embedding(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        c4, c5 = self.extract_backbone_features(x)

        local_multi_scale = self.multi_atrous(c4)
        local_feature, attention_map = self.local_attention(local_multi_scale)

        global_feature = self.global_pool(c5).flatten(1)
        global_feature = self.global_projection(global_feature)

        fused_feature = self.orthogonal_fusion(
            local_feature=local_feature,
            global_feature=global_feature,
        )

        embedding = self.embedding_head(fused_feature)
        embedding = F.normalize(embedding, p=2, dim=1)

        if return_attention:
            return embedding, attention_map
        return embedding

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_theta: bool = False,
    ):
        embedding = self.extract_embedding(x, return_attention=False)

        if labels is None:
            return embedding

        if return_theta:
            logits, theta_y = self.arcface(
                embedding,
                labels,
                return_theta=True,
            )
            return embedding, logits, theta_y

        logits = self.arcface(
            embedding,
            labels,
            return_theta=False,
        )
        return embedding, logits

    def forward_emb(self, x: torch.Tensor) -> torch.Tensor:
        """供现有验证、开放集测试和特征提取代码调用。"""
        return self.extract_embedding(x, return_attention=False)

    @torch.no_grad()
    def forward_emb_with_attention(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """分析阶段使用：同时返回 embedding 与局部注意力图。"""
        return self.extract_embedding(x, return_attention=True)

    def get_model_info(self) -> Dict[str, float]:
        """返回便于写入日志的模型状态。"""
        return {
            "embedding_dim": float(self.embedding_dim),
            "local_dim": float(self.local_dim),
            "gem_p": self.global_pool.current_p(),
        }
