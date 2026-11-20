# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import resnet50, ResNet50_Weights


# class GeM(nn.Module):
#     def __init__(self, p=3.0, eps=1e-6, learn_p=True):
#         super().__init__()
#         if learn_p:
#             self.p = nn.Parameter(torch.ones(1) * p)
#         else:
#             self.p = float(p)
#         self.eps = eps

#     def forward(self, x):
#         return F.adaptive_avg_pool2d(
#             x.clamp(min=self.eps).pow(self.p),
#             (1, 1)
#         ).pow(1.0 / self.p)
# class CowModel_resnet50(nn.Module):
#     """
#     Output: L2-normalized embedding [B, D]
#     """
#     def __init__(
#         self,
#         embedding_dim: int = 128,
#         pretrained: bool = True,
#         freeze_backbone: bool = False,
#         hidden_dims=(1024, 128),
#         dropout: float = 0.3,
#     ):
#         super().__init__()

#         weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
#         backbone = resnet50(weights=weights)

#         self.backbone = nn.Sequential(
#             backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
#             backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
#         )

#         if freeze_backbone:
#             for p in self.backbone.parameters():
#                 p.requires_grad = False

#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         # self.pool = GeM()
        
#         feature_dim = 2048

#         layers = []
#         in_dim = feature_dim
#         for h in hidden_dims:
#             layers += [
#                 nn.Linear(in_dim, h),
#                 nn.BatchNorm1d(h),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(dropout),
#             ]
#             in_dim = h

#         layers += [
#             nn.Linear(in_dim, embedding_dim),
#             nn.BatchNorm1d(embedding_dim),
#         ]
#         self.head = nn.Sequential(*layers)

#         self.embedding_dim = embedding_dim

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         feat = self.backbone(x)
#         pooled = self.pool(feat).flatten(1)
#         emb = self.head(pooled)
#         emb = F.normalize(emb, p=2, dim=1)
#         return emb


# class TripletCowModelWithCE(nn.Module):
#     """
#     Training: return (emb, logits). Validation: use emb only.
#     """
#     def __init__(
#         self,
#         num_train_classes: int,
#         embedding_dim: int = 512,
#         pretrained: bool = True,
#         freeze_backbone: bool = False,
#     ):
#         super().__init__()
#         self.embedder = CowModel_resnet50(
#             embedding_dim=embedding_dim,
#             pretrained=pretrained,
#             freeze_backbone=freeze_backbone,
#         )
#         self.classifier = nn.Linear(embedding_dim, num_train_classes)

#     def forward(self, x: torch.Tensor):
#         emb = self.embedder(x)
#         logits = self.classifier(emb)
#         return emb, logits

#     def forward_emb(self, x: torch.Tensor):
#         return self.embedder(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class GeM(nn.Module):
    """
    Generalized Mean Pooling

    p=1时等价于GAP；
    p增大时逐渐接近最大池化。
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

        self.eps = eps
        self.p_min = p_min
        self.p_max = p_max

        if learn_p:
            self.p = nn.Parameter(
                torch.tensor([float(p)], dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "p",
                torch.tensor([float(p)], dtype=torch.float32)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 避免p接近0、变成负数或过大
        p = self.p.clamp(
            min=self.p_min,
            max=self.p_max,
        )

        # ResNet最后经过ReLU，理论上本身非负；
        # 再clamp一次保证幂运算稳定
        x = x.clamp(min=self.eps)

        # 先p次方
        x = x.pow(p)

        # 再进行全局平均池化
        x = F.adaptive_avg_pool2d(
            x,
            output_size=(1, 1),
        )

        # 最后开p次方根
        x = x.pow(1.0 / p)

        return x


class CowModel_resnet50(nn.Module):
    """
    输出:
        L2归一化后的embedding，形状为[B, embedding_dim]
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        hidden_dims=(1024, 128),
        dropout: float = 0.3,
        use_gem: bool =False,
        gem_p: float = 3.0,
        gem_learn_p: bool = True,
    ):
        super().__init__()

        weights = (
            ResNet50_Weights.IMAGENET1K_V2
            if pretrained
            else None
        )

        backbone = resnet50(weights=weights)

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        # 可以通过use_gem控制使用GeM还是GAP
        if use_gem:
            self.pool = GeM(
                p=gem_p,
                learn_p=gem_learn_p,
                p_min=1.0,
                p_max=6.0,
            )
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        feature_dim = 2048

        layers = []
        in_dim = feature_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.extend([
            nn.Linear(in_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        ])

        self.head = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim
        self.use_gem = use_gem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResNet50最后一层特征图
        # 输入320时通常为[B, 2048, 10, 10]
        feat = self.backbone(x)

        # GAP或GeM池化
        pooled = self.pool(feat).flatten(1)

        # 映射为embedding
        emb = self.head(pooled)

        # 用于Triplet、余弦距离等度量学习
        emb = F.normalize(
            emb,
            p=2,
            dim=1,
        )

        return emb


class TripletCowModelWithCE(nn.Module):
    """
    训练输出:
        emb:    [B, embedding_dim]
        logits: [B, num_train_classes]

    验证和测试阶段主要使用emb。
    """

    def __init__(
        self,
        num_train_classes: int,
        embedding_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_gem: bool = True,
        gem_p: float = 3.0,
        gem_learn_p: bool = True,
    ):
        super().__init__()

        self.embedder = CowModel_resnet50(
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            use_gem=use_gem,
            gem_p=gem_p,
            gem_learn_p=gem_learn_p,
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_train_classes,
        )

    def forward(self, x: torch.Tensor):
        emb = self.embedder(x)
        logits = self.classifier(emb)

        return emb, logits

    def forward_emb(self, x: torch.Tensor):
        return self.embedder(x)

    def get_gem_p(self):
        """
        返回当前GeM的p值。
        使用GAP时返回None。
        """
        if isinstance(self.embedder.pool, GeM):
            return float(
                self.embedder.pool.p
                .detach()
                .clamp(
                    self.embedder.pool.p_min,
                    self.embedder.pool.p_max,
                )
                .cpu()
                .item()
            )

        return None
