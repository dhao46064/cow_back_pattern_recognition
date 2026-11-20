"""DOLG 中可复用的基础模块。

建议放置位置：
    src/sup/models/dolg_blocks.py
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Generalized Mean Pooling。

    p=1 时等价于全局平均池化；p 增大时逐渐接近最大池化。
    """

    def __init__(
        self,
        p: float = 3.0,
        eps: float = 1e-6,
        learn_p: bool = True,
        p_min: float = 1.0,
        p_max: float = 6.0,
    ) -> None:
        super().__init__()
        if p_min <= 0 or p_max <= p_min:
            raise ValueError("要求 0 < p_min < p_max")

        self.eps = float(eps)
        self.p_min = float(p_min)
        self.p_max = float(p_max)

        p_tensor = torch.tensor([float(p)], dtype=torch.float32)
        if learn_p:
            self.p = nn.Parameter(p_tensor)
        else:
            self.register_buffer("p", p_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"GeM 输入应为 [B,C,H,W]，实际为 {tuple(x.shape)}")

        p = self.p.clamp(min=self.p_min, max=self.p_max)
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x.pow(1.0 / p)

    def current_p(self) -> float:
        return float(
            self.p.detach()
            .clamp(min=self.p_min, max=self.p_max)
            .cpu()
            .item()
        )


class ConvBNReLU(nn.Sequential):
    """卷积、BN、ReLU 组合。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class MultiAtrousModule(nn.Module):
    """DOLG 局部分支中的多空洞卷积模块。

    在同一分辨率特征图上使用多个 dilation，模拟多尺度特征金字塔；
    同时加入全局上下文分支，最后通过 1×1 卷积完成通道融合。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 512,
        dilation_rates: Sequence[int] = (1, 2, 3),
    ) -> None:
        super().__init__()

        rates: Tuple[int, ...] = tuple(int(r) for r in dilation_rates)
        if len(rates) == 0:
            raise ValueError("dilation_rates 至少包含一个值")
        if any(r <= 0 for r in rates):
            raise ValueError("dilation rate 必须为正整数")

        # 三个空洞卷积分支 + 一个全局上下文分支。
        num_branches = len(rates) + 1
        branch_channels = max(out_channels // num_branches, 1)

        self.atrous_branches = nn.ModuleList(
            [
                ConvBNReLU(
                    in_channels=in_channels,
                    out_channels=branch_channels,
                    kernel_size=3,
                    padding=rate,
                    dilation=rate,
                )
                for rate in rates
            ]
        )

        # 全局池化后的特征空间仅为 1×1。这里不使用 BN，避免最后一个
        # batch 恰好只有 1 张图时 BatchNorm2d 报错。
        self.global_context = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )

        concat_channels = branch_channels * num_branches
        self.fusion = ConvBNReLU(
            in_channels=concat_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"MultiAtrousModule 输入应为 [B,C,H,W]，实际为 {tuple(x.shape)}"
            )

        height, width = x.shape[-2:]
        features = [branch(x) for branch in self.atrous_branches]

        global_feature = F.adaptive_avg_pool2d(x, output_size=1)
        global_feature = self.global_context(global_feature)
        global_feature = F.interpolate(
            global_feature,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        features.append(global_feature)

        return self.fusion(torch.cat(features, dim=1))


class LocalSelfAttention(nn.Module):
    """DOLG 局部分支中的局部重要性建模模块。

    特征分支：1×1 Conv-BN 后沿通道维进行 L2 归一化。
    权重分支：1×1 Conv + SoftPlus，得到非负空间注意力图。
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.feature_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.attention_transform = nn.Conv2d(
            channels,
            1,
            kernel_size=1,
            bias=True,
        )
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(
                f"LocalSelfAttention 输入应为 [B,C,H,W]，实际为 {tuple(x.shape)}"
            )

        local_feature = self.feature_transform(x)
        local_feature = F.normalize(local_feature, p=2, dim=1)

        attention_map = self.softplus(self.attention_transform(x))
        weighted_local_feature = local_feature * attention_map

        return weighted_local_feature, attention_map


class OrthogonalFusion(nn.Module):
    """DOLG 正交融合模块。

    对每个空间位置的局部向量，去掉其在全局向量方向上的投影，
    再将正交局部成分与全局特征拼接并聚合。

    输入：
        local_feature:  [B, C, H, W]
        global_feature: [B, C]
    输出：
        fused_vector:   [B, 2C]
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        local_feature: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> torch.Tensor:
        if local_feature.ndim != 4:
            raise ValueError(
                "local_feature 应为 [B,C,H,W]，"
                f"实际为 {tuple(local_feature.shape)}"
            )
        if global_feature.ndim != 2:
            raise ValueError(
                "global_feature 应为 [B,C]，"
                f"实际为 {tuple(global_feature.shape)}"
            )

        batch_l, channels_l, _, _ = local_feature.shape
        batch_g, channels_g = global_feature.shape
        if batch_l != batch_g or channels_l != channels_g:
            raise ValueError(
                "局部与全局特征的 batch、通道必须一致："
                f"local={tuple(local_feature.shape)}, "
                f"global={tuple(global_feature.shape)}"
            )

        global_map = global_feature.unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]

        # Proj_fg(fl) = ((fl·fg)/(fg·fg)) fg
        dot_product = (local_feature * global_map).sum(
            dim=1,
            keepdim=True,
        )  # [B,1,H,W]
        global_norm_sq = global_feature.pow(2).sum(
            dim=1,
            keepdim=True,
        ).clamp_min(self.eps)  # [B,1]
        projection_scale = dot_product / global_norm_sq.unsqueeze(-1).unsqueeze(-1)
        projected_local = projection_scale * global_map

        orthogonal_local = local_feature - projected_local
        expanded_global = global_map.expand_as(orthogonal_local)

        fused_map = torch.cat(
            [orthogonal_local, expanded_global],
            dim=1,
        )  # [B,2C,H,W]

        # 局部特征已经经过 SoftPlus 注意力加权，因此这里用全局平均完成聚合。
        return F.adaptive_avg_pool2d(fused_map, output_size=1).flatten(1)
