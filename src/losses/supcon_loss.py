# -*- coding: utf-8 -*-
"""
Supervised Contrastive Loss

输入:
    features: [B, D]
    labels:   [B]

要求:
    同一个 batch 中，每个类别至少出现 2 次。
    因此训练时需要配合 PKSampler，并保证 K >= 2。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        eps: float = 1e-12,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(
                f"temperature必须大于0，当前为{temperature}"
            )

        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features:
            shape = [batch_size, feature_dim]

        labels:
            shape = [batch_size]

        Returns
        -------
        loss:
            标量SupCon损失
        """

        if features.ndim != 2:
            raise ValueError(
                f"features应为[B, D]，当前shape={features.shape}"
            )

        if labels.ndim != 1:
            labels = labels.view(-1)

        if features.size(0) != labels.size(0):
            raise ValueError(
                "features与labels的batch维不一致："
                f"{features.size(0)} vs {labels.size(0)}"
            )

        batch_size = features.size(0)

        if batch_size <= 1:
            raise ValueError(
                "SupConLoss要求batch_size至少为2"
            )

        # 统一使用余弦相似度
        features = F.normalize(
            features,
            p=2,
            dim=1,
        )

        labels = labels.long()

        # [B, B]
        logits = torch.matmul(
            features,
            features.T,
        )

        logits = logits / self.temperature

        # 数值稳定处理
        logits = logits - logits.max(
            dim=1,
            keepdim=True,
        ).values.detach()

        device = features.device

        # 排除自己和自己
        self_mask = torch.eye(
            batch_size,
            dtype=torch.bool,
            device=device,
        )

        valid_pair_mask = ~self_mask

        # 标签相同且不是自身，即为正样本
        positive_mask = (
            labels.view(-1, 1)
            == labels.view(1, -1)
        )

        positive_mask = (
            positive_mask
            & valid_pair_mask
        )

        # softmax分母中排除自身
        exp_logits = torch.exp(logits)
        exp_logits = (
            exp_logits
            * valid_pair_mask.float()
        )

        log_denominator = torch.log(
            exp_logits.sum(
                dim=1,
                keepdim=True,
            )
            + self.eps
        )

        log_prob = logits - log_denominator

        positive_count = positive_mask.sum(
            dim=1
        )

        # 防止某些anchor没有正样本
        valid_anchor_mask = positive_count > 0

        if not valid_anchor_mask.any():
            raise RuntimeError(
                "当前batch中没有任何正样本对。"
                "请确认使用PKSampler并保证K >= 2。"
            )

        mean_log_prob_positive = (
            positive_mask.float()
            * log_prob
        ).sum(dim=1)

        mean_log_prob_positive = (
            mean_log_prob_positive
            / positive_count.clamp_min(1).float()
        )

        loss = -mean_log_prob_positive[
            valid_anchor_mask
        ].mean()

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"SupConLoss出现非有限值：{loss.item()}"
            )

        return loss