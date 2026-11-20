# # src/sup/losses/arcface.py
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class ArcMarginProduct(nn.Module):
#     """
#     输入:
#         embeddings: [B, D]，建议已经 L2 normalize
#         labels:     [B]
#     输出:
#         logits:     [B, num_classes]
#     """
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         s: float = 30.0,
#         m: float = 0.50,
#         easy_margin: bool = False
#     ):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
#         self.easy_margin = easy_margin

#         self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)

#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m

#     def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         # embeddings: [B, D]
#         # normalize weight
#         cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))  # [B, C]
#         sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=1e-7, max=1.0))
#         phi = cosine * self.cos_m - sine * self.sin_m

#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#         else:
#             phi = torch.where(cosine > self.th, phi, cosine - self.mm)

#         one_hot = torch.zeros_like(cosine)
#         one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

#         logits = one_hot * phi + (1.0 - one_hot) * cosine
#         logits *= self.s
#         return logits

#添加L5正则化
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """
    输入:
        embeddings: [B, D]，建议已经 L2 normalize
        labels:     [B]
    输出:
        logits:     [B, num_classes]
        theta_y:    [B]，每个样本与其真类中心的角度（弧度）
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.50,
        easy_margin: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        ##False 代表不使用L5正则化
        return_theta: bool = False
    ):
        # [B, C]
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=1e-7, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.s

        if not return_theta:
            return logits

        # 取出每个样本对应真类的 cosine 和 theta
        cosine_y = cosine.gather(1, labels.view(-1, 1)).squeeze(1)   # [B]
        theta_y = torch.acos(cosine_y)                                # [B], 弧度

        return logits, theta_y