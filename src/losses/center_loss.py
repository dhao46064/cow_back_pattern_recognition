import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """
    Center Loss

    目标:
        拉近 embedding 与对应类别中心

    Lc = 1/2 * sum(||x_i-c_y||^2)

    Args:
        num_classes:
            训练牛数量

        feat_dim:
            embedding维度
    """

    def __init__(
        self,
        num_classes,
        feat_dim,
        device="cuda"
    ):
        super().__init__()

        self.num_classes = num_classes
        self.feat_dim = feat_dim


        self.centers = nn.Parameter(
            torch.randn(
                num_classes,
                feat_dim
            ).to(device)
        )


    def forward(
        self,
        features,
        labels
    ):

        centers_batch = self.centers[
            labels
        ]


        loss = torch.sum(
            (features - centers_batch) ** 2
        )

        loss = loss / (
            2.0 * features.size(0)
        )


        return loss