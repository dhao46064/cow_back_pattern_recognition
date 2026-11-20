import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    """
    embeddings: [B, D] (L2-normalized)
    labels: [B]
    """
    def __init__(self, margin=0.7):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(embeddings, embeddings, p=2)  # [B, B]
        B = labels.size(0)
        lab = labels.view(B, 1)

        is_pos = (lab == lab.t())
        is_neg = (lab != lab.t())

        diag = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        is_pos = is_pos & (~diag)

        dist_pos = dist.clone()
        dist_pos[~is_pos] = -1e9
        hardest_pos, _ = dist_pos.max(dim=1)

        dist_neg = dist.clone()
        dist_neg[~is_neg] = 1e9
        hardest_neg, _ = dist_neg.min(dim=1)

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        valid = hardest_pos > -1e8
        return loss[valid].mean() if valid.any() else loss.mean() * 0.0