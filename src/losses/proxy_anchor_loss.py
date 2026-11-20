import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyAnchorLoss(nn.Module):
    """
    Proxy Anchor Loss

    input:
        embeddings: [B,D]
        labels: [B]

    每个类别一个proxy
    """

    def __init__(
        self,
        num_classes,
        embedding_dim,
        margin=0.1,
        alpha=32
    ):
        super().__init__()

        self.num_classes = num_classes
        self.margin = margin
        self.alpha = alpha

        # proxy参数
        self.proxies = nn.Parameter(
            torch.randn(num_classes, embedding_dim)
        )

        nn.init.kaiming_normal_(self.proxies)


    def forward(self, embeddings, labels):

        # normalize
        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1
        )

        proxies = F.normalize(
            self.proxies,
            p=2,
            dim=1
        )


        # cosine similarity
        sim = embeddings @ proxies.T
        # [B,C]


        labels_one_hot = torch.zeros_like(sim)

        labels_one_hot.scatter_(
            1,
            labels.unsqueeze(1),
            1
        )


        positive_mask = labels_one_hot.bool()
        negative_mask = ~positive_mask


        # positive
        pos_sim = sim[positive_mask]

        pos_loss = torch.log(
            1+
            torch.sum(
                torch.exp(
                    -self.alpha*(pos_sim-self.margin)
                )
            )
        )


        # negative

        neg_sim = sim[negative_mask]


        neg_loss = torch.log(
            1+
            torch.sum(
                torch.exp(
                    self.alpha*(neg_sim+self.margin)
                )
            )
        )


        loss = pos_loss + neg_loss


        return loss