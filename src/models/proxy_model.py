# src/sup/models/proxy_model.py

import torch
import torch.nn as nn

from src.models.triplet_model import CowModel_resnet50


class ProxyAnchorCowModel(nn.Module):

    def __init__(
        self,
        embedding_dim=128,
        pretrained=True,
        freeze_backbone=False
    ):

        super().__init__()


        self.embedder = CowModel_resnet50(
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )


    def forward(self,x):

        emb = self.embedder(x)

        return emb


    def forward_emb(self,x):

        return self.embedder(x)