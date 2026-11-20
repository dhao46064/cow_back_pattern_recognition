# -*- coding: utf-8 -*-

"""
Backbone Ablation Model

Support:

ResNet18
ResNet50
DenseNet121
EfficientNet-B0
Swin-T
HRNet-W32


Output:

[B,C,H,W]

"""

import torch

import torch.nn as nn

import torch.nn.functional as F



from torchvision.models import (

    resnet18,

    resnet50,

    densenet121,

    efficientnet_b0,


    ResNet18_Weights,

    ResNet50_Weights,

    DenseNet121_Weights,

    EfficientNet_B0_Weights,

)



from src.losses.arcface_loss import ArcMarginProduct





# =====================================================
# timm wrapper
# =====================================================


class TimmFeatureWrapper(nn.Module):


    def __init__(self, model):

        super().__init__()

        self.model=model



    def forward(self,x):


        features=self.model(x)


        # 取最后stage

        x=features[-1]


        # timm:

        # B,H,W,C

        # 转成:

        # B,C,H,W


        if x.ndim==4:

            x=x.permute(
                0,
                3,
                1,
                2
            ).contiguous()


        return x





# =====================================================
# backbone factory
# =====================================================



def build_backbone(
    name,
    pretrained=True
):


    name=name.lower()



    # -----------------------------
    # ResNet18
    # -----------------------------


    if name=="resnet18":


        model=resnet18(

            weights=

            ResNet18_Weights.IMAGENET1K_V1

            if pretrained

            else None

        )



        backbone=nn.Sequential(

            model.conv1,

            model.bn1,

            model.relu,

            model.maxpool,


            model.layer1,

            model.layer2,

            model.layer3,

            model.layer4,

        )



        dim=512




    # -----------------------------
    # ResNet50
    # -----------------------------


    elif name=="resnet50":


        model=resnet50(

            weights=

            ResNet50_Weights.IMAGENET1K_V2

            if pretrained

            else None

        )



        backbone=nn.Sequential(

            model.conv1,

            model.bn1,

            model.relu,

            model.maxpool,


            model.layer1,

            model.layer2,

            model.layer3,

            model.layer4,

        )



        dim=2048





    # -----------------------------
    # DenseNet121
    # -----------------------------


    elif name=="densenet121":


        model=densenet121(

            weights=

            DenseNet121_Weights.IMAGENET1K_V1

            if pretrained

            else None

        )


        backbone=nn.Sequential(

            model.features,

            nn.ReLU(inplace=True)

        )


        dim=1024






    # -----------------------------
    # EfficientNet-B0
    # -----------------------------


    elif name=="efficientnet_b0":


        model=efficientnet_b0(

            weights=

            EfficientNet_B0_Weights.IMAGENET1K_V1

            if pretrained

            else None

        )



        backbone=model.features


        dim=1280





    # -----------------------------
    # Swin Transformer
    # -----------------------------


    elif name=="swin_t":


        import timm



        model=timm.create_model(

            "swin_tiny_patch4_window7_224",

            pretrained=pretrained,

            features_only=True

        )


        backbone=TimmFeatureWrapper(

            model

        )


        dim=768






    # -----------------------------
    # HRNet
    # -----------------------------


    elif name=="hrnet_w32":


        import timm



        model=timm.create_model(

            "hrnet_w32",

            pretrained=pretrained,

            features_only=True

        )



        backbone=TimmFeatureWrapper(

            model

        )



        dim=480





    else:


        raise ValueError(

            f"Unknown backbone {name}"

        )




    return backbone,dim







# =====================================================
# ArcFace model
# =====================================================



class ArcFaceCowModel(nn.Module):



    def __init__(

        self,

        num_classes,

        backbone="resnet50",

        embedding_dim=128,

        s=30.0,

        m=0.25,

        pretrained=True,

    ):


        super().__init__()



        self.backbone_name=backbone



        self.backbone,self.feature_dim=(

            build_backbone(

                backbone,

                pretrained

            )

        )




        self.pool=nn.AdaptiveAvgPool2d(

            (1,1)

        )




        self.head=nn.Sequential(

            nn.Linear(

                self.feature_dim,

                1024

            ),


            nn.BatchNorm1d(

                1024

            ),


            nn.ReLU(inplace=True),


            nn.Dropout(0.3),



            nn.Linear(

                1024,

                embedding_dim

            ),



            nn.BatchNorm1d(

                embedding_dim

            )

        )





        self.arcface=ArcMarginProduct(

            in_features=embedding_dim,

            out_features=num_classes,

            s=s,

            m=m

        )







    def encode(self,x):


        x=self.backbone(x)



        # 防止某些模型输出list

        if isinstance(x,list):

            x=x[-1]



        x=self.pool(x)



        x=torch.flatten(

            x,

            1

        )



        x=self.head(x)



        x=F.normalize(

            x,

            p=2,

            dim=1

        )



        return x







    def forward(
    self,
    x,
    labels=None,
    return_theta=False,
):


        emb=self.encode(x)



        if labels is None:

            return emb



        logits=self.arcface(

            emb,

            labels

        )



        return emb,logits







    def forward_emb(self,x):


        return self.encode(x)