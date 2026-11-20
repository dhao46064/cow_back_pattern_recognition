import os
import sys
import gc
import time
import random
import hashlib
from pathlib import Path

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)


if PROJECT_ROOT not in sys.path:

    sys.path.append(
        PROJECT_ROOT
    )
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt


# =====================================================
# 0. 基础
# =====================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def fresh_start():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    plt.close("all")



def tensor_sha16(t):

    arr=t.detach().cpu().numpy().tobytes()

    return hashlib.sha256(arr).hexdigest()[:16]



# =====================================================
# 1. path
# =====================================================


PROJECT_ROOT=Path(__file__).resolve().parents[1]


if str(PROJECT_ROOT) not in sys.path:

    sys.path.append(
        str(PROJECT_ROOT)
    )


from src.sup.models.proxy_model import ProxyAnchorCowModel

from src.sup.datasets.split_cow import get_cow_ids



# =====================================================
# 2. transform
# =====================================================


def build_eval_transforms(
        img_size=224
):

    return transforms.Compose([

        transforms.Resize(
            (img_size,img_size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]

        )
    ])



# =====================================================
# 3. dataset
# =====================================================


IMG_EXTS={
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}



def list_images_for_cow(
        root_dir,
        cow_id
):

    normal_dir=Path(root_dir)/cow_id/"normal"


    files=[]


    if normal_dir.exists():

        for p in normal_dir.iterdir():

            if p.suffix.lower() in IMG_EXTS:

                files.append(
                    str(p)
                )

    return sorted(files)



class CowTSNEDataset(Dataset):


    def __init__(
            self,
            root_dir,
            cow_ids,
            transform,
            max_per_cow=200,
            seed=42
    ):

        self.transform=transform

        self.samples=[]


        self.id2label={

            cid:i

            for i,cid in enumerate(cow_ids)

        }


        rng=random.Random(seed)



        for cid in cow_ids:


            paths=list_images_for_cow(
                root_dir,
                cid
            )


            if len(paths)>max_per_cow:

                paths=rng.sample(
                    paths,
                    max_per_cow
                )


            for p in paths:

                self.samples.append(

                    (
                        p,
                        self.id2label[cid],
                        cid
                    )

                )


        rng.shuffle(
            self.samples
        )



        print(
            "[INFO] samples:",
            len(self.samples)
        )



    def __len__(self):

        return len(self.samples)



    def __getitem__(self,index):


        path,label,cid=self.samples[index]


        img=Image.open(
            path
        ).convert(
            "RGB"
        )


        if self.transform:

            img=self.transform(img)


        return img,label,cid,path



# =====================================================
# 4. ProxyAnchor checkpoint
# =====================================================


def load_checkpoint(
        model_path
):


    ckpt=torch.load(

        model_path,

        map_location="cpu",

        weights_only=False

    )


    if "model" in ckpt:

        state_dict=ckpt["model"]

    else:

        state_dict=ckpt



    return ckpt,state_dict




def build_model_from_ckpt(
        model_path,
        device
):


    ckpt,state_dict=load_checkpoint(
        model_path
    )


    print("="*60)

    print(
        "[INFO] checkpoint:",
        model_path
    )


    print(
        "[INFO] epoch:",
        ckpt.get(
            "epoch",
            "unknown"
        )
    )


    model=ProxyAnchorCowModel(

        embedding_dim=128,

        pretrained=False

    )


    missing,unexpected=model.load_state_dict(

        state_dict,

        strict=False

    )


    print(
        "missing:",
        len(missing)
    )


    print(
        "unexpected:",
        len(unexpected)
    )


    model=model.to(device)

    model.eval()


    return model



# =====================================================
# 5. embedding
# =====================================================


@torch.no_grad()
def extract_embeddings(
        model,
        loader,
        device
):


    X=[]

    y=[]

    cids=[]

    paths=[]



    for imgs,labels,cid,p in loader:


        imgs=imgs.to(
            device
        )


        emb=model(
            imgs
        )


        emb=torch.nn.functional.normalize(
            emb,
            dim=1
        )



        X.append(
            emb.cpu().numpy()
        )


        y.append(
            labels.numpy()
        )


        cids.extend(
            cid
        )


        paths.extend(
            p
        )



    return (

        np.concatenate(X),

        np.concatenate(y),

        cids,

        paths

    )



# =====================================================
# 6. tsne
# =====================================================


def run_tsne(
        X
):


    if X.shape[1]>50:

        X=PCA(
            50,
            random_state=42
        ).fit_transform(
            X
        )



    tsne=TSNE(

        n_components=2,

        perplexity=30,

        random_state=42,

        init="pca",

        n_iter=1000

    )


    return tsne.fit_transform(
        X
    )



def plot_tsne(
        X2d,
        labels,
        cids,
        save_path
):


    plt.figure(
        figsize=(12,12)
    )


    scatter=plt.scatter(

        X2d[:,0],

        X2d[:,1],

        c=labels,

        cmap="tab20",

        s=12

    )


    for cid in sorted(set(cids)):


        idx=[

            i

            for i,x in enumerate(cids)

            if x==cid

        ]


        center=X2d[idx].mean(
            axis=0
        )


        plt.text(

            center[0],

            center[1],

            str(cid),

            fontsize=8

        )


    plt.colorbar(
        scatter
    )


    plt.title(
        "ProxyAnchor t-SNE"
    )


    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=300
    )


    plt.show()



# =====================================================
# main
# =====================================================


def main():

    import gc

    gc.collect()

    torch.cuda.empty_cache()

    torch.cuda.ipc_collect()

    root_dir=r"D:\code_project\triplet_code\data\dataset_OBB"


    model_path=r"D:\code_project\triplet_code\logs\exp003_7.22_proxyanchor_baseline\checkpoints\latest.pth"



    device="cuda"



    cow_ids=get_cow_ids(

        root_dir,

        split="seen_cows"

    )



    dataset=CowTSNEDataset(

        root_dir,

        cow_ids,

        build_eval_transforms(),

        max_per_cow=50

    )



    loader=DataLoader(

        dataset,

        batch_size=16,

        shuffle=False,

        num_workers=4,

        pin_memory=True

    )



    model=build_model_from_ckpt(

        model_path,

        device

    )



    X,y,cids,paths=extract_embeddings(

        model,

        loader,

        device

    )


    print(
        "embedding:",
        X.shape
    )


    X2d=run_tsne(
        X
    )


    plot_tsne(

        X2d,

        y,

        cids,

        "proxyanchor_seen_tsne.png"

    )



if __name__=="__main__":

    main()