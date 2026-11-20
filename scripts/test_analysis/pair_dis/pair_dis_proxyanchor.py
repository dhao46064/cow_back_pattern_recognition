import os
import sys
import gc
import json
import random
from pathlib import Path
from itertools import combinations

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
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt


# =========================================================
# 0. 基础
# =========================================================

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



# =========================================================
# 1. 项目路径
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(
        str(PROJECT_ROOT)
    )



from src.sup.models.proxy_model import ProxyAnchorCowModel

from src.sup.datasets.split_cow import get_cow_ids



# =========================================================
# 2. transform
# =========================================================


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



# =========================================================
# 3. 图片读取
# =========================================================

IMG_EXTS={
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}



def is_image_file(path):

    return path.suffix.lower() in IMG_EXTS



def list_normal_images_for_cow(
        root_dir,
        cow_id
):

    normal_dir = (
        Path(root_dir)
        /
        cow_id
        /
        "normal"
    )


    files=[]


    if normal_dir.exists():

        for p in normal_dir.iterdir():

            if (
                p.is_file()
                and
                is_image_file(p)
            ):

                files.append(
                    str(p)
                )


    return sorted(files)



# =========================================================
# 4. checkpoint
# =========================================================


def load_checkpoint(
        model_path
):

    ckpt=torch.load(

        model_path,

        map_location="cpu",

        weights_only=False

    )


    if isinstance(ckpt,dict):

        if "model" in ckpt:

            state_dict=ckpt["model"]


        elif "state_dict" in ckpt:

            state_dict=ckpt["state_dict"]


        elif "model_state_dict" in ckpt:

            state_dict=ckpt["model_state_dict"]


        else:

            state_dict=ckpt


    else:

        state_dict=ckpt



    return ckpt,state_dict





def build_model_from_ckpt(
        model_path,
        device
):


    ckpt,state_dict = load_checkpoint(
        model_path
    )


    print("="*60)

    print(
        "[INFO] checkpoint:",
        model_path
    )


    print(
        "[INFO] epoch:",
        ckpt.get("epoch","unknown")
        if isinstance(ckpt,dict)
        else "unknown"
    )


    embedding_dim=128



    model=ProxyAnchorCowModel(

        embedding_dim=embedding_dim,

        pretrained=False

    )



    missing,unexpected = model.load_state_dict(

        state_dict,

        strict=False

    )



    print(
        "[INFO] missing:",
        len(missing)
    )


    print(
        "[INFO] unexpected:",
        len(unexpected)
    )


    if len(missing)>0:

        print(
            missing[:10]
        )


    if len(unexpected)>0:

        print(
            unexpected[:10]
        )



    model=model.to(device)

    model.eval()


    return model



# =========================================================
# 5. dataset
# =========================================================


class ImageListDataset(Dataset):


    def __init__(
            self,
            img_paths,
            transform=None
    ):

        self.img_paths=img_paths

        self.transform=transform



    def __len__(self):

        return len(self.img_paths)



    def __getitem__(
            self,
            idx
    ):


        path=self.img_paths[idx]


        img=Image.open(path).convert(
            "RGB"
        )


        if self.transform:

            img=self.transform(img)


        return img,path





@torch.no_grad()
def extract_embeddings(
        model,
        img_paths,
        transform,
        device,
        batch_size=64
):


    dataset=ImageListDataset(

        img_paths,

        transform

    )


    loader=DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=4,

        pin_memory=True

    )



    embeddings=[]

    paths=[]



    for imgs,p in loader:


        imgs=imgs.to(
            device,
            non_blocking=True
        )


        emb=model(imgs)



        emb=F.normalize(
            emb,
            dim=1
        )


        embeddings.append(
            emb.cpu()
        )


        paths.extend(
            list(p)
        )



    return torch.cat(
        embeddings,
        dim=0
    ),paths



# =========================================================
# 6. distance
# =========================================================


def sample_same_pairs(
        emb,
        max_pairs=1000,
        seed=42
):


    rng=random.Random(seed)


    n=emb.shape[0]


    pairs=list(
        combinations(
            range(n),
            2
        )
    )


    if len(pairs)>max_pairs:

        pairs=rng.sample(
            pairs,
            max_pairs
        )


    dists=[]


    for i,j in pairs:

        d=1-torch.sum(
            emb[i]*emb[j]
        ).item()


        dists.append(d)


    return dists





def sample_diff_pairs(
        emb_by_cow,
        max_pairs=10000,
        seed=42
):


    rng=random.Random(seed)


    cows=list(
        emb_by_cow.keys()
    )


    dists=[]


    while len(dists)<max_pairs:


        c1,c2=rng.sample(
            cows,
            2
        )


        e1=emb_by_cow[c1]

        e2=emb_by_cow[c2]


        i=rng.randrange(
            e1.shape[0]
        )

        j=rng.randrange(
            e2.shape[0]
        )


        d=1-torch.sum(
            e1[i]*e2[j]
        ).item()


        dists.append(d)



    return dists




def summarize(
        name,
        arr
):


    arr=np.asarray(
        arr,
        dtype=np.float32
    )


    print("\n",name)

    print(
        "count:",
        len(arr)
    )

    print(
        "mean:",
        arr.mean()
    )

    print(
        "std:",
        arr.std()
    )

    print(
        "min:",
        arr.min()
    )

    print(
        "max:",
        arr.max()
    )



# =========================================================
# 7. 主分析
# =========================================================


def run_pair_analysis(
        save_path,
        root_dir,
        model_path,
        split_mode="unseen",
        max_per_cow=150,
        batch_size=64,
        seed=42
):


    set_seed(seed)

    fresh_start()


    device="cuda" if torch.cuda.is_available() else "cpu"



    if split_mode=="seen":

        cow_ids=get_cow_ids(
            root_dir,
            split="seen_cows"
        )


    else:

        cow_ids=get_cow_ids(
            root_dir,
            split="unseen_cows"
        )



    print(
        "[INFO] cows:",
        len(cow_ids)
    )



    model=build_model_from_ckpt(
        model_path,
        device
    )



    transform=build_eval_transforms()



    emb_by_cow={}



    for cid in cow_ids:


        paths=list_normal_images_for_cow(
            root_dir,
            cid
        )


        if len(paths)<2:
            continue



        if len(paths)>max_per_cow:

            paths=random.sample(
                paths,
                max_per_cow
            )



        emb,_=extract_embeddings(

            model,

            paths,

            transform,

            device,

            batch_size

        )


        emb_by_cow[cid]=emb



        print(
            cid,
            emb.shape
        )



    same=[]


    for cid,emb in emb_by_cow.items():

        same.extend(
            sample_same_pairs(
                emb
            )
        )



    diff=sample_diff_pairs(
        emb_by_cow
    )



    summarize(
        "same",
        same
    )


    summarize(
        "diff",
        diff
    )


    print(
        "ratio:",
        np.mean(diff)/np.mean(same)
    )



    plt.figure(
        figsize=(8,6)
    )


    plt.hist(
        same,
        bins=50,
        alpha=0.6,
        label="same"
    )


    plt.hist(
        diff,
        bins=50,
        alpha=0.6,
        label="diff"
    )


    plt.legend()

    plt.xlabel(
        "cosine distance"
    )


    plt.savefig(
        save_path,
        dpi=300
    )


    plt.close()



    result={

        "same_mean":float(np.mean(same)),

        "diff_mean":float(np.mean(diff)),

        "ratio":float(
            np.mean(diff)/np.mean(same)
        )

    }


    json_path=save_path.replace(
        ".png",
        ".json"
    )


    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )



    print(
        "[INFO] saved:",
        json_path
    )




# =========================================================
# main
# =========================================================


def main():


    root_dir=r"D:\code_project\triplet_code\data\dataset_OBB"


    model_path=r"D:\code_project\triplet_code\logs\exp003_7.22_proxyanchor_baseline\checkpoints\epoch_150.pth"


    save_path=str(
        PROJECT_ROOT/
        "proxyanchor_unseen_pair_hist.png"
    )



    run_pair_analysis(

        save_path,

        root_dir,

        model_path,

        split_mode="seen",

        max_per_cow=150,

        batch_size=64

    )



if __name__=="__main__":

    main()