# -*- coding: utf-8 -*-

import pandas as pd

from pathlib import Path
from tqdm import tqdm


OPENSET_ROOT = Path(
    r"D:\datasets\cow_recognition8.3\openset\openset_consensus_hard_11to1_v2"
)


FARM_CSV = Path(
    r"D:\code_project\triplet_code\data\dataset_OBB\farm_distribution\eid_farm_detail.csv"
)


SAVE_TXT = Path(
    r"D:\code_project\cow_recognition\openset_hard_statistics.txt"
)



IMG_EXTS={
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}



def count_images(path):

    if not path.exists():
        return 0

    return len(
        [
            x for x in path.iterdir()
            if x.is_file()
            and x.suffix.lower() in IMG_EXTS
        ]
    )



def load_farm_info():


    df=pd.read_csv(
        FARM_CSV,
        dtype={
            "eid":str
        }
    )


    df["eid"]=(
        df["eid"]
        .astype(str)
    )


    return df





def scan_openset():


    farm_df=load_farm_info()


    farm_dict={}


    split_dict={}


    if "eid" in farm_df.columns:

        for _,r in farm_df.iterrows():

            eid=str(r["eid"])


            split=str(
                r.get(
                    "split",
                    ""
                )
            )


            if split=="unseen_":

                farm_dict[eid]="unknown"

            else:

                name=r.get(
                    "farm_name",
                    None
                )

                if pd.notna(name) and str(name)!="":

                    farm_dict[eid]=str(name)

                else:

                    farm_id=r.get(
                        "farm_id",
                        None
                    )

                    if pd.notna(farm_id):

                        farm_dict[eid]=(
                            "farm_"
                            +
                            str(farm_id)
                        )

                    else:

                        farm_dict[eid]="unknown"



            split_dict[eid]=split



    rows=[]


    cow_dirs=[
        x for x in OPENSET_ROOT.iterdir()
        if x.is_dir()
    ]


    print(
        "开放集牛数量:",
        len(cow_dirs)
    )


    for cow_dir in tqdm(
        cow_dirs
    ):


        eid=cow_dir.name


        normal=count_images(
            cow_dir/"normal"
        )


        abnormal=count_images(
            cow_dir/"abnormal"
        )


        rows.append({

            "eid":eid,

            "farm":
            farm_dict.get(
                eid,
                "unknown"
            ),

            "split":
            split_dict.get(
                eid,
                "unknown"
            ),

            "normal":
            normal,

            "abnormal":
            abnormal

        })


    return pd.DataFrame(rows)





def write_report(df):


    with open(
        SAVE_TXT,
        "w",
        encoding="utf-8"
    ) as f:



        f.write(
            "="*50
            +
            "\n开放集总体统计\n"
            +
            "="*50
            +
            "\n\n"
        )


        cow_num=len(df)


        normal_sum=df.normal.sum()

        abnormal_sum=df.abnormal.sum()


        f.write(
            f"牛数量: {cow_num}\n"
        )


        f.write(
            f"normal图片: {normal_sum}\n"
        )


        f.write(
            f"abnormal图片: {abnormal_sum}\n"
        )


        f.write(
            f"总图片: {normal_sum+abnormal_sum}\n\n"
        )


        f.write(
            f"平均每头牛normal: "
            f"{df.normal.mean():.4f}\n"
        )


        f.write(
            f"平均每头牛abnormal: "
            f"{df.abnormal.mean():.4f}\n"
        )


        f.write(
            f"平均normal/abnormal: "
            f"{(df.normal/df.abnormal.replace(0,1)).mean():.4f}\n"
        )



        # ----------------------
        # 牛场
        # ----------------------

        f.write(
            "\n\n"
            +
            "="*50
            +
            "\n牛场统计\n"
            +
            "="*50
            +
            "\n\n"
        )


        farm_stat=(

            df.groupby(
                "farm"
            )

            .agg(

                cow_num=(
                    "eid",
                    "count"
                ),

                normal_images=(
                    "normal",
                    "sum"
                ),

                abnormal_images=(
                    "abnormal",
                    "sum"
                ),

                avg_normal_per_cow=(
                    "normal",
                    "mean"
                ),

                avg_abnormal_per_cow=(
                    "abnormal",
                    "mean"
                )

            )

            .reset_index()

        )



        f.write(
            farm_stat.to_string(
                index=False
            )
        )



        # ----------------------
        # split
        # ----------------------

        f.write(
            "\n\n"
            +
            "="*50
            +
            "\nSplit统计\n"
            +
            "="*50
            +
            "\n\n"
        )



        split_stat=(

            df.groupby(
                "split"
            )

            .agg(

                cow_num=(
                    "eid",
                    "count"
                ),

                normal_images=(
                    "normal",
                    "sum"
                ),

                abnormal_images=(
                    "abnormal",
                    "sum"
                )

            )

            .reset_index()

        )


        f.write(
            split_stat.to_string(
                index=False
            )
        )


    print(
        "报告保存:",
        SAVE_TXT
    )





def main():


    df=scan_openset()


    write_report(
        df
    )



if __name__=="__main__":

    main()