# -*- coding: utf-8 -*-
"""根据最新逐牛macro结果绘制三联雨云图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


# =========================================================
# 1. 路径与基本参数
# =========================================================

DATA_DIR = Path(
    r"D:\code_project\cow_recognition"
    r"\outputs\cow_macro_distance"
)

PNG_PATH = DATA_DIR / "macro_distance_raincloud.png"
PDF_PATH = DATA_DIR / "macro_distance_raincloud.pdf"

MODEL_ORDER = [
    "r50_arcface",
    "r50_center",
    "r50_supcon",
    "dolg_arcface",
    "dolg_center",
    "dolg_supcon",
]

MODEL_LABELS = {
    "r50_arcface": "R50-AF",
    "r50_center": "R50-AF-Center",
    "r50_supcon": "R50-AF-SupCon",
    "dolg_arcface": "DOLG-AF",
    "dolg_center": "DOLG-AF-Center",
    "dolg_supcon": "DOLG-AF-SupCon",
}

COLORS = {
    "r50_arcface": "#9ECAE1",
    "r50_center": "#6BAED6",
    "r50_supcon": "#3182BD",
    "dolg_arcface": "#FDCC8A",
    "dolg_center": "#FC8D59",
    "dolg_supcon": "#B85C2E",
}

PANELS = [
    {
        "column": "same_normal",
        "title": "(a) Intra-class compactness",
        "ylabel": "Same-normal cosine distance",
        "direction": "Lower is better",
    },
    {
        "column": "hard_na",
        "title": "(b) Hard abnormal separation",
        "ylabel": "Hard normal-abnormal cosine distance",
        "direction": "Higher is better",
    },
    {
        "column": "hard_gap",
        "title": "(c) Absolute separation margin",
        "ylabel": "Hard N-A - Same normal",
        "direction": "Higher is better",
    },
]

SEED = 42
BOOTSTRAP_REPEATS = 10000


# =========================================================
# 2. 读取最新逐牛结果
# =========================================================

def load_per_cow_results():
    frames = []

    for model_key in MODEL_ORDER:
        csv_path = DATA_DIR / f"{model_key}_per_cow.csv"

        if not csv_path.is_file():
            raise FileNotFoundError(
                f"缺少逐牛结果文件：{csv_path}"
            )

        frame = pd.read_csv(
            csv_path,
            dtype={"cow_id": str},
        )

        required_columns = {
            "cow_id",
            "same_normal",
            "hard_na",
            "hard_gap",
        }

        missing = required_columns - set(frame.columns)

        if missing:
            raise KeyError(
                f"{csv_path.name}缺少字段：{sorted(missing)}"
            )

        frame["model_key"] = model_key
        frames.append(frame)

        print(
            f"[LOAD] {model_key}: "
            f"cows={frame['cow_id'].nunique()}"
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# =========================================================
# 3. 牛只级bootstrap 95% CI
# =========================================================

def bootstrap_mean_ci(values, seed):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(
        BOOTSTRAP_REPEATS,
        dtype=np.float64,
    )

    for index in range(BOOTSTRAP_REPEATS):
        sampled = rng.choice(
            values,
            size=len(values),
            replace=True,
        )
        bootstrap_means[index] = sampled.mean()

    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return (
        float(values.mean()),
        float(ci_low),
        float(ci_high),
    )


# =========================================================
# 4. 绘制左半边小提琴
# =========================================================

def draw_half_violin(ax, x_position, values, color):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(np.unique(values)) < 2:
        return

    kde = gaussian_kde(values)

    value_min = values.min()
    value_max = values.max()

    padding = max(
        (value_max - value_min) * 0.08,
        0.003,
    )

    y_grid = np.linspace(
        value_min - padding,
        value_max + padding,
        300,
    )

    density = kde(y_grid)

    if density.max() > 0:
        density = density / density.max()

    # 控制小提琴最大宽度
    density = density * 0.38

    ax.fill_betweenx(
        y_grid,
        x_position - density,
        x_position,
        facecolor=color,
        edgecolor=color,
        linewidth=1.0,
        alpha=0.30,
        zorder=1,
    )


# =========================================================
# 5. 绘制单个panel
# =========================================================

def draw_panel(
    ax,
    frame,
    column,
    title,
    ylabel,
    direction,
    panel_index,
):
    for model_index, model_key in enumerate(MODEL_ORDER):
        color = COLORS[model_key]

        values = (
            frame.loc[
                frame["model_key"] == model_key,
                column,
            ]
            .dropna()
            .astype(float)
            .to_numpy()
        )

        if len(values) == 0:
            raise RuntimeError(
                f"{model_key}的{column}没有有效数据"
            )

        x_position = model_index

        # 左侧：密度分布
        draw_half_violin(
            ax,
            x_position,
            values,
            color,
        )

        # 右侧：每头牛一个散点
        rng = np.random.default_rng(
            SEED + panel_index * 100 + model_index
        )

        jitter = rng.uniform(
            0.04,
            0.20,
            size=len(values),
        )

        ax.scatter(
            np.full(len(values), x_position) + jitter,
            values,
            s=14,
            color=color,
            alpha=0.55,
            edgecolors="none",
            zorder=2,
        )

        # 中间：箱线图
        ax.boxplot(
            [values],
            positions=[x_position],
            widths=0.14,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
            medianprops={
                "color": "#222222",
                "linewidth": 1.4,
            },
            boxprops={
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 1.3,
            },
            whiskerprops={
                "color": color,
                "linewidth": 1.2,
            },
            capprops={
                "color": color,
                "linewidth": 1.2,
            },
            zorder=3,
        )

        # 黑色菱形：macro均值
        # 黑色误差棒：牛只级bootstrap 95% CI
        mean, ci_low, ci_high = bootstrap_mean_ci(
            values,
            seed=SEED + panel_index * 100 + model_index,
        )

        ax.errorbar(
            x_position,
            mean,
            yerr=[
                [mean - ci_low],
                [ci_high - mean],
            ],
            fmt="D",
            markersize=5.5,
            markerfacecolor="#111111",
            markeredgecolor="#111111",
            ecolor="#111111",
            elinewidth=1.5,
            capsize=4,
            zorder=5,
        )

        print(
            f"{column:12s} | {model_key:15s} | "
            f"n={len(values):3d} | "
            f"mean={mean:.6f} | "
            f"95%CI=[{ci_low:.6f}, {ci_high:.6f}]"
        )

    ax.set_title(
        title,
        fontsize=13,
        pad=10,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=10.5,
    )

    ax.set_xticks(
        range(len(MODEL_ORDER))
    )

    ax.set_xticklabels(
        [MODEL_LABELS[key] for key in MODEL_ORDER],
        rotation=25,
        ha="right",
        fontsize=9,
    )

    ax.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.7,
        alpha=0.75,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.02,
        0.97,
        direction,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#666666",
    )


# =========================================================
# 6. 主函数
# =========================================================

def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"数据目录不存在：{DATA_DIR}"
        )

    frame = load_per_cow_results()

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6.2),
    )

    for panel_index, (axis, panel) in enumerate(
        zip(axes, PANELS)
    ):
        draw_panel(
            ax=axis,
            frame=frame,
            column=panel["column"],
            title=panel["title"],
            ylabel=panel["ylabel"],
            direction=panel["direction"],
            panel_index=panel_index,
        )

    figure.suptitle(
        "Cow-level embedding compactness "
        "and hard-abnormal separation",
        fontsize=15,
        y=0.99,
    )

    figure.text(
        0.5,
        0.015,
        "Dots: individual cows; "
        "diamond: cow-macro mean; "
        "error bar: cow-level bootstrap 95% CI",
        ha="center",
        fontsize=10,
        color="#444444",
    )

    figure.subplots_adjust(
        left=0.055,
        right=0.99,
        top=0.89,
        bottom=0.22,
        wspace=0.18,
    )

    figure.savefig(
        PNG_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        PDF_PATH,
        bbox_inches="tight",
    )

    plt.close(figure)

    print()
    print("=" * 80)
    print(f"PNG已保存：{PNG_PATH}")
    print(f"PDF已保存：{PDF_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()