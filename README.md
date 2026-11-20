# 奶牛背纹识别与开放集异常样本清洗

本项目面向奶牛背部图像中的 RFID 身份误匹配问题：利用同一牛只文件夹内多数图像形成的主体身份特征，识别并筛除与主体身份不一致的少数图像，从而提高数据集身份纯度，并降低错误身份关联对体况评分（BCS）稳定性的影响。

项目覆盖牛背分割、OBB 旋转裁剪与对齐、身份特征学习、开放集异常筛查、困难负样本构建、颜色域偏移分析以及统计验证。当前代码以研究复现和论文实验为主要用途。

![项目模型框架](paper_materials/cow_model_paper.svg)

## 研究问题

- 原始数据按牛只 ID 存放，但 RFID 误识别会使其他牛只图像混入当前文件夹。
- 清洗任务将正确属于当前 ID 的图像定义为 `Normal`，将误匹配图像定义为 `Abnormal`。
- 模型优先保证被保留图像的身份纯度，同时尽量减少真实正常图像的误删除。
- 除普通开放集外，项目还构建了由多模型一致性排序得到的困难开放集，用于测试纹理相似异牛样本的拒识能力。

## 方法概览

1. 使用 YOLOv8s-seg 分割牛背区域。
2. 根据分割掩码计算最小外接旋转矩形，完成 OBB 旋转裁剪和方向对齐。
3. 使用 ResNet50 或 DOLG 提取 128 维 L2 归一化身份特征。
4. 使用 ArcFace 作为分类约束，并比较 Center Loss 与 SupCon Loss 的辅助作用。
5. 以每头牛的正常样本特征中心和类内距离建立开放集阈值，筛查异常图像。
6. 使用逐牛配对统计、bootstrap 置信区间和 Wilcoxon 检验解释模型差异。

## 主要实验结论

- 在困难开放集上，最终采用的 `DOLG + ArcFace + SupCon (lambda=0.1)` 达到 98.79% Normal Precision、93.10% Normal Recall 和 95.83% Normal F1。
- Center Loss 主要压缩正常样本的类内距离；SupCon 更明显地扩大困难异常样本与正常特征簇之间的分离裕量。
- DOLG + SupCon 的宏观 Hard gap 为 0.2649，在所比较配置中最高。
- 红色域增强能够减小同牛正常样本的类内波动，但没有进一步提高困难开放集的 Precision、Recall 或 F1，因此最终模型不采用该增强。
- 对逐牛结果的分析显示，Hard gap 改善与 Normal Recall 改善呈显著正相关（`r=0.593`，`p<0.001`，`n=192`）。

完整的项目过程、数据统计、实验结果与讨论见 [背纹总结](docs/%E8%83%8C%E7%BA%B9%E6%80%BB%E7%BB%939.17.docx)。

## 仓库结构

```text
.
├── src/                     # 数据集、模型、损失函数、训练器与工具
├── scripts/
│   ├── train/               # 主实验与消融实验训练脚本
│   ├── test/                # 开放集评价脚本
│   └── test_analysis/       # 距离、统计与可视化分析
├── paper_preparetion/       # 数据集统计脚本
├── paper_materials/         # 精选论文图表、表格与可编辑素材
├── statistics/              # 汇总统计结果
├── docs/                    # 项目总结文档
├── hard_openset.py          # 困难开放集构建
└── hard_openset_ori.py      # 原始开放集相关处理
```

## 环境

推荐环境：

- Python 3.10
- PyTorch 2.3.0
- torchvision 0.18.0
- CUDA 12.1（GPU 训练时）

安装基础依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如需 CUDA 12.1 版本的 PyTorch，可按 PyTorch 官方安装说明选择对应 wheel。

## 数据组织

数据集不随仓库分发。训练脚本通常期望以下结构：

```text
dataset_root/
├── split.json
├── <cow_id_1>/
│   ├── normal/
│   └── abnormal/
└── <cow_id_2>/
    ├── normal/
    └── abnormal/
```

原始脚本中仍保留部分本机默认路径，运行时请优先通过命令行参数覆盖，例如 `--obb_root`、`--output_root`、`--model_path` 等。部分早期 `train_sup_*` 脚本依赖旧版 `src.sup` 目录，仅作为实验演进记录保留；当前主实验请使用 `run_*` 系列脚本。

## 运行示例

训练 DOLG 三组主实验：

```bash
python scripts/train/run_dolg_selected_loss_10ep.py ^
  --obb_root D:\path\to\train_dataset ^
  --output_root D:\path\to\outputs\dolg_selected_loss_10ep
```

训练红色增强概率网格：

```bash
python scripts/train/run_dolg_red_loss_grid_10ep.py ^
  --obb_root D:\path\to\train_dataset ^
  --output_root D:\path\to\outputs\dolg_red_loss_grid_10ep
```

脚本参数和输入文件要求以各文件中的 `argparse` 定义与说明为准。

## 未随仓库上传的内容

以下内容体积较大、包含原始数据或与本机环境绑定，因此未纳入版本控制：

- 原始牛只图像、分割掩码及农场记录
- 训练 checkpoint、模型权重与完整 `outputs/`
- 训练日志、缓存、临时文件和重复备份
- 逐图 embedding、逐图清单和超大中间统计文件

如需公开数据或权重，建议先确认数据授权与隐私要求，再通过 GitHub Releases、对象存储或数据集平台单独发布。

## 当前状态

本仓库是研究项目的整理版本。核心实验结果和分析材料已经保留，但部分历史脚本尚未统一配置接口。欢迎通过 Issue 记录复现问题或后续整理计划。
