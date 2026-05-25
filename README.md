# BrainMapNet: Classifying Visual Stimuli from fMRI Activation Maps

# fMRI Brain-State Classifier

CNN-based decoder for cognitive state classification from fMRI-derived 
cortical surface activation maps.

## Overview
- Curated 4,000+ labeled brain surface images from fMRI data
- Trained CNN with convolutional layers, pooling, and dropout (PyTorch)
- Achieved 94% F1-score (5-fold cross-validation)
- Applied gradient-based saliency maps to identify discriminative cortical regions

## Stack
Python · PyTorch · NumPy · scikit-learn

## 项目概述

这个项目实现了从fMRI脑激活图中分类视觉刺激（人脸 vs 房屋）的分类器。

### 数据结构

数据位于 `dataset_combined_vertical` 文件夹下，结构如下：
```
dataset_combined_vertical/
├── pmindo100/
│   └── combined_vertical/
│       ├── brain_cond1_vertical.png  (house)
│       ├── brain_cond2_vertical.png  (house)
│       ├── ...
│       ├── brain_cond6_vertical.png  (house)
│       ├── brain_cond7_vertical.png  (face)
│       ├── ...
│       └── brain_cond12_vertical.png (face)
├── pmindo105/
└── ...
```

- **cond1-6**: house (标签=1)
- **cond7-12**: face (标签=0)
- 每个subject有12张图像

## 安装依赖

```bash
pip install opencv-python
```

其他依赖已在原项目的 `requirements.txt` 中。

## 使用方法

### 1. 训练CNN模型

```bash
cd brainmapnet
python train_cnn.py
```

默认使用ResNet-18预训练模型。可以在脚本中修改配置：
- `architecture`: "resnet18" 或 "mobilenet"
- `pretrained`: True/False
- `freeze_backbone`: True/False (是否冻结backbone)
- `batch_size`: 16
- `learning_rate`: 1e-4

### 2. 训练MLP基线模型

```bash
python train_mlp.py
```

MLP使用top 2000个激活体素作为输入。

### 3. 评估模型

```bash
# 评估CNN模型
python evaluate.py --model cnn --architecture resnet18

# 评估MLP模型
python evaluate.py --model mlp

# 生成Grad-CAM可视化（仅CNN）
python evaluate.py --model cnn --architecture resnet18 --visualize --num_samples 10
```

## 文件结构

```
brainmapnet/
├── dataset.py          # 数据集加载器（扫描subject文件夹）
├── model/
│   ├── fmri_cnn.py    # CNN分类器（ResNet/MobileNet）
│   └── fmri_mlp.py    # MLP基线模型
├── train_cnn.py       # CNN训练脚本
├── train_mlp.py      # MLP训练脚本
├── evaluate.py        # 评估和可视化脚本
├── gradcam.py         # Grad-CAM实现
└── README.md          # 本文件
```

## 数据划分

代码会自动进行**subject-level**划分（不是image-level），避免数据泄漏：
- 每个subject的所有图像都在同一个划分中
- 默认：70% train, 15% val, 15% test
- 使用随机种子确保可重复性

## 模型配置

### CNN模型
- **输入**: 224x224 RGB图像
- **架构**: ResNet-18 或 MobileNet-V3-Small
- **预训练**: ImageNet权重
- **输出**: 2类（face/house）

### MLP模型
- **输入**: Top 2000个激活体素（从图像中提取）
- **架构**: 2000 → 512 → 256 → 2
- **Dropout**: 0.3

## 评估指标

评估脚本会输出：
- **Accuracy**: 准确率
- **F1 Score**: F1分数
- **ROC-AUC**: ROC曲线下面积
- **Confusion Matrix**: 混淆矩阵可视化

## Grad-CAM可视化

Grad-CAM可以可视化模型关注的脑区：
- 仅支持CNN模型
- 生成热力图叠加在原始激活图上
- 显示预测类别和置信度
- 保存到 `gradcam_visualizations/` 目录

## 训练建议

对于~200个subject（~2400张图像）的数据集：

1. **使用预训练模型**: 设置 `pretrained=True`
2. **小批次**: 使用 `batch_size=8-16`
3. **冻结backbone**: 如果过拟合，尝试 `freeze_backbone=True`
4. **早停**: 默认patience=10，可根据需要调整

## 输出文件

- **训练曲线**: `fmri_cnn_training_plot_*.png`
- **混淆矩阵**: `fmri_cnn_confusion_matrix.png` 或 `fmri_mlp_confusion_matrix.png`
- **Grad-CAM可视化**: `gradcam_visualizations/sample_*.png`
- **模型检查点**: `checkpoints/fmri_cnn/` 或 `checkpoints/fmri_mlp/`

## 注意事项

1. 确保 `dataset_combined_vertical` 文件夹在 `project2` 目录下
2. 图像会自动转换为RGB格式（如果是灰度图）
3. 图像会自动调整大小为224x224
4. 使用subject-level划分避免数据泄漏

## 引用

如果使用此代码，请引用相关论文：
- Kamitani & Tong (2005) - Brain decoding
- Selvaraju et al. (2017) - Grad-CAM
- Haxby et al. (2001) - Face/object representations

# EECS442_Project
