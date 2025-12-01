# Colab Setup Guide

由于notebook文件会很长，我建议你按照以下步骤在Colab中创建notebook：

## 方法1: 直接复制代码到Colab

### 1. 创建新的Colab notebook

### 2. 按顺序添加以下单元格：

#### Cell 1: 安装依赖
```python
!pip install imageio scikit-learn opencv-python
```

#### Cell 2: 设置数据路径
```python
# Mount Google Drive (如果数据在Drive中)
from google.colab import drive
drive.mount('/content/drive')

# 或者直接上传数据到Colab
# 更新这个路径到你的数据位置
DATA_ROOT = "/content/dataset_combined_vertical"  # 修改为你的数据路径
```

#### Cell 3: 导入库和设置
```python
import os
import random
import itertools
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, MobileNet_V3_Small_Weights
from torch.utils.data import Dataset, DataLoader
import numpy as np
from imageio.v2 import imread
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.nn.functional import softmax

# Set random seed
SEED = 445
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

#### Cell 4-N: 复制以下文件的内容到单独的单元格
- `dataset.py` - 数据集类
- `model/fmri_cnn.py` - CNN模型
- `train_common.py` 中的函数（从项目根目录）
- `utils.py` 中的函数（从项目根目录）
- `train_cnn.py` 中的main函数内容

## 方法2: 使用GitHub

1. 将代码上传到GitHub
2. 在Colab中克隆仓库：
```python
!git clone https://github.com/yourusername/brainmapnet.git
!cd brainmapnet && pip install -r requirements.txt
```

## 方法3: 直接上传文件

1. 在Colab中上传所有Python文件
2. 修改导入路径
3. 运行训练脚本

## 快速开始模板

我已经创建了 `train_cnn.ipynb` 的基础结构，你可以：
1. 打开这个notebook
2. 添加剩余的代码单元格
3. 或者直接复制 `dataset.py`, `model/fmri_cnn.py` 等文件的内容到新的单元格

## 注意事项

- 确保数据路径正确
- 如果使用GPU，Colab会自动检测
- 检查点会保存在 `/content/checkpoints/` 目录
- 训练图会保存在当前目录

