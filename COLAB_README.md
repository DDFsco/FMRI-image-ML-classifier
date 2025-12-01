# Colab Notebooks 使用说明

我已经为你创建了三个Jupyter notebook文件，可以直接在Google Colab上使用：

## 创建的文件

1. **train_cnn.ipynb** - CNN模型训练（ResNet-18/MobileNet）
2. **train_mlp.ipynb** - MLP基线模型训练
3. **evaluate.ipynb** - 模型评估和可视化

## 使用步骤

### 1. 上传数据到Colab

有两种方式：

#### 方式A: 上传到Google Drive
```python
# 在notebook的第一个单元格中取消注释：
from google.colab import drive
drive.mount('/content/drive')

# 然后设置数据路径：
DATA_ROOT = "/content/drive/MyDrive/dataset_combined_vertical"
```

#### 方式B: 直接上传到Colab
1. 在Colab中点击左侧文件夹图标
2. 上传 `dataset_combined_vertical` 文件夹
3. 设置路径：
```python
DATA_ROOT = "/content/dataset_combined_vertical"
```

### 2. 打开Notebook

1. 在Google Colab中，点击"文件" -> "上传notebook"
2. 选择 `train_cnn.ipynb`（或其他notebook）
3. 或者直接在Colab中创建新notebook，然后复制代码

### 3. 修改配置

在每个notebook中，找到配置单元格并修改：

```python
# 数据路径
DATA_ROOT = "/content/dataset_combined_vertical"  # 修改为你的路径

# 训练配置（train_cnn.ipynb）
architecture = "resnet18"  # 或 "mobilenet"
batch_size = 16
learning_rate = 1e-4
patience = 10
```

### 4. 运行Notebook

1. 点击"运行时" -> "更改运行时类型" -> 选择"GPU"（推荐）
2. 按顺序运行所有单元格（Runtime -> Run all）
3. 或者逐个运行单元格

## Notebook结构

每个notebook包含以下部分：

1. **安装依赖** - 安装必要的Python包
2. **数据路径设置** - 设置数据位置
3. **导入库** - 导入所有必要的库
4. **数据集代码** - 数据集加载类
5. **模型代码** - 模型定义
6. **训练/评估代码** - 主训练或评估逻辑

## 注意事项

1. **GPU使用**: Colab提供免费GPU，建议使用GPU加速训练
2. **数据路径**: 确保 `DATA_ROOT` 指向正确的数据文件夹
3. **检查点**: 训练后的模型会保存在 `/content/checkpoints/` 目录
4. **下载结果**: 训练图和模型可以下载到本地

## 快速开始

1. 上传 `train_cnn.ipynb` 到Colab
2. 上传数据文件夹
3. 修改 `DATA_ROOT` 路径
4. 运行所有单元格
5. 等待训练完成

## 故障排除

- **找不到数据**: 检查 `DATA_ROOT` 路径是否正确
- **内存不足**: 减小 `batch_size`
- **CUDA错误**: 确保运行时类型设置为GPU
- **导入错误**: 确保所有依赖都已安装

## 输出文件

训练完成后，你会得到：
- 训练曲线图（保存在notebook目录）
- 模型检查点（保存在checkpoints目录）
- 训练日志（在notebook输出中）

## 下一步

训练完成后，使用 `evaluate.ipynb` 来：
- 评估模型性能
- 生成混淆矩阵
- 创建Grad-CAM可视化

