# 快速开始指南

## 步骤1: 检查数据

确保数据在正确的位置：
```bash
# 应该在 project2/dataset_combined_vertical/ 下
ls ../dataset_combined_vertical/
```

应该看到很多subject文件夹（如 pmindo100, pmindo105 等）。

## 步骤2: 训练CNN模型

```bash
cd brainmapnet
python train_cnn.py
```

训练过程会：
- 自动扫描所有subject文件夹
- 进行subject-level划分（70/15/15）
- 加载cond1-12图像并分配标签
- 保存检查点到 `./checkpoints/fmri_cnn/`
- 生成训练曲线图

## 步骤3: 训练MLP基线（可选）

```bash
python train_mlp.py
```

## 步骤4: 评估模型
CNN:
# 查看特定epoch
python evaluate.py --model cnn --epoch 10

# 基于验证准确率选择最佳
python evaluate.py --model cnn --best_by accuracy

# 使用最新epoch
python evaluate.py --model cnn --use_latest


MLP:
# 查看特定epoch
python evaluate.py --model mlp --epoch 10

# 基于验证准确率选择最佳
python evaluate.py --model mlp --best_by accuracy

# 使用最新epoch
python evaluate.py --model mlp --use_latest

## 修改配置

编辑 `train_cnn.py` 或 `train_mlp.py` 中的配置参数：

```python
# CNN配置
architecture = "resnet18"  # 或 "mobilenet"
pretrained = True
freeze_backbone = False
batch_size = 16
learning_rate = 1e-4

# MLP配置
input_dim = 2000
hidden_dims = [512, 256]
dropout_rate = 0.3
```

## 常见问题

**Q: 找不到数据？**
A: 确保 `dataset_combined_vertical` 文件夹在 `project2` 目录下，与 `brainmapnet` 同级。

**Q: 内存不足？**
A: 减小 `batch_size`（如改为8或4）

**Q: 训练很慢？**
A: 如果有GPU，可以在 `evaluate.py` 中使用 `--device cuda`

**Q: 如何查看训练进度？**
A: 训练过程中会实时显示训练和验证指标，并生成训练曲线图。

