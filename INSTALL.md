# 安装说明

## 必需的依赖

```bash
pip install imageio scikit-learn opencv-python
```

## 可选依赖（用于Grad-CAM可视化）

如果你需要使用Grad-CAM可视化功能，需要安装 `opencv-python`：

```bash
pip install opencv-python
```

**注意**: 如果不安装 `opencv-python`，评估脚本仍然可以运行，只是不能使用 `--visualize` 选项。

## 完整安装

```bash
# 基础依赖
pip install imageio scikit-learn

# 如果需要Grad-CAM可视化
pip install opencv-python

# 或者一次性安装所有
pip install imageio scikit-learn opencv-python
```

## 验证安装

运行以下命令验证安装：

```bash
python -c "import imageio; import sklearn; print('Basic dependencies OK')"
python -c "import cv2; print('OpenCV OK')"  # 可选
```

## Colab使用

在Colab notebook中，第一个单元格应该包含：

```python
!pip install imageio scikit-learn opencv-python
```

