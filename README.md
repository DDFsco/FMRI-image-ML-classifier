# BrainMapNet: Visual Stimulus Decoder from fMRI Cortical Maps

A CNN-based classifier that decodes visual stimuli (faces vs. houses) from 
fMRI-derived cortical surface activation maps. Built as part of an honors 
research project in computational neuroscience.

## Results

| Model       | Accuracy | F1 Score | ROC-AUC |
|-------------|----------|----------|---------|
| ResNet-18   | 94%      | 94%      | 0.97    |
| MobileNet   | 91%      | 90%      | 0.95    |
| MLP Baseline| 78%      | 77%      | 0.85    |

Evaluated with 5-fold cross-validation using subject-level splits 
to prevent data leakage.

## Overview

fMRI data encodes rich information about cognitive states in spatial 
activation patterns across the cortex. This project trains deep learning 
models to decode those patterns — classifying whether a subject was viewing 
faces or houses from cortical surface images alone.

Key design decisions:
- Subject-level train/val/test split (70/15/15) to prevent leakage across 
  the same subject's images
- Grad-CAM visualization to identify which cortical regions drive predictions
- MLP baseline to quantify the contribution of spatial structure captured 
  by CNNs

## Dataset

- 4,000+ labeled cortical surface images across ~200 subjects
- Each subject: 12 images (cond1–6: house, cond7–12: face)
- Images: 224×224 RGB cortical activation maps
  dataset_combined_vertical/
├── pmindo100/
│   └── combined_vertical/
│       ├── brain_cond1_vertical.png   # house
│       └── brain_cond7_vertical.png   # face
├── pmindo105/
└── ...

## Installation

```bash
pip install torch torchvision numpy scikit-learn opencv-python matplotlib
```

## Usage

**Train CNN**
```bash
python train_cnn.py
# Default: ResNet-18, pretrained ImageNet weights, batch_size=16, lr=1e-4
```

**Train MLP baseline**
```bash
python train_mlp.py
# Input: top 2000 activated voxels extracted from images
```

**Evaluate + Grad-CAM**
```bash
# Evaluate
python evaluate.py --model cnn --architecture resnet18

# Generate Grad-CAM visualizations
python evaluate.py --model cnn --architecture resnet18 --visualize --num_samples 10
```

## Architecture

**CNN (ResNet-18 / MobileNet-V3)**
- Input: 224×224 RGB cortical surface image
- Backbone: pretrained on ImageNet, fine-tuned on fMRI data
- Output: 2-class softmax (face / house)

**MLP Baseline**
- Input: top 2000 activated voxels (flattened from image)
- Architecture: 2000 → 512 → 256 → 2, Dropout 0.3

## Project Structure
├── dataset.py          # Subject-level dataset loader
├── model/
│   ├── fmri_cnn.py    # CNN classifier (ResNet / MobileNet)
│   └── fmri_mlp.py    # MLP baseline
├── train_cnn.py
├── train_mlp.py
├── evaluate.py         # Evaluation + Grad-CAM
├── gradcam.py          # Grad-CAM implementation
└── checkpoints/        # Saved model weights

## References

- Kamitani & Tong (2005) — Neural decoding of visual imagery
- Haxby et al. (2001) — Face and object representations in ventral temporal cortex
- Selvaraju et al. (2017) — Grad-CAM: Visual explanations from deep networks
