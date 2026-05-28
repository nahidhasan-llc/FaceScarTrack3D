# Face Burn Edge Detection Pipeline

Detects burned areas on facial images and draws precise edge contours using a
UNet segmentation model trained on the IEEE DataPort Synthetic Facial Injury Dataset.

---

## Pipeline Overview

```
[IEEE DataPort Dataset]          [SAM2 Zero-Shot]
        │                               │
        ▼                               ▼
step1_prepare_dataset.py    step2b_sam2_zeroshot.py
        │                         (no training needed)
        ▼
step2_train.py
  UNet (ResNet34 encoder)
  Pretrained on ImageNet
  Fine-tuned on burn masks
        │
        ▼
step3_inference.py
  → Binary burn mask
  → Contour edge extraction
  → Overlay on original image
```

---

## Quick Start

### Option A — With Training (Best Quality)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
#    Go to: https://ieee-dataport.org/documents/synthetic-healthy-and-injured-facial-image-dataset-paired-and-unpaired-learning-tasks
#    Place images → data/raw/images/
#    Place masks  → data/raw/masks/

# 3. Prepare dataset (resize, augment, split 70/15/15)
python step1_prepare_dataset.py

# 4. Train UNet (GPU recommended, ~2-4 hours on RTX 3080)
python step2_train.py

# 5. Run inference
python step3_inference.py --image your_face.jpg --show
```

### Option B — Zero-Shot with SAM2 (No Training, Immediate Results)

```bash
pip install sam2

# Download SAM2 checkpoint (~900MB)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Interactive mode: click on burn area
python step2b_sam2_zeroshot.py --image face.jpg --interactive

# Auto mode: heuristic burn detection
python step2b_sam2_zeroshot.py --image face.jpg --auto

# Box prompt: draw box around burn area
python step2b_sam2_zeroshot.py --image face.jpg --box 100 50 400 350
```

---

## File Descriptions

| File | Purpose |
|------|---------|
| `step1_prepare_dataset.py` | Download, validate, augment, split dataset |
| `step2_train.py`           | Train UNet segmentation model |
| `step2b_sam2_zeroshot.py`  | Zero-shot SAM2 inference (no training) |
| `step3_inference.py`       | Run trained model, draw burn edges |
| `requirements.txt`         | Python dependencies |

---

## Inference Options

```bash
# Single image
python step3_inference.py --image face.jpg

# Batch folder
python step3_inference.py --folder images/ --output results/

# Video
python step3_inference.py --video input.mp4 --output results/

# Interactive ROI selection
python step3_inference.py --image face.jpg --interactive

# Adjust sensitivity (lower = more sensitive)
python step3_inference.py --image face.jpg --threshold 0.35

# Resume training
python step2_train.py --resume checkpoints/latest_model.pth
```

---

## Model Performance (Expected)

| Metric | Expected Range |
|--------|---------------|
| Dice Score | 0.75 – 0.88 |
| IoU | 0.65 – 0.80 |
| Inference time | ~50ms/image (GPU) |

---

## Dataset

**Primary:** IEEE DataPort — Synthetic Healthy and Injured Facial Image Dataset
- Face-specific burn images with pixel-level masks
- Burns, bruises, lacerations, scars
- URL: https://ieee-dataport.org/documents/synthetic-healthy-and-injured-facial-image-dataset-paired-and-unpaired-learning-tasks

**Fallback:** Roboflow Skin Burns (general body burns, no face-specific)
- 1,095 images with bounding boxes
- URL: https://universe.roboflow.com/aibuildersclub/skin-burns-4yoo2

---

## Architecture

```
Input (512×512×3)
      │
 ResNet34 Encoder (ImageNet pretrained)
 [64] → [64] → [128] → [256] → [512]
      │
 UNet Decoder (skip connections)
 [256] → [128] → [64] → [32] → [16]
      │
 Conv 1×1 → sigmoid
      │
 Output Mask (512×512×1)
      │
 Morphological Cleanup (close + open)
      │
 cv2.findContours → Edge overlay
```

---

## Loss Function

```
L = 0.5 × BCE + 0.5 × Dice
```

BCE handles pixel-level accuracy.
Dice handles class imbalance (small burn regions vs large normal area).

---

## Tips

- **GPU strongly recommended** for training. CPU training is ~20× slower.
- If Dice < 0.6 after training: lower the threshold (--threshold 0.35)
- For real clinical images: fine-tune with a few manually labeled examples
- SAM2 interactive mode works well for quick single-image results
