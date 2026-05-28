# ── burn_edge_detection_pytorch.py ──────────────────────────────────────────
# Pipeline: Pretrained UNet → burn segmentation mask → edge overlay
# Install: pip install segmentation-models-pytorch albumentations opencv-python torch torchvision

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# ── 1. Model ─────────────────────────────────────────────────────────────────

def build_model(encoder="resnet34", weights="imagenet"):
    """UNet with pretrained encoder. Change encoder to efficientnet-b4 for better accuracy."""
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=1,              # binary: burn vs non-burn
        activation=None,        # raw logits; we apply sigmoid manually
    )
    return model

# ── 2. Dataset ────────────────────────────────────────────────────────────────

class BurnDataset(Dataset):
    """
    Expects folder structure:
        data/
          images/  *.jpg
          masks/   *.png  (binary: 255=burn, 0=normal)
    """
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask  = (mask > 127).astype(np.float32)   # binarize

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask.unsqueeze(0)            # (C,H,W), (1,H,W)

def get_transforms(img_size=512):
    train_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])
    return train_tf, val_tf

# ── 3. Training loop ──────────────────────────────────────────────────────────

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def dice_loss(preds, targets, smooth=1):
    preds   = torch.sigmoid(preds)
    inter   = (preds * targets).sum(dim=(2,3))
    union   = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    return 1 - ((2 * inter + smooth) / (union + smooth)).mean()

def combined_loss(preds, targets):
    return nn.BCEWithLogitsLoss()(preds, targets) + dice_loss(preds, targets)

# ── 4. Inference + Edge Overlay ───────────────────────────────────────────────

def predict_and_draw_edges(
    model,
    image_path,
    device,
    img_size=512,
    threshold=0.5,
    edge_color=(0, 255, 0),   # green edges; change to (255,0,0) for red
    edge_thickness=2,
):
    """Returns original image with burn-area edges drawn on it."""
    # -- load & preprocess
    orig  = cv2.imread(image_path)
    orig  = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    h, w  = orig.shape[:2]

    tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])
    tensor = tf(image=orig)["image"].unsqueeze(0).to(device)

    # -- inference
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        mask   = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # -- resize mask back to original image size
    mask = cv2.resize(mask, (w, h))
    binary_mask = (mask > threshold).astype(np.uint8) * 255

    # -- morphological cleanup (optional but recommended)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel)

    # -- find contours → draw edges
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    result = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR).copy()
    cv2.drawContours(result, contours, -1, edge_color[::-1], edge_thickness)  # BGR

    return result, binary_mask

# ── 5. Quick-start (no dataset yet → SAM zero-shot fallback) ─────────────────

def zero_shot_with_sam(image_path, edge_color=(0, 255, 0)):
    """
    Use Meta's SAM (Segment Anything) for zero-shot burn detection.
    Install: pip install segment-anything
    Download checkpoint: https://github.com/facebookresearch/segment-anything
    """
    from segment_anything import sam_model_registry, SamPredictor

    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
    predictor = SamPredictor(sam)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # Click-prompt: center of image as a starting point
    h, w = image_rgb.shape[:2]
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])

    masks, _, _ = predi