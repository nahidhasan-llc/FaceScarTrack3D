"""
STEP 2 — UNet Training
========================
Trains a UNet with ResNet34 encoder (pretrained on ImageNet) on the
prepared facial burn segmentation dataset.

Install:
    pip install torch torchvision segmentation-models-pytorch albumentations tqdm

Usage:
    python step2_train.py
    python step2_train.py --resume checkpoints/best_model.pth   # resume training
"""

import os
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

import segmentation_models_pytorch as smp


# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    "data_dir":       "data/processed",
    "checkpoint_dir": "checkpoints",
    "log_path":       "training_log.json",

    # Model
    "encoder":        "resnet34",        # resnet34 | efficientnet-b4 | mit-b2
    "encoder_weights":"imagenet",
    "img_size":       512,

    # Training
    "epochs":         50,
    "batch_size":     4,                 # reduce to 2 if OOM on GPU
    "lr":             1e-4,
    "weight_decay":   1e-5,
    "num_workers":    4,

    # Early stopping
    "patience":       10,

    # Device
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
}


# ── Dataset ───────────────────────────────────────────────────────────────────
class BurnFaceDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.image_dir = Path(data_dir) / split / "images"
        self.mask_dir  = Path(data_dir) / split / "masks"
        self.transform = transform

        self.image_files = sorted(self.image_dir.glob("*.jpg"))
        self.mask_files  = sorted(self.mask_dir.glob("*.png"))

        assert len(self.image_files) == len(self.mask_files), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.mask_files)} masks"

        print(f"  [{split}] {len(self.image_files)} samples loaded")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img  = cv2.imread(str(self.image_files[idx]))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_files[idx]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            aug  = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        return img, mask.unsqueeze(0)    # (3,H,W), (1,H,W)


def get_transforms(img_size):
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    train_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.HueSaturationValue(p=1),
            A.CLAHE(p=1),
        ], p=0.4),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

    val_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

    return train_tf, val_tf


def get_dataloaders(cfg):
    train_tf, val_tf = get_transforms(cfg["img_size"])

    train_ds = BurnFaceDataset(cfg["data_dir"], "train", train_tf)
    val_ds   = BurnFaceDataset(cfg["data_dir"], "val",   val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(cfg):
    model = smp.Unet(
        encoder_name=cfg["encoder"],
        encoder_weights=cfg["encoder_weights"],
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model.to(cfg["device"])


# ── Loss ──────────────────────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    """BCE + Dice loss. Handles class imbalance well for small burn regions."""
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce      = nn.BCEWithLogitsLoss()
        self.bce_w    = bce_weight
        self.smooth   = smooth

    def dice_loss(self, logits, targets):
        preds  = torch.sigmoid(logits)
        inter  = (preds * targets).sum(dim=(2, 3))
        union  = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice   = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, targets):
        return self.bce_w * self.bce(logits, targets) + (1 - self.bce_w) * self.dice_loss(logits, targets)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(logits, targets, threshold=0.5):
    preds   = (torch.sigmoid(logits) > threshold).float()
    targets = targets.float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    dice = (2 * tp + 1) / (2 * tp + fp + fn + 1)
    iou  = (tp + 1) / (tp + fp + fn + 1)

    return {"dice": dice.item(), "iou": iou.item()}


# ── Train / Val loops ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_dice, total_iou = 0, 0, 0

    for images, masks in tqdm(loader, desc="  Train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        metrics     = compute_metrics(logits, masks)
        total_loss += loss.item()
        total_dice += metrics["dice"]
        total_iou  += metrics["iou"]

    n = len(loader)
    return {"loss": total_loss/n, "dice": total_dice/n, "iou": total_iou/n}


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou = 0, 0, 0

    for images, masks in tqdm(loader, desc="  Val  ", leave=False):
        images, masks = images.to(device), masks.to(device)
        logits  = model(images)
        loss    = criterion(logits, masks)

        metrics     = compute_metrics(logits, masks)
        total_loss += loss.item()
        total_dice += metrics["dice"]
        total_iou  += metrics["iou"]

    n = len(loader)
    return {"loss": total_loss/n, "dice": total_dice/n, "iou": total_iou/n}


# ── Main training loop ────────────────────────────────────────────────────────
def train(cfg, resume_path=None):
    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Face Burn Segmentation — UNet Training")
    print(f"  Device  : {cfg['device']}")
    print(f"  Encoder : {cfg['encoder']} (pretrained: {cfg['encoder_weights']})")
    print(f"  Image   : {cfg['img_size']}×{cfg['img_size']}")
    print(f"  Epochs  : {cfg['epochs']}  |  Batch: {cfg['batch_size']}")
    print(f"{'='*60}\n")

    # Data
    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(cfg)

    # Model
    model     = build_model(cfg)
    criterion = CombinedLoss(bce_weight=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)

    # Resume
    start_epoch = 0
    best_dice   = 0.0
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=cfg["device"])
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt["best_dice"]
        print(f"Resumed from epoch {start_epoch}, best Dice: {best_dice:.4f}")

    history      = []
    no_improve   = 0

    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, cfg["device"])
        val_metrics   = val_epoch(model, val_loader, criterion, cfg["device"])
        scheduler.step()

        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch+1:03d}/{cfg['epochs']}] "
            f"| Train Loss: {train_metrics['loss']:.4f}  Dice: {train_metrics['dice']:.4f}  IoU: {train_metrics['iou']:.4f}"
            f"| Val   Loss: {val_metrics['loss']:.4f}  Dice: {val_metrics['dice']:.4f}  IoU: {val_metrics['iou']:.4f}"
            f"  [{elapsed:.1f}s]"
        )

        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val":   val_metrics,
            "lr":    scheduler.get_last_lr()[0],
        })

        # Save best
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            no_improve = 0
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "best_dice":  best_dice,
                "cfg":        cfg,
            }, f"{cfg['checkpoint_dir']}/best_model.pth")
            print(f"  ✅ New best Dice: {best_dice:.4f} — checkpoint saved")
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"\nEarly stopping after {cfg['patience']} epochs without improvement.")
                break

        # Save latest always
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_dice": best_dice,
            "cfg":       cfg,
        }, f"{cfg['checkpoint_dir']}/latest_model.pth")

        # Save log
        with open(cfg["log_path"], "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best Dice (val): {best_dice:.4f}")
    print(f"  Checkpoint: {cfg['checkpoint_dir']}/best_model.pth")
    print(f"  Run step3_inference.py next")
    print(f"{'='*60}")


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",      type=str, default=None)
    parser.add_argument("--encoder",     type=str, default=CFG["encoder"])
    parser.add_argument("--batch_size",  type=int, default=CFG["batch_size"])
    parser.add_argument("--epochs",      type=int, default=CFG["epochs"])
    args = parser.parse_args()

    CFG["encoder"]    = args.encoder
    CFG["batch_size"] = args.batch_size
    CFG["epochs"]     = args.epochs

    train(CFG, resume_path=args.resume)
