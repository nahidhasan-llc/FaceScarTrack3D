"""
STEP 1 — Dataset Download & Preparation
========================================
Downloads the IEEE DataPort Synthetic Facial Injury Dataset and prepares it
for training. Also handles augmentation to expand the dataset.

Dataset: "Synthetic Healthy and Injured Facial Image Dataset"
URL: https://ieee-dataport.org/documents/synthetic-healthy-and-injured-facial-image-dataset-paired-and-unpaired-learning-tasks

Folder structure after running this script:
    data/
      raw/
        images/        ← original face images (injured)
        masks/         ← binary burn masks (255=burn, 0=normal)
      processed/
        train/
          images/
          masks/
        val/
          images/
          masks/
        test/
          images/
          masks/
"""

import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
RAW_IMAGES_DIR = "data/raw/images"   # put your downloaded images here
RAW_MASKS_DIR  = "data/raw/masks"    # put your downloaded masks here
PROCESSED_DIR  = "data/processed"
IMG_SIZE       = 512
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
TEST_RATIO     = 0.15
SEED           = 42

random.seed(SEED)
np.random.seed(SEED)


# ── 1. Download instructions (manual step) ───────────────────────────────────
def print_download_instructions():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         DATASET DOWNLOAD INSTRUCTIONS                    ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  1. Go to:                                               ║
    ║     https://ieee-dataport.org/documents/                 ║
    ║     synthetic-healthy-and-injured-facial-image-dataset   ║
    ║                                                          ║
    ║  2. Create a free IEEE DataPort account                  ║
    ║                                                          ║
    ║  3. Download the dataset archive (~500MB)                ║
    ║                                                          ║
    ║  4. Extract and place:                                   ║
    ║     - Injured face images → data/raw/images/             ║
    ║     - Segmentation masks  → data/raw/masks/              ║
    ║                                                          ║
    ║  Mask format: PNG, binary (255=burn area, 0=normal)      ║
    ║  Image/mask filenames must match (e.g. 001.jpg / 001.png)║
    ║                                                          ║
    ║  ALTERNATIVE: Use Roboflow burn dataset (no login needed)║
    ║  Run: python step1_prepare_dataset.py --roboflow         ║
    ╚══════════════════════════════════════════════════════════╝
    """)


# ── 2. Roboflow alternative (no account needed for public datasets) ───────────
def download_roboflow_dataset():
    """
    Downloads burn dataset from Roboflow Universe as fallback.
    Contains general skin burns — not face-specific, but usable for pretraining.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        os.system("pip install roboflow -q")
        from roboflow import Roboflow

    print("Downloading burn dataset from Roboflow Universe...")
    # Public dataset — no API key needed for download
    rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")  # free at roboflow.com
    project = rf.workspace("aibuildersclub").project("skin-burns-4yoo2")
    dataset = project.version(1).download("coco-segmentation", location="data/raw/roboflow")
    print(f"Downloaded to: data/raw/roboflow")
    return "data/raw/roboflow"


# ── 3. Validate dataset ───────────────────────────────────────────────────────
def validate_dataset(images_dir, masks_dir):
    """Check images and masks are paired and valid."""
    images_dir = Path(images_dir)
    masks_dir  = Path(masks_dir)

    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])
    mask_files = sorted([
        f for f in masks_dir.iterdir()
        if f.suffix.lower() in [".png", ".jpg"]
    ])

    print(f"Found {len(image_files)} images, {len(mask_files)} masks")

    # Check pairing by stem name
    image_stems = {f.stem for f in image_files}
    mask_stems  = {f.stem for f in mask_files}
    paired = image_stems & mask_stems
    unpaired = (image_stems | mask_stems) - paired

    if unpaired:
        print(f"WARNING: {len(unpaired)} unpaired files — they will be skipped")

    paired_images = [f for f in image_files if f.stem in paired]
    paired_masks  = [masks_dir / (f.stem + ".png") for f in paired_images]

    print(f"Valid pairs: {len(paired_images)}")
    return paired_images, paired_masks


# ── 4. Preprocess + augment ───────────────────────────────────────────────────
def preprocess_image_mask(img_path, mask_path, size=512):
    """Load, resize, and ensure mask is binary."""
    img  = cv2.imread(str(img_path))
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    img  = cv2.resize(img,  (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

    # Binarize mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return img, mask


def augment_pair(img, mask):
    """
    Returns list of augmented (img, mask) pairs.
    Geometric augmentations applied identically to both.
    Color augmentations only to image.
    """
    pairs = [(img, mask)]  # original

    # Horizontal flip
    pairs.append((
        cv2.flip(img, 1),
        cv2.flip(mask, 1)
    ))

    # Rotate ±15°
    for angle in [-15, 15]:
        M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1.0)
        pairs.append((
            cv2.warpAffine(img,  M, (IMG_SIZE, IMG_SIZE)),
            cv2.warpAffine(mask, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_NEAREST)
        ))

    # Brightness/contrast jitter (image only)
    alpha = random.uniform(0.8, 1.2)   # contrast
    beta  = random.randint(-20, 20)    # brightness
    bright_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    pairs.append((bright_img, mask))

    return pairs


# ── 5. Split and save ─────────────────────────────────────────────────────────
def split_and_save(image_paths, mask_paths, output_dir, augment=True):
    output_dir = Path(output_dir)

    # Shuffle
    combined = list(zip(image_paths, mask_paths))
    random.shuffle(combined)
    image_paths, mask_paths = zip(*combined)

    n = len(image_paths)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": (image_paths[:n_train],         mask_paths[:n_train]),
        "val":   (image_paths[n_train:n_train+n_val], mask_paths[n_train:n_train+n_val]),
        "test":  (image_paths[n_train+n_val:],   mask_paths[n_train+n_val:]),
    }

    for split_name, (imgs, msks) in splits.items():
        img_out  = output_dir / split_name / "images"
        mask_out = output_dir / split_name / "masks"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        idx = 0
        for img_p, msk_p in tqdm(zip(imgs, msks), desc=f"Processing {split_name}", total=len(imgs)):
            img, mask = preprocess_image_mask(img_p, msk_p, IMG_SIZE)

            pairs = augment_pair(img, mask) if (augment and split_name == "train") else [(img, mask)]

            for aug_img, aug_mask in pairs:
                cv2.imwrite(str(img_out / f"{idx:05d}.jpg"), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(mask_out / f"{idx:05d}.png"), aug_mask)
                idx += 1

        print(f"  {split_name}: {idx} samples saved")

    print(f"\nDataset prepared at: {output_dir}")


# ── 6. Quick sanity check ─────────────────────────────────────────────────────
def visualize_samples(processed_dir, n=4):
    """Save a grid of sample image+mask pairs for visual check."""
    train_imgs  = sorted(Path(processed_dir, "train", "images").glob("*.jpg"))[:n]
    train_masks = sorted(Path(processed_dir, "train", "masks").glob("*.png"))[:n]

    grid_rows = []
    for img_p, msk_p in zip(train_imgs, train_masks):
        img  = cv2.imread(str(img_p))
        mask = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)

        # Overlay mask in red
        overlay = img.copy()
        overlay[mask > 127] = (overlay[mask > 127] * 0.5 + np.array([0, 0, 128]) * 0.5).astype(np.uint8)

        # Contour on overlay
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        row = np.hstack([img, overlay])
        grid_rows.append(row)

    grid = np.vstack(grid_rows)
    out_path = Path(processed_dir) / "sanity_check.jpg"
    cv2.imwrite(str(out_path), grid)
    print(f"Sanity check saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print_download_instructions()

    if "--roboflow" in sys.argv:
        download_roboflow_dataset()
    else:
        if not Path(RAW_IMAGES_DIR).exists():
            print(f"ERROR: {RAW_IMAGES_DIR} not found.")
            print("Run with --roboflow flag to use Roboflow dataset instead.")
            sys.exit(1)

        image_paths, mask_paths = validate_dataset(RAW_IMAGES_DIR, RAW_MASKS_DIR)
        split_and_save(image_paths, mask_paths, PROCESSED_DIR, augment=True)
        visualize_samples(PROCESSED_DIR)
        print("\n✅ Step 1 complete! Run step2_train.py next.")
