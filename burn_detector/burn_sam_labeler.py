"""
Burn Area Labeler — Segment Anything Model (SAM)
=================================================
Click on a burn patch → SAM segments it cleanly.
Press keys to save mask, undo, or move to next image.

SETUP (run once):
    pip install segment-anything opencv-python numpy matplotlib tifffile imagecodecs

    Download SAM checkpoint (ViT-B, ~375MB):
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    Put it in the same folder as this script.

USAGE:
    # Single image
    python burn_sam_labeler.py --image pat1day28A_front.png

    # All images in a folder
    python burn_sam_labeler.py --batch scans/ --output burn_masks/

CONTROLS (in the popup window):
    Left click          → add a point ON the burn area  (green dot)
    Right click         → add a point NOT burn area     (red dot — tells SAM what to exclude)
    SPACE               → segment using current points
    S                   → save current mask + move to next image
    U                   → undo last point
    R                   → reset all points
    Q / ESC             → quit
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")        # interactive window — needs display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
import argparse
import os
import sys
import tifffile

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("ERROR: segment_anything not installed.")
    print("Run: pip install segment-anything")
    sys.exit(1)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"   # path to downloaded checkpoint
SAM_MODEL_TYPE = "vit_b"                  # matches the checkpoint above


# ─────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────
def load_image(path):
    """Load .tif or standard image → RGB uint8 numpy array."""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tif", ".tiff"]:
        raw = tifffile.imread(path)
        if raw.ndim == 3 and raw.shape[2] >= 3:
            raw = raw[:, :, :3]
        if raw.dtype != np.uint8:
            raw = ((raw - raw.min()) / (raw.max() - raw.min()) * 255).astype(np.uint8)
        return raw  # tifffile gives RGB already
    else:
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot load: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────
# SAM LABELER CLASS
# ─────────────────────────────────────────────
class BurnLabeler:
    def __init__(self, predictor, image_rgb, image_name, output_dir):
        self.predictor   = predictor
        self.image_rgb   = image_rgb
        self.image_name  = image_name
        self.output_dir  = output_dir
        self.h, self.w   = image_rgb.shape[:2]

        self.points      = []   # list of [x, y]
        self.labels      = []   # 1 = foreground (burn), 0 = background
        self.current_mask = None
        self.saved       = False

        # Set image in SAM predictor
        self.predictor.set_image(image_rgb)

        # Build figure
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.suptitle(
            f"{image_name}\n"
            "LEFT CLICK = burn area  |  RIGHT CLICK = not burn  |  "
            "SPACE = segment  |  S = save  |  U = undo  |  R = reset  |  Q = quit",
            fontsize=10
        )

        self.axes[0].set_title("Original — click to add points")
        self.axes[1].set_title("SAM mask")
        self.axes[2].set_title("Overlay (burn = red, edge = green)")

        for ax in self.axes:
            ax.axis("off")

        self.im0 = self.axes[0].imshow(image_rgb)
        self.im1 = self.axes[1].imshow(np.zeros((self.h, self.w), dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
        self.im2 = self.axes[2].imshow(image_rgb)

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event",    self._on_key)

        # Instruction text bottom
        self.fig.text(0.5, 0.01,
            "Green dots = burn points | Red dots = background points | Press SPACE after clicking to segment",
            ha="center", fontsize=9, color="gray"
        )

        plt.tight_layout(rect=[0, 0.04, 1, 0.92])
        self._refresh_display()

    def _on_click(self, event):
        if event.inaxes != self.axes[0]:
            return
        x, y = int(event.xdata), int(event.ydata)
        label = 1 if event.button == 1 else 0   # left=burn, right=not burn
        self.points.append([x, y])
        self.labels.append(label)
        print(f"  Point added: ({x}, {y}) — {'BURN' if label == 1 else 'background'}")
        self._refresh_display()

    def _on_key(self, event):
        if event.key == " ":
            self._run_sam()
        elif event.key == "s":
            self._save()
        elif event.key == "u":
            self._undo()
        elif event.key == "r":
            self._reset()
        elif event.key in ["q", "escape"]:
            plt.close(self.fig)

    def _run_sam(self):
        if not self.points:
            print("  No points yet — click on the burn area first")
            return

        pts_arr = np.array(self.points)
        lbl_arr = np.array(self.labels)

        print(f"  Running SAM with {len(self.points)} point(s)...")
        masks, scores, _ = self.predictor.predict(
            point_coords=pts_arr,
            point_labels=lbl_arr,
            multimask_output=True     # returns 3 mask candidates
        )

        # Pick the mask with the highest score
        best_idx = np.argmax(scores)
        self.current_mask = masks[best_idx].astype(np.uint8) * 255
        print(f"  SAM done — best mask score: {scores[best_idx]:.3f}  "
              f"({int(np.sum(self.current_mask == 255)):,} pixels = "
              f"{round(np.sum(self.current_mask == 255) / self.current_mask.size * 100, 1)}% of image)")

        self._refresh_display()

    def _save(self):
        if self.current_mask is None:
            print("  No mask yet — press SPACE to segment first")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "masks"),  exist_ok=True)

        base = os.path.splitext(self.image_name)[0]

        # Save mask
        mask_path = os.path.join(self.output_dir, "masks",  f"{base}.png")
        img_path  = os.path.join(self.output_dir, "images", f"{base}.png")
        overlay_path = os.path.join(self.output_dir, f"{base}_overlay.png")

        cv2.imwrite(mask_path, self.current_mask)

        # Save original image as PNG (for U-Net training)
        bgr = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, bgr)

        # Save overlay for visual review
        overlay = self._make_overlay()
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(f"  SAVED → {mask_path}")
        print(f"  SAVED → {img_path}")
        print(f"  SAVED → {overlay_path}")
        self.saved = True

        # Update title to confirm save
        self.fig.suptitle(
            f"✓ SAVED: {self.image_name} — close window or press Q for next image",
            fontsize=11, color="green"
        )
        self.fig.canvas.draw()

    def _undo(self):
        if self.points:
            self.points.pop()
            self.labels.pop()
            print(f"  Undo — {len(self.points)} point(s) remaining")
            self._refresh_display()

    def _reset(self):
        self.points       = []
        self.labels       = []
        self.current_mask = None
        print("  Reset — all points cleared")
        self._refresh_display()

    def _make_overlay(self):
        if self.current_mask is None:
            return self.image_rgb.copy()

        vis     = self.image_rgb.copy()
        overlay = self.image_rgb.copy()
        mask_bool = self.current_mask == 255

        # Red fill on burn area
        overlay[mask_bool] = [200, 30, 30]
        vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

        # Green edge
        edges = cv2.Canny(self.current_mask, 30, 100)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        vis[edges == 255] = [0, 230, 0]

        return vis

    def _refresh_display(self):
        # Panel 0 — original with dots
        img_with_pts = self.image_rgb.copy()
        for (px, py), lbl in zip(self.points, self.labels):
            color = (0, 220, 0) if lbl == 1 else (220, 0, 0)
            cv2.circle(img_with_pts, (px, py), 8, color, -1)
            cv2.circle(img_with_pts, (px, py), 8, (255, 255, 255), 2)
        self.im0.set_data(img_with_pts)
        self.axes[0].set_title(
            f"Original — {len(self.points)} point(s) | "
            f"green=burn  red=background"
        )

        # Panel 1 — mask
        if self.current_mask is not None:
            self.im1.set_data(self.current_mask)
            burn_pct = round(np.sum(self.current_mask == 255) / self.current_mask.size * 100, 1)
            self.axes[1].set_title(f"SAM mask ({burn_pct}% of image)")
        else:
            self.im1.set_data(np.zeros((self.h, self.w), dtype=np.uint8))
            self.axes[1].set_title("SAM mask — press SPACE to segment")

        # Panel 2 — overlay
        self.im2.set_data(self._make_overlay())
        self.axes[2].set_title("Overlay (red=burn, green=edge)")

        self.fig.canvas.draw()

    def run(self):
        plt.show(block=True)
        return self.saved, self.current_mask


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SAM-based burn area labeler")
    parser.add_argument("--image",      type=str, help="Single image path")
    parser.add_argument("--batch",      type=str, help="Folder of images to label one by one")
    parser.add_argument("--output",     type=str, default="sam_labeled", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=SAM_CHECKPOINT,
                        help=f"Path to SAM checkpoint (default: {SAM_CHECKPOINT})")
    args = parser.parse_args()

    # ── Load SAM model ──────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"\nERROR: SAM checkpoint not found at: {args.checkpoint}")
        print("\nDownload it with:")
        print("  curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o sam_vit_b_01ec64.pth")
        print("\nOr from your browser:")
        print("  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        sys.exit(1)

    print(f"Loading SAM model from {args.checkpoint} ...")
    sam       = sam_model_registry[SAM_MODEL_TYPE](checkpoint=args.checkpoint)
    predictor = SamPredictor(sam)
    print("SAM model loaded.\n")

    # ── Collect images ──────────────────────────────────────────
    images = []
    if args.image:
        images = [args.image]
    elif args.batch:
        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        images = sorted([
            os.path.join(args.batch, f) for f in os.listdir(args.batch)
            if os.path.splitext(f)[1].lower() in exts
        ])
        print(f"Found {len(images)} images in {args.batch}")
    else:
        print("Usage:")
        print("  python burn_sam_labeler.py --image pat1day28A_front.png")
        print("  python burn_sam_labeler.py --batch scans/ --output labeled/")
        sys.exit(0)

    # ── Process each image ──────────────────────────────────────
    saved_count = 0
    for i, img_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] {img_path}")
        try:
            image_rgb  = load_image(img_path)
            image_name = os.path.basename(img_path)
            labeler    = BurnLabeler(predictor, image_rgb, image_name, args.output)
            saved, _   = labeler.run()
            if saved:
                saved_count += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\nDone. {saved_count}/{len(images)} images labeled.")
    print(f"Training data saved to: {args.output}/")
    print(f"  {args.output}/images/  ← input images for U-Net")
    print(f"  {args.output}/masks/   ← binary masks for U-Net")


if __name__ == "__main__":
    main()