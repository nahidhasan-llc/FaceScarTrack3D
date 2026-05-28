"""
normal_color_tool.py
====================
Interactive tool for:
  1. Manually selecting a "normal skin" zone on a patient's face TIF image
  2. Normalizing all timepoint images (per patient) using CIELAB color space,
     anchored to that normal zone — so the normal zone stays color-constant
     across all sessions, and burn area changes become clearly visible.

DATASET FOLDER STRUCTURE expected:
  ../../Dataset/
      pat1day0C.tif        <- patient 1, day 0, scan C
      pat1day0C            <- companion range file (ignored here)
      pat1day28A.tif       <- patient 1, day 28, scan A (mouth open)
      pat1day28C2.tif      <- patient 1, day 28, scan C variant 2
      pat2day0A.tif        <- patient 2, day 0, scan A
      ...

OUTPUT FOLDER:
  ../../Dataset/color_normalized/
      pat1/
          pat1day0C_normalized.tif
          pat1day28A_normalized.tif
          pat1day28C2_normalized.tif
          pat1_normal_zone_mask.npy   <- saved polygon mask
          pat1_normal_zone_stats.json <- LAB stats of normal zone per timepoint
      pat2/
          ...

USAGE:
  python normal_color_tool.py

  On first run: opens an interactive window, you draw the normal skin zone.
  On subsequent runs: reuses saved mask, just re-normalizes if new files found.

  Options:
    --patient pat1          process only one patient
    --reset-mask pat1       delete saved mask and re-draw for that patient
    --show-comparison       display before/after side by side after processing
    --reference-day 0       which day to use as the reference (default: lowest day found)

REQUIREMENTS:
  pip install numpy pillow opencv-python scikit-image matplotlib
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, RadioButtons
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from PIL import Image
from skimage import color as skcolor
from skimage.exposure import match_histograms
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR   = Path("D:/NahidW/Dataset")
OUTPUT_DIR    = DATASET_DIR / "color_normalized"
MASK_FILENAME = "{patient_id}_normal_zone_mask.npy"
STATS_FILENAME= "{patient_id}_normal_zone_stats.json"


# ─────────────────────────────────────────────────────────────────────────────
#  FILENAME PARSING
# ─────────────────────────────────────────────────────────────────────────────

# Matches: pat1day0C.tif, pat1day28A.tif, pat1day28C2.tif, pat12day180AB.tif
SCAN_PATTERN = re.compile(
    r"^(?P<patient>pat\d+)"          # patient id: pat1, pat12
    r"day(?P<day>\d+)"               # day number
    r"(?P<variant>[A-Z][A-Z0-9]?)"   # scan variant: A, B, C, C2, AB
    r"\.tif$",
    re.IGNORECASE
)

@dataclass
class ScanFile:
    path: Path
    patient_id: str
    day: int
    variant: str

    @property
    def label(self):
        return f"Day {self.day} — Scan {self.variant.upper()}"

    @property
    def out_name(self):
        return self.path.stem + "_normalized.tif"


def discover_scans(dataset_dir: Path, patient_filter: Optional[str] = None) -> dict:
    """
    Returns dict: { patient_id -> sorted list of ScanFile }
    Grouped and sorted by day then variant.
    """
    patients = {}
    for f in sorted(dataset_dir.glob("*/*.tif")):
        m = SCAN_PATTERN.match(f.name)
        if not m:
            continue
        pid     = m.group("patient").lower()
        day     = int(m.group("day"))
        variant = m.group("variant").upper()

        if patient_filter and pid != patient_filter.lower():
            continue

        sf = ScanFile(path=f, patient_id=pid, day=day, variant=variant)
        patients.setdefault(pid, []).append(sf)

    # Sort each patient's scans by day then variant
    for pid in patients:
        patients[pid].sort(key=lambda s: (s.day, s.variant))

    return patients


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_tif(path: Path) -> np.ndarray:
    """Load TIF as uint8 RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB (H,W,3) to float32 CIELAB (H,W,3)."""
    img_f = img_rgb.astype(np.float32) / 255.0
    return skcolor.rgb2lab(img_f).astype(np.float32)


def lab_to_rgb(img_lab: np.ndarray) -> np.ndarray:
    """Convert float32 CIELAB back to uint8 RGB."""
    img_rgb = skcolor.lab2rgb(img_lab)
    img_rgb = np.clip(img_rgb * 255, 0, 255).astype(np.uint8)
    return img_rgb


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE POLYGON SELECTION
# ─────────────────────────────────────────────────────────────────────────────

class PolygonSelector:
    """
    Interactive matplotlib tool for drawing a polygon ROI on an image.

    Controls:
      Left click      — add polygon vertex
      Right click     — undo last vertex
      Enter / D       — confirm selection
      Escape / Q      — cancel / restart
      Z               — zoom into selection area
    """

    def __init__(self, image: np.ndarray, title: str = "Select Normal Skin Zone"):
        self.image    = image
        self.title    = title
        self.points   = []          # list of (x, y) pixel coords
        self.finished = False
        self.cancelled= False
        self._zoom_box= None

        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.patch.set_facecolor("#1a1a2e")
        self.ax.set_facecolor("#1a1a2e")

        self.ax.imshow(image)
        self.ax.set_title(
            f"{title}\n"
            "LEFT CLICK = add point  |  RIGHT CLICK = undo  |  "
            "ENTER or D = confirm  |  ESC = restart",
            color="white", fontsize=10, pad=10
        )
        self.ax.axis("off")

        self._poly_line, = self.ax.plot([], [], 'o-',
            color='#00ff88', linewidth=2, markersize=6,
            markerfacecolor='#00ff88', markeredgecolor='white',
            markeredgewidth=1.5
        )
        self._close_line, = self.ax.plot([], [], '--',
            color='#00ff88', linewidth=1.5, alpha=0.6
        )
        self._fill = None

        # Instruction overlay
        self._info = self.ax.text(
            10, 10,
            "Click to place points on the UNBURNED skin area.\n"
            "Aim for a region that stays burn-free across all timepoints\n"
            "(e.g. forehead, side of neck, or ear area).",
            color='#aaffcc', fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d0d1a', alpha=0.75)
        )

        self._point_labels = []

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event',    self._on_key)

        plt.tight_layout()

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:  # left click — add point
            x, y = int(event.xdata), int(event.ydata)
            self.points.append((x, y))
            lbl = self.ax.text(x+4, y-4, str(len(self.points)),
                               color='yellow', fontsize=7, fontweight='bold')
            self._point_labels.append(lbl)
            self._redraw()
        elif event.button == 3:  # right click — undo
            if self.points:
                self.points.pop()
                lbl = self._point_labels.pop()
                lbl.remove()
                self._redraw()

    def _on_key(self, event):
        if event.key in ('enter', 'd'):
            if len(self.points) >= 3:
                self.finished = True
                plt.close(self.fig)
            else:
                self._info.set_text("Need at least 3 points to define a zone!")
                self.fig.canvas.draw()
        elif event.key in ('escape', 'q'):
            self.points.clear()
            for lbl in self._point_labels:
                lbl.remove()
            self._point_labels.clear()
            if self._fill:
                self._fill.remove()
                self._fill = None
            self._redraw()
        elif event.key == 'z' and len(self.points) >= 2:
            self._zoom_to_selection()

    def _zoom_to_selection(self):
        if not self.points:
            return
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        pad = 40
        self.ax.set_xlim(max(0, min(xs)-pad), min(self.image.shape[1], max(xs)+pad))
        self.ax.set_ylim(min(self.image.shape[0], max(ys)+pad), max(0, min(ys)-pad))
        self.fig.canvas.draw()

    def _redraw(self):
        if not self.points:
            self._poly_line.set_data([], [])
            self._close_line.set_data([], [])
            if self._fill:
                self._fill.remove()
                self._fill = None
            self.fig.canvas.draw()
            return

        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        self._poly_line.set_data(xs, ys)

        if len(self.points) >= 2:
            # dashed closing line
            self._close_line.set_data(
                [xs[-1], xs[0]], [ys[-1], ys[0]]
            )

        if len(self.points) >= 3:
            # semi-transparent fill
            if self._fill:
                self._fill.remove()
            poly_arr = np.array(self.points)
            patch = MplPolygon(poly_arr, closed=True,
                               facecolor='#00ff88', alpha=0.20,
                               edgecolor='none')
            self._fill = self.ax.add_patch(patch)

        self.fig.canvas.draw()

    def run(self) -> Optional[np.ndarray]:
        """
        Opens the interactive window. Returns polygon points as Nx2 numpy array,
        or None if cancelled.
        """
        plt.show(block=True)
        if self.finished and len(self.points) >= 3:
            return np.array(self.points, dtype=np.int32)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  MASK CREATION FROM POLYGON
# ─────────────────────────────────────────────────────────────────────────────

def polygon_to_mask(polygon_pts: np.ndarray, img_shape: tuple) -> np.ndarray:
    """
    Convert polygon points (Nx2 x,y) to a binary mask of shape (H, W).
    Uses OpenCV fillPoly for sub-pixel accuracy.
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts  = polygon_pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def extract_zone_pixels(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return pixels inside the mask as (N, 3) array."""
    return img_rgb[mask > 0]


# ─────────────────────────────────────────────────────────────────────────────
#  CIELAB ZONE-ANCHORED NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_zone_lab_stats(img_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """
    Compute mean L*, a*, b* of the normal zone in CIELAB color space.
    Also compute std per channel.
    """
    lab = rgb_to_lab(img_rgb)
    zone_pixels = lab[mask > 0]  # (N, 3)
    return {
        "L_mean": float(np.mean(zone_pixels[:, 0])),
        "L_std":  float(np.std(zone_pixels[:, 0])),
        "a_mean": float(np.mean(zone_pixels[:, 1])),
        "a_std":  float(np.std(zone_pixels[:, 1])),
        "b_mean": float(np.mean(zone_pixels[:, 2])),
        "b_std":  float(np.std(zone_pixels[:, 2])),
        "n_pixels": int(np.sum(mask > 0))
    }


def normalize_to_reference(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    ref_stats: dict,
) -> np.ndarray:
    """
    Normalize an image so its normal zone LAB stats match the reference stats.

    Strategy: Compute per-channel affine shift in LAB space so that:
      - The mean of the zone L* matches ref L_mean  (brightness correction)
      - The mean of the zone a* matches ref a_mean  (red-green color correction)
      - The mean of the zone b* matches ref b_mean  (blue-yellow color correction)

    The same shift is applied to the WHOLE image (not just the zone),
    so the relationship between burn and normal skin is preserved.

    Additionally, scale (std matching) is applied per channel so that
    the contrast within the zone also normalizes — this handles cases where
    one session had much harsher directional lighting.
    """
    lab = rgb_to_lab(img_rgb)

    # Compute this image's zone stats
    cur_stats = compute_zone_lab_stats(img_rgb, mask)

    normalized_lab = lab.copy()

    for ch_idx, (mean_key, std_key) in enumerate([
        ("L_mean", "L_std"),
        ("a_mean", "a_std"),
        ("b_mean", "b_std"),
    ]):
        cur_mean = cur_stats[mean_key]
        cur_std  = cur_stats[std_key]
        ref_mean = ref_stats[mean_key]
        ref_std  = ref_stats[std_key]

        # Affine normalization: shift + scale to match reference
        # new_val = (val - cur_mean) * (ref_std / max(cur_std, 1e-6)) + ref_mean
        # This centers the zone on ref_mean and scales variance to ref
        channel = lab[:, :, ch_idx]

        if cur_std > 1e-6:
            scale = ref_std / cur_std
        else:
            scale = 1.0

        normalized_lab[:, :, ch_idx] = (channel - cur_mean) * scale + ref_mean

    # Clip to valid LAB ranges
    normalized_lab[:, :, 0] = np.clip(normalized_lab[:, :, 0],   0, 100)
    normalized_lab[:, :, 1] = np.clip(normalized_lab[:, :, 1], -128, 127)
    normalized_lab[:, :, 2] = np.clip(normalized_lab[:, :, 2], -128, 127)

    return lab_to_rgb(normalized_lab)


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE / LOAD MASKS AND STATS
# ─────────────────────────────────────────────────────────────────────────────

def get_patient_output_dir(patient_id: str) -> Path:
    d = OUTPUT_DIR / patient_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def mask_path(patient_id: str) -> Path:
    return get_patient_output_dir(patient_id) / MASK_FILENAME.format(patient_id=patient_id)


def stats_path(patient_id: str) -> Path:
    return get_patient_output_dir(patient_id) / STATS_FILENAME.format(patient_id=patient_id)


def save_mask(patient_id: str, polygon_pts: np.ndarray, img_shape: tuple):
    """Save polygon points and the resulting binary mask."""
    out = get_patient_output_dir(patient_id)
    np.save(mask_path(patient_id), polygon_pts)
    print(f"  ✓ Mask saved: {mask_path(patient_id)}")


def load_mask(patient_id: str, img_shape: tuple) -> Optional[np.ndarray]:
    """Load saved mask. Returns binary mask (H,W) or None."""
    mp = mask_path(patient_id)
    if not mp.exists():
        return None
    polygon_pts = np.load(mp)
    return polygon_to_mask(polygon_pts, img_shape)


def save_stats(patient_id: str, stats_by_scan: dict):
    with open(stats_path(patient_id), "w") as f:
        json.dump(stats_by_scan, f, indent=2)
    print(f"  ✓ Zone stats saved: {stats_path(patient_id)}")


def load_stats(patient_id: str) -> dict:
    sp = stats_path(patient_id)
    if not sp.exists():
        return {}
    with open(sp) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  COMPARISON VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def show_comparison(scans: list, normalized_imgs: list, stats_by_scan: dict):
    """
    Show a side-by-side comparison: original vs normalized for all timepoints.
    Also shows the normal zone LAB stats over time as a small chart.
    """
    n = len(scans)
    if n == 0:
        return

    fig = plt.figure(figsize=(max(16, n * 4), 10))
    fig.patch.set_facecolor("#0d1117")

    # Title
    fig.suptitle(
        f"Color Normalization Results — {scans[0].patient_id.upper()}\n"
        "Top: Original  |  Bottom: CIELAB Normalized (anchored to normal zone)",
        color="white", fontsize=12, y=0.98
    )

    cols = n
    rows = 3  # original, normalized, diff

    for i, (scan, norm_img) in enumerate(zip(scans, normalized_imgs)):
        orig_img = load_tif(scan.path)

        # Original
        ax1 = fig.add_subplot(rows, cols, i + 1)
        ax1.imshow(orig_img)
        ax1.set_title(scan.label, color="#aaaaff", fontsize=8)
        ax1.axis("off")

        # Normalized
        ax2 = fig.add_subplot(rows, cols, cols + i + 1)
        ax2.imshow(norm_img)
        ax2.set_title("Normalized", color="#aaffaa", fontsize=8)
        ax2.axis("off")

        # Difference (amplified x3 for visibility)
        ax3 = fig.add_subplot(rows, cols, 2*cols + i + 1)
        diff = np.clip(
            (norm_img.astype(np.int16) - orig_img.astype(np.int16)) * 3 + 128,
            0, 255
        ).astype(np.uint8)
        ax3.imshow(diff)
        ax3.set_title("Diff ×3", color="#ffaaaa", fontsize=8)
        ax3.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=False)

    # ── LAB stats chart ──
    if stats_by_scan:
        labels  = list(stats_by_scan.keys())
        L_means = [stats_by_scan[k]["L_mean"] for k in labels]
        a_means = [stats_by_scan[k]["a_mean"] for k in labels]
        b_means = [stats_by_scan[k]["b_mean"] for k in labels]

        fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig2.patch.set_facecolor("#0d1117")
        fig2.suptitle("Normal Zone LAB Stats Across Timepoints (Before Normalization)",
                      color="white", fontsize=11)

        for ax, vals, name, clr in zip(
            axes,
            [L_means, a_means, b_means],
            ["L* (Brightness)", "a* (Red-Green)", "b* (Blue-Yellow)"],
            ["#ffcc44", "#ff6688", "#44aaff"]
        ):
            ax.set_facecolor("#161b22")
            ax.plot(range(len(labels)), vals, 'o-', color=clr, linewidth=2)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7, color="white")
            ax.set_title(name, color="white", fontsize=10)
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#333344")
            ax.grid(True, color="#222233", linewidth=0.5)

        plt.tight_layout()
        plt.show(block=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def process_patient(
    patient_id: str,
    scans: list,
    reference_day: Optional[int] = None,
    reset_mask: bool = False,
    show_comparison_flag: bool = False,
):
    print(f"\n{'='*60}")
    print(f"  Patient: {patient_id.upper()}")
    print(f"  Scans found: {len(scans)}")
    for s in scans:
        print(f"    {s.path.name}  →  {s.label}")
    print(f"{'='*60}")

    out_dir = get_patient_output_dir(patient_id)

    # ── Pick the reference scan (lowest day, or user-specified) ──
    if reference_day is not None:
        ref_scans = [s for s in scans if s.day == reference_day]
        if not ref_scans:
            print(f"  ⚠ No scan found for day {reference_day}. Using earliest.")
            ref_scan = scans[0]
        else:
            ref_scan = ref_scans[0]
    else:
        ref_scan = scans[0]  # already sorted by day

    print(f"\n  Reference scan: {ref_scan.path.name} ({ref_scan.label})")

    # ── Load reference image ──
    ref_img = load_tif(ref_scan.path)
    h, w    = ref_img.shape[:2]
    print(f"  Image size: {w} × {h} px")

    # ── Load or draw the normal zone mask ──
    mask = None
    if not reset_mask:
        mask = load_mask(patient_id, ref_img.shape)

    if mask is None:
        print(f"\n  No saved mask found. Opening interactive selector on reference scan...")
        print(f"  TIP: Select an area of UNBURNED skin that will stay normal across all timepoints.")
        print(f"       Good choices: forehead edge, side of neck, behind ear.")

        selector = PolygonSelector(ref_img,
            title=f"{patient_id.upper()} — Draw normal skin zone on: {ref_scan.label}")
        polygon_pts = selector.run()

        if polygon_pts is None:
            print("  ✗ Selection cancelled. Skipping patient.")
            return

        save_mask(patient_id, polygon_pts, ref_img.shape)
        mask = polygon_to_mask(polygon_pts, ref_img.shape)
        print(f"  ✓ Zone selected: {int(np.sum(mask > 0))} pixels")
    else:
        print(f"  ✓ Loaded saved mask ({int(np.sum(mask > 0))} pixels)")

    # ── Compute reference zone stats ──
    print(f"\n  Computing reference zone LAB stats...")
    ref_stats = compute_zone_lab_stats(ref_img, mask)
    print(f"    L* = {ref_stats['L_mean']:.1f} ± {ref_stats['L_std']:.1f}  "
          f"(brightness)")
    print(f"    a* = {ref_stats['a_mean']:.2f} ± {ref_stats['a_std']:.2f}  "
          f"(red-green)")
    print(f"    b* = {ref_stats['b_mean']:.2f} ± {ref_stats['b_std']:.2f}  "
          f"(blue-yellow)")

    # ── Process all scans ──
    stats_by_scan = {}
    normalized_imgs = []

    print(f"\n  Normalizing all scans...")

    for scan in scans:
        img = load_tif(scan.path)

        # If image size differs from reference, resize mask
        if img.shape[:2] != (h, w):
            print(f"    ⚠ {scan.path.name} has different size {img.shape[:2]} vs ref {(h,w)}, resizing mask")
            scan_mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        else:
            scan_mask = mask

        # Store original zone stats (before normalization, for the chart)
        orig_stats = compute_zone_lab_stats(img, scan_mask)
        stats_by_scan[f"Day{scan.day}_{scan.variant}"] = orig_stats

        # Normalize
        norm_img = normalize_to_reference(img, scan_mask, ref_stats)
        normalized_imgs.append(norm_img)

        # Save output
        out_path = out_dir / scan.out_name
        Image.fromarray(norm_img).save(out_path, format="TIFF")
        print(f"    ✓ {scan.path.name}  →  {out_path.name}")

        # Also save a PNG for quick preview (TIF can be slow to open)
        png_path = out_dir / (scan.path.stem + "_normalized_preview.png")
        Image.fromarray(norm_img).save(png_path, format="PNG")

    # Save zone stats for later analysis
    save_stats(patient_id, stats_by_scan)

    # Also save the mask as a visual PNG for documentation
    mask_visual = np.stack([mask, mask, mask], axis=-1)
    overlay     = (ref_img.astype(np.float32) * 0.6 +
                   mask_visual.astype(np.float32) * [0, 1.5, 0.5]).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_dir / f"{patient_id}_normal_zone_overlay.png")
    print(f"  ✓ Zone overlay saved for documentation")

    print(f"\n  ✅ Done! Output: {out_dir}")

    # ── Optional comparison view ──
    if show_comparison_flag:
        print("\n  Opening comparison viewer...")
        show_comparison(scans, normalized_imgs, stats_by_scan)

    return stats_by_scan


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Normal skin zone selection and CIELAB color normalization tool"
    )
    p.add_argument("--patient",       type=str, default=None,
                   help="Process only this patient (e.g. pat1)")
    p.add_argument("--reset-mask",    type=str, default=None, metavar="PATIENT",
                   help="Delete and re-draw the mask for this patient")
    p.add_argument("--reference-day", type=int, default=None,
                   help="Day number to use as reference (default: lowest day found)")
    p.add_argument("--show-comparison", action="store_true",
                   help="Show before/after comparison after processing")
    p.add_argument("--dataset",       type=str, default=None,
                   help="Override dataset directory path")
    return p.parse_args()


def main():
    args = parse_args()

    # Allow dataset dir override
    global DATASET_DIR, OUTPUT_DIR
    if args.dataset:
        DATASET_DIR = Path(args.dataset)
        OUTPUT_DIR  = DATASET_DIR / "color_normalized"

    if not DATASET_DIR.exists():
        print(f"✗ Dataset directory not found: {DATASET_DIR.resolve()}")
        print("  Edit DATASET_DIR at the top of this script, or use --dataset <path>")
        sys.exit(1)

    print(f"Dataset dir : {DATASET_DIR.resolve()}")
    print(f"Output dir  : {OUTPUT_DIR.resolve()}")

    # Handle reset-mask
    if args.reset_mask:
        mp = mask_path(args.reset_mask.lower())
        if mp.exists():
            mp.unlink()
            print(f"✓ Deleted mask for {args.reset_mask}")
        else:
            print(f"No mask found for {args.reset_mask}")

    # Discover scans
    patients = discover_scans(DATASET_DIR, patient_filter=args.patient)

    if not patients:
        print(f"\n✗ No TIF files matched the expected naming pattern.")
        print("  Expected pattern: pat<N>day<D><VARIANT>.tif")
        print(f"  Examples: pat1day0C.tif, pat1day28A.tif, pat2day180AB.tif")
        print(f"\n  Files found in {DATASET_DIR}:")
        for f in list(DATASET_DIR.glob("*.tif"))[:10]:
            print(f"    {f.name}")
        sys.exit(1)

    print(f"\nFound {len(patients)} patient(s): {', '.join(patients.keys())}")

    # Process each patient
    all_stats = {}
    for patient_id, scans in patients.items():
        reset = (args.reset_mask and args.reset_mask.lower() == patient_id)
        stats = process_patient(
            patient_id       = patient_id,
            scans            = scans,
            reference_day    = args.reference_day,
            reset_mask       = reset,
            show_comparison_flag = args.show_comparison,
        )
        if stats:
            all_stats[patient_id] = stats

    print(f"\n{'='*60}")
    print("  ALL DONE")
    print(f"  Normalized images → {OUTPUT_DIR.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()




# # Basic — processes all patients found in the dataset folder
# python normal_color_tool.py

# # Only one patient
# python normal_color_tool.py --patient pat1

# # Show before/after comparison after processing
# python normal_color_tool.py --patient pat1 --show-comparison

# # Re-draw the mask (if you selected wrong area)
# python normal_color_tool.py --reset-mask pat1

# # Override dataset path if needed
# python normal_color_tool.py --dataset "D:/your/dataset/folder"