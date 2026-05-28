"""
STEP 2B — Zero-Shot with SAM2 (No Training Needed)
====================================================
Use this INSTEAD of step2_train.py if you want immediate results
without training. SAM2 is a foundation model that can segment
burn areas from a simple point or box prompt.

Install:
    pip install sam2 opencv-python numpy

Download checkpoint:
    # SAM2 large (best quality, ~900MB)
    wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

    # SAM2 small (faster, ~180MB)
    wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

Usage:
    # Interactive — click on burn area
    python step2b_sam2_zeroshot.py --image face.jpg --interactive

    # Auto-mode — tries to detect burn region automatically
    python step2b_sam2_zeroshot.py --image face.jpg --auto

    # Box prompt — you specify region
    python step2b_sam2_zeroshot.py --image face.jpg --box 100 50 400 350
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

# ── SAM2 setup ────────────────────────────────────────────────────────────────
try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("SAM2 not installed. Run: pip install sam2")


# ── Config ────────────────────────────────────────────────────────────────────
SAM2_CHECKPOINT = "sam2.1_hiera_large.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"
EDGE_COLOR      = (0, 255, 80)    # green
FILL_COLOR      = (0, 60, 255)    # blue-red
FILL_ALPHA      = 0.25


# ── Model ─────────────────────────────────────────────────────────────────────
def load_sam2(checkpoint=SAM2_CHECKPOINT, config=SAM2_CONFIG):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2   = build_sam2(config, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2)
    print(f"SAM2 loaded on {device}")
    return predictor


# ── Prediction helpers ────────────────────────────────────────────────────────
def predict_from_points(predictor, image_rgb, points, labels):
    """
    points: list of (x, y) tuples
    labels: list of 1 (foreground) or 0 (background) per point
    """
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=np.array(points),
        point_labels=np.array(labels),
        multimask_output=True,
    )
    # Take highest-scoring mask
    best_idx = np.argmax(scores)
    return masks[best_idx].astype(np.uint8) * 255, scores[best_idx]


def predict_from_box(predictor, image_rgb, box):
    """box: [x1, y1, x2, y2]"""
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        box=np.array(box),
        multimask_output=True,
    )
    best_idx = np.argmax(scores)
    return masks[best_idx].astype(np.uint8) * 255, scores[best_idx]


def auto_detect_burn(predictor, image_rgb):
    """
    Heuristic: burn areas often have different color signature.
    Uses color clustering to generate candidate prompts, then SAM2 segments.
    """
    # Convert to LAB for better skin/burn differentiation
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    # Burn areas typically have abnormal redness (higher A channel in LAB)
    a_channel = lab[:, :, 1].astype(np.float32)

    # Find pixels significantly more red than average (potential burn)
    mean_a = a_channel.mean()
    std_a  = a_channel.std()
    burn_candidate = (a_channel > mean_a + 0.5 * std_a).astype(np.uint8)

    # Find centroid of candidate region
    moments = cv2.moments(burn_candidate)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        h, w = image_rgb.shape[:2]
        cx, cy = w // 2, h // 2

    points = [(cx, cy)]
    labels = [1]  # foreground

    # Add background points (corners)
    h, w = image_rgb.shape[:2]
    for bx, by in [(20, 20), (w-20, 20), (20, h-20), (w-20, h-20)]:
        points.append((bx, by))
        labels.append(0)

    return predict_from_points(predictor, image_rgb, points, labels)


# ── Visualization ─────────────────────────────────────────────────────────────
def draw_edges(image_bgr, binary_mask, score=None):
    result = image_bgr.copy()

    # Fill
    overlay = result.copy()
    overlay[binary_mask > 127] = FILL_COLOR
    result = cv2.addWeighted(overlay, FILL_ALPHA, result, 1 - FILL_ALPHA, 0)

    # Edges
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, EDGE_COLOR, 2)

    # Stats
    h, w = binary_mask.shape
    pct  = (binary_mask > 127).sum() / (h * w) * 100
    cv2.putText(result, f"Burn: {pct:.1f}%  Regions: {len(contours)}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if score is not None:
        cv2.putText(result, f"SAM2 confidence: {score:.2f}",
                    (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

    return result


# ── Interactive mode ──────────────────────────────────────────────────────────
class InteractiveAnnotator:
    """Click points on image to guide SAM2 segmentation."""

    def __init__(self, image_bgr, predictor):
        self.image_bgr  = image_bgr
        self.image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor  = predictor
        self.points     = []
        self.labels     = []
        self.result     = image_bgr.copy()

        predictor.set_image(self.image_rgb)

        cv2.namedWindow("SAM2 Burn Detection", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("SAM2 Burn Detection", self._on_click)
        self._draw_instructions()

    def _draw_instructions(self):
        disp = self.image_bgr.copy()
        cv2.putText(disp, "LEFT CLICK = burn area (foreground)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(disp, "RIGHT CLICK = background",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(disp, "ENTER = save | R = reset | Q = quit",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("SAM2 Burn Detection", disp)

    def _on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            self.labels.append(1)
            self._predict_and_show()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.append([x, y])
            self.labels.append(0)
            self._predict_and_show()

    def _predict_and_show(self):
        if not self.points:
            return

        masks, scores, _ = self.predictor.predict(
            point_coords=np.array(self.points),
            point_labels=np.array(self.labels),
            multimask_output=True,
        )
        best = np.argmax(scores)
        mask = masks[best].astype(np.uint8) * 255

        self.result = draw_edges(self.image_bgr, mask, scores[best])

        # Draw click points
        for (px, py), lbl in zip(self.points, self.labels):
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(self.result, (px, py), 6, color, -1)

        cv2.imshow("SAM2 Burn Detection", self.result)

    def run(self):
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER
                break
            elif key == ord("r"):
                self.points, self.labels = [], []
                self._draw_instructions()
            elif key == ord("q"):
                self.result = self.image_bgr
                break

        cv2.destroyAllWindows()
        return self.result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",       required=True)
    parser.add_argument("--checkpoint",  default=SAM2_CHECKPOINT)
    parser.add_argument("--config",      default=SAM2_CONFIG)
    parser.add_argument("--output",      default="results/sam2_result.jpg")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--auto",        action="store_true")
    parser.add_argument("--box",         nargs=4, type=int, metavar=("X1","Y1","X2","Y2"))
    args = parser.parse_args()

    if not SAM2_AVAILABLE:
        print("SAM2 not available. Install with: pip install sam2")
        return

    # Load
    predictor = load_sam2(args.checkpoint, args.config)
    image_bgr = cv2.imread(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.interactive:
        annotator = InteractiveAnnotator(image_bgr, predictor)
        result = annotator.run()

    elif args.box:
        mask, score = predict_from_box(predictor, image_rgb, args.box)
        result = draw_edges(image_bgr, mask, score)

    else:  # auto
        mask, score = auto_detect_burn(predictor, image_rgb)
        result = draw_edges(image_bgr, mask, score)

    cv2.imwrite(args.output, result)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
