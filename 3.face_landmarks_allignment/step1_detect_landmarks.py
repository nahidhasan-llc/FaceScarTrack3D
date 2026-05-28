"""
step1_detect_landmarks.py
=========================
Detect 5 face landmarks on every patient TIF image.
Saves results to Dataset/landmarks_raw/<patient>/

DATASET STRUCTURE:
  Dataset/
    pat1/
        pat1day0C/
            pat1day0C.tif
        pat1day28A/
            pat1day28A.tif
    pat2/
        pat2day0A/
            pat2day0A.tif

WHAT IT DOES:
  - Loads each .tif from Dataset/<patient>/<scan_folder>/<scan>.tif
  - Runs MediaPipe Face Landmarker to find 5 landmarks
  - If MediaPipe fails -> opens interactive manual click window
  - Saves:
      <scan>_landmarks.json   <- pixel coords of each landmark
      <scan>_landmarks.png    <- original image with colored dots

LANDMARK JSON FORMAT:
  {
    "scan":      "pat1day0C",
    "patient":   "pat1",
    "day":       0,
    "variant":   "C",
    "image_w":   457,
    "image_h":   480,
    "method":    "mediapipe",   or "manual"
    "landmarks": {
      "sellion":   [234, 189],
      "pronasale": [231, 267],
      "menton":    [228, 390],
      "l_tragion": [112, 220],
      "r_tragion": [361, 215]
    }
  }

OUTPUT FOLDER:
  Dataset/landmarks_raw/
      pat1/
          pat1day0C_landmarks.json
          pat1day0C_landmarks.png
          pat1day28A_landmarks.json
          pat1day28A_landmarks.png

USAGE:
  python step1_detect_landmarks.py
  python step1_detect_landmarks.py --patient pat1
  python step1_detect_landmarks.py --manual pat1day0C   <- force manual
  python step1_detect_landmarks.py --redetect           <- redo all

REQUIREMENTS:
  pip install mediapipe opencv-python pillow numpy matplotlib
  face_landmarker.task must be in same folder as this script
  Download: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

import re, sys, json, argparse, urllib.request, warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR = Path("D:/NahidW/Dataset")
OUTPUT_DIR  = DATASET_DIR / "landmarks_raw"

# MediaPipe 478-point model indices for our 5 landmarks
LANDMARK_NAMES = {
    "sellion":   168,   # nose bridge / between eyes — centre of face
    "pronasale": 4,     # nose tip
    "menton":    152,   # chin centre
    "l_tragion": 234,   # left ear tragus
    "r_tragion": 454,   # right ear tragus
}

# Visualization colors (RGB)
LM_COLOR = {
    "sellion":   (0,   220, 0),     # green
    "pronasale": (220, 220, 0),     # yellow
    "menton":    (220, 0,   220),   # magenta
    "l_tragion": (0,   120, 255),   # blue
    "r_tragion": (255, 120, 0),     # orange
}

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_FILE = "face_landmarker.task"


# ─────────────────────────────────────────────────────────────────────────────
#  FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

SCAN_RE = re.compile(
    r"^(?P<patient>pat\d+)day(?P<day>\d+)(?P<variant>[A-Z][A-Z0-9]?)\.tif$",
    re.IGNORECASE
)

class ScanFile:
    def __init__(self, path, patient_id, day, variant):
        self.path, self.patient_id = path, patient_id
        self.day, self.variant = day, variant
    @property
    def label(self): return f"Day {self.day} Scan {self.variant}"
    @property
    def stem(self):  return self.path.stem


def discover_scans(dataset_dir: Path, patient_filter=None) -> dict:
    patients = {}
    for f in sorted(dataset_dir.glob("*/*/*.tif")):
        m = SCAN_RE.match(f.name)
        if not m:
            continue
        pid = m.group("patient").lower()
        if patient_filter and pid != patient_filter.lower():
            continue
        sf = ScanFile(f, pid, int(m.group("day")), m.group("variant").upper())
        patients.setdefault(pid, []).append(sf)
    for pid in patients:
        patients[pid].sort(key=lambda s: (s.day, s.variant))
    return patients


# ─────────────────────────────────────────────────────────────────────────────
#  MEDIAPIPE
# ─────────────────────────────────────────────────────────────────────────────

def ensure_model() -> str:
    model_path = Path(__file__).parent / MODEL_FILE
    if not model_path.exists():
        print(f"  Downloading MediaPipe model (~30MB)...")
        urllib.request.urlretrieve(MODEL_URL, str(model_path))
        print(f"  Saved: {model_path}")
    return str(model_path)


def mediapipe_detect(img_rgb: np.ndarray, model_path: str) -> Optional[dict]:
    """
    Run MediaPipe on img_rgb (H,W,3 uint8).
    Returns {landmark_name: [px, py]} or None if no face found.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
    except ImportError:
        print("  Run: pip install mediapipe")
        sys.exit(1)

    H, W = img_rgb.shape[:2]
    opts = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.05,
        min_face_presence_confidence=0.05,
        min_tracking_confidence=0.05,
    )
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    with vision.FaceLandmarker.create_from_options(opts) as det:
        res = det.detect(mp_img)

    if not res.face_landmarks:
        return None

    lms = res.face_landmarks[0]
    return {
        name: [int(lms[idx].x * W), int(lms[idx].y * H)]
        for name, idx in LANDMARK_NAMES.items()
    }


def try_all_rotations(img_rgb: np.ndarray, model_path: str):
    """
    Try MediaPipe at 0, 90, 180, 270 degrees.
    Returns (landmarks_in_original_coords, rotation_that_worked)
    or (None, None).

    Landmarks are converted back to original image coordinates
    so step2 always works in original image space.
    """
    def rotate(img, deg):
        return np.array(Image.fromarray(img).rotate(deg, expand=True))

    def back_to_original(px, py, deg, img_orig_shape, img_rot_shape):
        """Convert pixel in rotated image back to original image coords."""
        Ho, Wo = img_orig_shape[:2]
        Hr, Wr = img_rot_shape[:2]
        if deg == 0:
            return px, py
        elif deg == 90:
            # PIL rotate(90): (x,y) in original -> (y, W-1-x) in rotated
            # inverse: orig_x = W_rot-1-py_rot, orig_y = px_rot
            return Wr - 1 - py, px
        elif deg == 180:
            return Wr - 1 - px, Hr - 1 - py
        elif deg == 270:
            # PIL rotate(270): (x,y) in original -> (H-1-y, x) in rotated
            # inverse: orig_x = py_rot, orig_y = H_rot-1-px_rot
            return py, Hr - 1 - px

    print(f"    MediaPipe trying rotations: ", end="", flush=True)
    for deg in [0, 90, 180, 270]:
        rotated = rotate(img_rgb, deg)
        lm = mediapipe_detect(rotated, model_path)
        print(f"{deg}° ", end="", flush=True)
        if lm is None:
            continue

        # Validate: sellion above menton, l_tragion left of r_tragion
        sel = lm["sellion"]; men = lm["menton"]
        lt  = lm["l_tragion"]; rt  = lm["r_tragion"]
        pro = lm["pronasale"]
        if not (sel[1] < men[1] and lt[0] < rt[0] and sel[1] < pro[1] < men[1]):
            continue  # looks wrong orientation, try next

        # Convert all coords back to original image space
        lm_orig = {}
        for name, (px, py) in lm.items():
            ox, oy = back_to_original(px, py, deg,
                                      img_rgb.shape, rotated.shape)
            lm_orig[name] = [int(ox), int(oy)]

        print(f"-> OK at {deg}°")
        return lm_orig, deg

    print(f"-> all failed")
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  MANUAL CLICK FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def manual_click(scan: ScanFile) -> Optional[dict]:
    """
    Opens the image. User clicks landmarks in order:
      sellion -> pronasale -> menton -> l_tragion -> r_tragion
    Right click = undo last point.
    Returns {name: [px, py]} or None if window closed early.
    """
    img   = np.array(Image.open(scan.path).convert("RGB"))
    order = list(LANDMARK_NAMES.keys())
    state = {"lm": {}, "i": 0}
    dots, lbls = [], []

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0d1117")
    ax.imshow(img)
    ax.axis("off")

    instructions = (
        "Click landmarks IN ORDER below.\n"
        "RIGHT CLICK = undo last point.\n"
        "Close window when done.\n\n"
        "Order:\n" +
        "\n".join(f"  {i+1}. {n}" for i, n in enumerate(order))
    )
    fig.text(0.01, 0.5, instructions, color="white", fontsize=9,
             va="center", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.8))
    plt.subplots_adjust(left=0.18)

    def set_title():
        i = state["i"]
        if i < len(order):
            ax.set_title(
                f"{scan.path.name}\n"
                f"Click on: {order[i].upper()}  ({i+1}/{len(order)})",
                color="white", fontsize=11
            )
        else:
            ax.set_title("All done! Close window to continue.",
                         color="#aaffaa", fontsize=12)
        fig.canvas.draw()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1 and state["i"] < len(order):
            name = order[state["i"]]
            px, py = int(event.xdata), int(event.ydata)
            state["lm"][name] = [px, py]
            clr = [c/255 for c in LM_COLOR[name]]
            d = ax.plot(px, py, 'o', color=clr, ms=11,
                        mec='white', mew=1.5)[0]
            l = ax.text(px+8, py-8, name, color=clr,
                        fontsize=8, fontweight='bold')
            dots.append(d); lbls.append(l)
            state["i"] += 1
            set_title()
        elif event.button == 3 and state["i"] > 0:
            state["i"] -= 1
            state["lm"].pop(order[state["i"]], None)
            dots.pop().remove(); lbls.pop().remove()
            set_title()

    fig.canvas.mpl_connect('button_press_event', on_click)
    set_title()
    plt.show(block=True)

    if len(state["lm"]) == len(order):
        return state["lm"]
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_landmark_image(img_rgb: np.ndarray, landmarks: dict, path: Path):
    """Draw colored dots + labels on the original image and save as PNG."""
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for name, (px, py) in landmarks.items():
        r, g, b = LM_COLOR[name]
        cv2.circle(img, (px, py), 10, (b, g, r), -1)
        cv2.circle(img, (px, py), 11, (255, 255, 255), 1)
        cv2.putText(img, name, (px + 12, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (b, g, r), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def save_landmark_json(scan: ScanFile, landmarks: dict,
                       method: str, rotation: int, path: Path):
    img = Image.open(scan.path)
    w, h = img.size
    data = {
        "scan":      scan.stem,
        "patient":   scan.patient_id,
        "day":       scan.day,
        "variant":   scan.variant,
        "label":     scan.label,
        "image_w":   w,
        "image_h":   h,
        "method":    method,
        "rotation_detected_at": rotation,
        "landmarks": landmarks
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS ONE SCAN
# ─────────────────────────────────────────────────────────────────────────────

def process_scan(scan: ScanFile, model_path: str, out_dir: Path,
                 force_manual=False) -> dict:
    print(f"\n  ── {scan.path.name}  ({scan.label})")

    json_path = out_dir / f"{scan.stem}_landmarks.json"
    img_path  = out_dir / f"{scan.stem}_landmarks.png"

    status = {"scan": scan.path.name, "status": "pending", "method": None}

    # Skip if already done
    if json_path.exists() and not force_manual:
        print(f"    Already done (cached). Use --redetect to redo.")
        status["status"] = "cached"
        return status

    img_rgb = np.array(Image.open(scan.path).convert("RGB"))

    landmarks = None
    method    = None
    rotation  = 0

    if not force_manual:
        landmarks, rotation = try_all_rotations(img_rgb, model_path)
        if landmarks is not None:
            method = "mediapipe"

    if landmarks is None:
        print(f"    Opening manual click tool...")
        landmarks = manual_click(scan)
        if landmarks is None:
            print(f"    Skipped (window closed without completing)")
            status["status"] = "skipped"
            return status
        method   = "manual"
        rotation = 0

    # Print what was found
    print(f"    Method: {method}")
    for name, (px, py) in landmarks.items():
        print(f"        {name:12s}: ({px}, {py})")

    # Save
    save_landmark_json(scan, landmarks, method, rotation, json_path)
    save_landmark_image(img_rgb, landmarks, img_path)
    print(f"    Saved: {json_path.name}")
    print(f"    Saved: {img_path.name}")

    status["status"] = "success"
    status["method"] = method
    return status


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Step 1: Detect face landmarks on all patient TIF images"
    )
    p.add_argument("--patient",  type=str, default=None,
                   help="Process only this patient (e.g. pat1)")
    p.add_argument("--manual",   type=str, default=None, metavar="SCAN_STEM",
                   help="Force manual click for this scan (e.g. pat1day0C)")
    p.add_argument("--redetect", action="store_true",
                   help="Re-run detection even if JSON already exists")
    p.add_argument("--dataset",  type=str, default=None,
                   help="Override dataset path")
    return p.parse_args()


def main():
    args = parse_args()
    global DATASET_DIR, OUTPUT_DIR
    if args.dataset:
        DATASET_DIR = Path(args.dataset)
        OUTPUT_DIR  = DATASET_DIR / "landmarks_raw"

    if not DATASET_DIR.exists():
        print(f"Dataset not found: {DATASET_DIR}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Step 1 — Face Landmark Detection")
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Output  : {OUTPUT_DIR}")
    print(f"{'='*60}")

    patients = discover_scans(DATASET_DIR, patient_filter=args.patient)
    if not patients:
        print("No TIF files found.")
        sys.exit(1)

    total = sum(len(v) for v in patients.values())
    print(f"\n  {len(patients)} patient(s), {total} scan(s) found")
    print(f"  Checking MediaPipe model...")
    model_path = ensure_model()

    report = []
    for patient_id, scans in patients.items():
        print(f"\n{'─'*60}")
        print(f"  Patient: {patient_id.upper()}  ({len(scans)} scans)")

        pat_out = OUTPUT_DIR / patient_id
        pat_out.mkdir(parents=True, exist_ok=True)

        for scan in scans:
            is_manual = bool(args.manual and
                             args.manual.lower() == scan.stem.lower())
            if args.redetect or is_manual:
                c = pat_out / f"{scan.stem}_landmarks.json"
                if c.exists(): c.unlink()

            status = process_scan(scan, model_path, pat_out,
                                  force_manual=is_manual)
            report.append(status)

    ok   = sum(1 for r in report if r["status"] in ("success", "cached"))
    fail = [r for r in report if r["status"] == "skipped"]

    print(f"\n{'='*60}")
    print(f"  DONE  {ok}/{len(report)} processed")
    if fail:
        print(f"  Skipped (re-run with --manual <scan>):")
        for r in fail:
            print(f"    {r['scan']}")
    print(f"\n  Output : {OUTPUT_DIR}")
    print(f"  Next   : run step2_align_images.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()