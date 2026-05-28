"""
face_landmark_align.py
======================
Pipeline:
  1. Detect 5 landmarks on the raw TIF (wherever the face is, any orientation)
  2. Compute a similarity transform that maps detected landmarks
     to fixed standard pixel positions (the "template")
  3. Apply transform -> output image where every landmark is always
     at the same pixel across all scans of the patient
  4. If MediaPipe fails -> interactive manual click fallback

The key insight: we do NOT try to rotate first. We detect wherever
landmarks are, then let the math (similarity transform) handle
rotation + scale + translation in one step.

Standard template positions (on 600x700 canvas):
  sellion   -> (300, 180)   nose bridge
  pronasale -> (300, 270)   nose tip
  menton    -> (300, 440)   chin
  l_tragion -> (145, 235)   left ear
  r_tragion -> (455, 235)   right ear

DATASET:
  Dataset/pat1day0C/pat1day0C.tif  etc.

OUTPUT:
  Dataset/landmark_aligned/pat1/
      pat1day0C_aligned.tif
      pat1day0C_aligned_preview.png
      pat1day0C_landmarks_debug.png   <- dots on original image
      pat1day0C_meta.json
  Dataset/landmark_aligned/alignment_report.json

USAGE:
  python face_landmark_align.py
  python face_landmark_align.py --patient pat1
  python face_landmark_align.py --patient pat1 --show
  python face_landmark_align.py --manual pat1day0C    <- manual click
  python face_landmark_align.py --redetect            <- clear cache

REQUIREMENTS:
  pip install numpy pillow opencv-python mediapipe matplotlib
  face_landmarker.task in same folder as this script
"""

import re, sys, json, argparse, urllib.request, warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  — edit DATASET_DIR if needed
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR = Path("D:/NahidW/Dataset")
OUTPUT_DIR  = DATASET_DIR / "landmark_aligned"

# MediaPipe 478-point model indices
LANDMARK_NAMES = {
    "sellion":   168,   # nose bridge / between eyes
    "pronasale": 4,     # nose tip
    "menton":    152,   # chin centre
    "l_tragion": 234,   # left ear
    "r_tragion": 454,   # right ear
}

# RGB colors for visualization
LM_COLOR = {
    "sellion":   (0,   220, 0),
    "pronasale": (220, 220, 0),
    "menton":    (220, 0,   220),
    "l_tragion": (0,   120, 255),
    "r_tragion": (255, 120, 0),
}

# ── Output canvas ──────────────────────────────────────────────────────────
OUT_W = 600
OUT_H = 700

# ── Standard landmark positions on the output canvas ──────────────────────
# Every aligned image will have these face points at exactly these pixels.
# Adjust these numbers if you want the face positioned differently.
TEMPLATE = {
    "sellion":   np.array([300, 180], dtype=np.float32),
    "pronasale": np.array([300, 270], dtype=np.float32),
    "menton":    np.array([300, 440], dtype=np.float32),
    "l_tragion": np.array([145, 235], dtype=np.float32),
    "r_tragion": np.array([455, 235], dtype=np.float32),
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
    for f in sorted(dataset_dir.glob("*/*.tif")):
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
#  MEDIAPIPE — detect on raw image, all 4 rotations, pick best
# ─────────────────────────────────────────────────────────────────────────────

def ensure_model() -> str:
    mp = Path(__file__).parent / MODEL_FILE
    if not mp.exists():
        print("  Downloading MediaPipe model (~30MB)...")
        urllib.request.urlretrieve(MODEL_URL, str(mp))
    return str(mp)


def _mediapipe_detect(img_rgb: np.ndarray, model_path: str) -> Optional[dict]:
    """Raw MediaPipe call. Returns {name: [px,py]} or None."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
    except ImportError:
        print("pip install mediapipe")
        sys.exit(1)

    H, W = img_rgb.shape[:2]
    opts = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    )
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    with vision.FaceLandmarker.create_from_options(opts) as det:
        res = det.detect(mp_img)

    if not res.face_landmarks:
        return None
    lms = res.face_landmarks[0]
    return {name: [int(lms[idx].x * W), int(lms[idx].y * H)]
            for name, idx in LANDMARK_NAMES.items()}


def _rotate_np(img: np.ndarray, deg: int) -> np.ndarray:
    """Rotate by 0/90/180/270 using PIL (lossless, handles non-square)."""
    return np.array(Image.fromarray(img).rotate(deg, expand=True))


def _is_upright(lm: dict) -> bool:
    """
    Sanity checks that confirm the face is in a normal upright orientation:
    1. sellion (nose bridge) Y < menton (chin) Y  — head is not upside down
    2. l_tragion X < r_tragion X                  — left/right not mirrored
    3. pronasale Y is between sellion and menton Y — nose tip in middle
    """
    sel = lm["sellion"]
    men = lm["menton"]
    pro = lm["pronasale"]
    lt  = lm["l_tragion"]
    rt  = lm["r_tragion"]

    head_upright   = sel[1] < men[1]
    not_mirrored   = lt[0] < rt[0]
    nose_between   = sel[1] < pro[1] < men[1]

    return head_upright and not_mirrored and nose_between


def detect_landmarks(img_rgb: np.ndarray, model_path: str):
    """
    Try detecting landmarks at 0, 90, 180, 270 degrees.
    For each rotation, run MediaPipe and validate orientation.
    Return (landmarks_in_ORIGINAL_image_coords, rotation_used)
    or (None, None) if all fail.

    Landmarks are always returned in ORIGINAL image coordinates
    (un-rotated), so the caller can work in the original image space.
    """
    H_orig, W_orig = img_rgb.shape[:2]

    print(f"    Detecting: ", end="", flush=True)
    for deg in [0, 90, 180, 270]:
        rotated = _rotate_np(img_rgb, deg)
        lm      = _mediapipe_detect(rotated, model_path)
        print(f"{deg}° ", end="", flush=True)

        if lm is None:
            continue
        if not _is_upright(lm):
            continue

        print(f"-> found at {deg}°")

        # Convert landmark coords from rotated image back to original image space
        Hr, Wr = rotated.shape[:2]
        lm_orig = {}
        for name, (px, py) in lm.items():
            if deg == 0:
                ox, oy = px, py
            elif deg == 90:
                # rotate(90) maps (x,y) -> (y, Wr-1-x) in rotated
                # inverse: orig_x = Wr-1-py_rot, orig_y = px_rot
                ox = Wr - 1 - py
                oy = px
            elif deg == 180:
                ox = Wr - 1 - px
                oy = Hr - 1 - py
            elif deg == 270:
                # rotate(270) maps (x,y) -> (Hr-1-y, x) in rotated
                # inverse: orig_x = py_rot, orig_y = Hr-1-px_rot
                ox = py
                oy = Hr - 1 - px
            lm_orig[name] = [int(ox), int(oy)]

        return lm_orig, deg

    print("-> all failed")
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSFORM — landmarks in original image -> standard canvas
# ─────────────────────────────────────────────────────────────────────────────

def compute_transform(landmarks: dict) -> Optional[np.ndarray]:
    """
    Compute a 2D similarity transform (rotation + uniform scale + translation)
    that maps detected landmark positions to TEMPLATE positions.

    Similarity transform preserves face proportions — no shear, no stretching.
    Uses all 5 landmarks, least-median-of-squares for robustness.

    Returns 2x3 matrix for cv2.warpAffine, or None.
    """
    src = np.array([landmarks[n]    for n in LANDMARK_NAMES], dtype=np.float32)
    dst = np.array([TEMPLATE[n]     for n in LANDMARK_NAMES], dtype=np.float32)

    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return None

    n_inliers = int(inliers.sum()) if inliers is not None else "?"
    print(f"    Transform: {n_inliers}/5 inliers")
    return M


def apply_transform(img_rgb: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Warp image to OUT_W x OUT_H canvas using computed transform."""
    return cv2.warpAffine(
        img_rgb, M, (OUT_W, OUT_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )


def transform_landmarks(landmarks: dict, M: np.ndarray) -> dict:
    """Apply 2x3 transform to landmark coords (for verification)."""
    out = {}
    for name, (px, py) in landmarks.items():
        pt  = np.array([px, py, 1.0], dtype=np.float64)
        res = M @ pt
        out[name] = [int(round(res[0])), int(round(res[1]))]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  MANUAL FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def manual_click(tif_path: Path):
    """
    Interactive: show the image, user clicks 5 landmarks in order.
    Returns (landmarks_dict, rotation=0) or (None, None).
    No rotation needed here — user works on the image as-is,
    and the similarity transform handles everything.
    """
    img    = np.array(Image.open(tif_path).convert("RGB"))
    order  = list(LANDMARK_NAMES.keys())
    state  = {"lm": {}, "i": 0}

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("#0d1117")
    ax.imshow(img)
    ax.set_title(
        f"MANUAL LANDMARK ENTRY — {tif_path.name}\n"
        f"Click on: {order[0].upper()}  (1/{len(order)})\n"
        "RIGHT CLICK = undo last point",
        color="white", fontsize=10
    )
    ax.axis("off")
    dots, lbls = [], []

    def refresh_title():
        i = state["i"]
        if i < len(order):
            ax.set_title(
                f"MANUAL — {tif_path.name}\n"
                f"Click on: {order[i].upper()}  ({i+1}/{len(order)})\n"
                "RIGHT CLICK = undo",
                color="white", fontsize=10
            )
        else:
            ax.set_title(
                "All landmarks placed!\nClose this window to continue.",
                color="#aaffaa", fontsize=11
            )
        fig.canvas.draw()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1 and state["i"] < len(order):
            name = order[state["i"]]
            px, py = int(event.xdata), int(event.ydata)
            state["lm"][name] = [px, py]
            clr = [c/255 for c in LM_COLOR[name]]
            d = ax.plot(px, py, 'o', color=clr, ms=10,
                        mec='white', mew=1.5)[0]
            l = ax.text(px+8, py-8, name, color=clr,
                        fontsize=8, fontweight='bold')
            dots.append(d); lbls.append(l)
            state["i"] += 1
            refresh_title()
        elif event.button == 3 and state["i"] > 0:  # undo
            state["i"] -= 1
            name = order[state["i"]]
            state["lm"].pop(name, None)
            dots.pop().remove(); lbls.pop().remove()
            refresh_title()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show(block=True)

    if len(state["lm"]) == len(order):
        return state["lm"], 0
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  DEBUG IMAGES
# ─────────────────────────────────────────────────────────────────────────────

def save_debug(img_rgb: np.ndarray, landmarks: dict, path: Path):
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for name, (px, py) in landmarks.items():
        r, g, b = LM_COLOR[name]
        cv2.circle(img, (px, py), 9, (b, g, r), -1)
        cv2.circle(img, (px, py), 10, (255,255,255), 1)
        cv2.putText(img, name, (px+10, py-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (b,g,r), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def overlay_template(img_rgb: np.ndarray) -> np.ndarray:
    """Draw template target positions as crosses on the aligned image."""
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for name, pos in TEMPLATE.items():
        r, g, b = LM_COLOR[name]
        cv2.drawMarker(img, (int(pos[0]), int(pos[1])), (b,g,r),
                       cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        cv2.putText(img, name, (int(pos[0])+6, int(pos[1])-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (b,g,r), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS ONE SCAN
# ─────────────────────────────────────────────────────────────────────────────

def process_scan(scan: ScanFile, model_path: str, out_dir: Path,
                 force_manual=False, show=False) -> dict:
    print(f"\n  ── {scan.path.name}  ({scan.label})")

    meta_path   = out_dir / f"{scan.stem}_meta.json"
    debug_path  = out_dir / f"{scan.stem}_landmarks_debug.png"
    aligned_tif = out_dir / f"{scan.stem}_aligned.tif"
    aligned_png = out_dir / f"{scan.stem}_aligned_preview.png"

    status = {"scan": scan.path.name, "patient": scan.patient_id,
              "day": scan.day, "variant": scan.variant,
              "status": "pending", "method": None}

    img_rgb   = np.array(Image.open(scan.path).convert("RGB"))
    landmarks = None
    rot_used  = None

    # ── 1. Get landmarks (cache or detect) ───────────────────────────────────
    if meta_path.exists() and not force_manual:
        with open(meta_path) as f:
            meta = json.load(f)
        landmarks = meta.get("landmarks")
        rot_used  = meta.get("rotation_detected", 0)
        print(f"    Cached landmarks (originally found at {rot_used}°)")
        status["method"] = "cached"
    else:
        if not force_manual:
            landmarks, rot_used = detect_landmarks(img_rgb, model_path)

        if landmarks is None:
            print(f"    Auto-detect failed -> opening manual tool")
            landmarks, rot_used = manual_click(scan.path)
            if landmarks is None:
                print(f"    Skipped")
                status["status"] = "skipped"
                return status
            status["method"] = "manual"
        else:
            status["method"] = "auto"

        # Save cache
        meta = {"landmarks": landmarks, "rotation_detected": rot_used,
                "scan": scan.path.name}
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        save_debug(img_rgb, landmarks, debug_path)
        print(f"    Landmarks (in original image):")
        for name, (px, py) in landmarks.items():
            tgt = TEMPLATE[name]
            print(f"        {name:12s}: detected=({px:4d},{py:4d})  "
                  f"-> target=({int(tgt[0])},{int(tgt[1])})")

    # ── 2. Compute similarity transform ──────────────────────────────────────
    M = compute_transform(landmarks)
    if M is None:
        print(f"    Transform failed")
        status["status"] = "transform_failed"
        return status

    # Verify: where do landmarks land after transform?
    lm_after = transform_landmarks(landmarks, M)
    print(f"    Verification (landmark positions after transform):")
    max_err = 0
    for name in LANDMARK_NAMES:
        ax_, ay_ = lm_after[name]
        tx, ty   = int(TEMPLATE[name][0]), int(TEMPLATE[name][1])
        err      = ((ax_-tx)**2 + (ay_-ty)**2)**0.5
        max_err  = max(max_err, err)
        print(f"        {name:12s}: got=({ax_:4d},{ay_:4d})  "
              f"target=({tx:4d},{ty:4d})  err={err:.1f}px")
    print(f"    Max error: {max_err:.1f}px")

    # ── 3. Apply transform ────────────────────────────────────────────────────
    aligned = apply_transform(img_rgb, M)
    Image.fromarray(aligned).save(str(aligned_tif), format="TIFF")
    Image.fromarray(aligned).save(str(aligned_png),  format="PNG")
    print(f"    Saved: {aligned_png.name}")
    status["status"] = "success"
    status["output"] = str(aligned_tif)

    # ── 4. Optional visual ────────────────────────────────────────────────────
    if show:
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        fig.patch.set_facecolor("#0d1117")
        fig.suptitle(f"{scan.path.name} — Alignment result", color="white", fontsize=13)

        # Panel 1: original with detected landmarks
        orig_vis = img_rgb.copy()
        for name, (px, py) in landmarks.items():
            r, g, b = LM_COLOR[name]
            cv2.circle(orig_vis, (px, py), 10, (r, g, b), -1)
            cv2.circle(orig_vis, (px, py), 11, (255,255,255), 1)
        axes[0].imshow(orig_vis)
        axes[0].set_title("Original + detected landmarks", color="#ffaaaa", fontsize=10)
        axes[0].axis("off")

        # Panel 2: aligned with template markers
        axes[1].imshow(overlay_template(aligned))
        axes[1].set_title(
            f"Aligned ({OUT_W}x{OUT_H})\nCross = template target position",
            color="#aaffaa", fontsize=10
        )
        axes[1].axis("off")

        # Panel 3: error chart
        names  = list(LANDMARK_NAMES.keys())
        errors = [((lm_after[n][0]-int(TEMPLATE[n][0]))**2 +
                   (lm_after[n][1]-int(TEMPLATE[n][1]))**2)**0.5
                  for n in names]
        ax3 = axes[2]
        ax3.set_facecolor("#161b22")
        bars = ax3.barh(names, errors,
                        color=[[c/255 for c in LM_COLOR[n]] for n in names])
        ax3.axvline(x=5, color='white', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_xlabel("Alignment error (pixels)", color="white")
        ax3.set_title("Per-landmark error\n(<5px = excellent)", color="white", fontsize=10)
        ax3.tick_params(colors="white")
        ax3.spines[:].set_color("#333344")
        for spine in ax3.spines.values():
            spine.set_color("#333344")

        plt.tight_layout()
        plt.show(block=True)

    return status


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--patient",  type=str, default=None,
                   help="Process only this patient (e.g. pat1)")
    p.add_argument("--manual",   type=str, default=None, metavar="SCAN_STEM",
                   help="Force manual landmark click for this scan")
    p.add_argument("--show",     action="store_true",
                   help="Show alignment result for each scan")
    p.add_argument("--redetect", action="store_true",
                   help="Ignore cache, re-run detection")
    p.add_argument("--dataset",  type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    global DATASET_DIR, OUTPUT_DIR
    if args.dataset:
        DATASET_DIR = Path(args.dataset)
        OUTPUT_DIR  = DATASET_DIR / "landmark_aligned"

    if not DATASET_DIR.exists():
        print(f"Dataset not found: {DATASET_DIR}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Face Landmark Alignment")
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Output  : {OUTPUT_DIR}")
    print(f"{'='*60}")

    patients = discover_scans(DATASET_DIR, patient_filter=args.patient)
    if not patients:
        print("No TIF files found.")
        sys.exit(1)

    total = sum(len(v) for v in patients.values())
    print(f"\n  {len(patients)} patient(s), {total} scan(s)")
    print(f"\n  Checking MediaPipe model...")
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
                c = pat_out / f"{scan.stem}_meta.json"
                if c.exists(): c.unlink()

            status = process_scan(scan, model_path, pat_out,
                                  force_manual=is_manual, show=args.show)
            report.append(status)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "alignment_report.json", "w") as f:
        json.dump(report, f, indent=2)

    ok   = sum(1 for r in report if r["status"] == "success")
    fail = [r for r in report if r["status"] != "success"]
    print(f"\n{'='*60}")
    print(f"  DONE  {ok}/{len(report)} aligned")
    if fail:
        print(f"  Failed:")
        for r in fail:
            print(f"    {r['scan']} — {r['status']}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()