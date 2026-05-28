#!/usr/bin/env python3
"""
generate_face_mesh.py
=====================
Batch apply MediaPipe 468-point face mesh to all patient TIF images.
Tries all 4 rotations (0/90/180/270) to find the correct face orientation.

READS FROM:
  Dataset/
    pat1/
        pat1day0C/
            pat1day0C.tif
        pat1day28A/
            pat1day28A.tif

WRITES TO:
  Dataset/face_mesh/
    pat1/
        pat1day0C_mesh.jpg       <- image with 468 dots drawn
        pat1day0C_mesh.json      <- all 468 landmark coords + rotation used
        pat1day28A_mesh.jpg
        pat1day28A_mesh.json
    face_mesh_report.json        <- summary: detected/failed per scan

USAGE:
  python generate_face_mesh.py
  python generate_face_mesh.py --patient pat1
  python generate_face_mesh.py --dataset "D:/NahidW/Dataset"
  python generate_face_mesh.py --overwrite

REQUIREMENTS:
  pip install mediapipe opencv-python numpy pillow
"""

import os, re, sys, json, argparse, warnings
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR = Path("D:/NahidW/Dataset")
OUTPUT_DIR  = DATASET_DIR / "face_mesh"

DOT_COLOR   = (0, 200, 100)   # BGR — green dots
DOT_RADIUS  = 2
ROTATIONS   = [0, 90, 180, 270]

# ─────────────────────────────────────────────────────────────────────────────
#  FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

SCAN_RE = re.compile(
    r"^(?P<patient>pat\d+)day(?P<day>\d+)(?P<variant>[A-Z][A-Z0-9]?)\.tif$",
    re.IGNORECASE
)


def discover_scans(dataset_dir: Path, patient_filter=None) -> list:
    scans = []
    for f in sorted(dataset_dir.glob("*/*/*.tif")):
        m = SCAN_RE.match(f.name)
        if not m:
            continue
        pid = m.group("patient").lower()
        if patient_filter and pid != patient_filter.lower():
            continue
        scans.append((pid, f.stem, f))
    scans.sort(key=lambda x: (x[0], x[1]))
    return scans


# ─────────────────────────────────────────────────────────────────────────────
#  FACE MESH DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def rotate_image_cv(img_bgr: np.ndarray, deg: int) -> np.ndarray:
    """Rotate BGR image by 0/90/180/270 degrees."""
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    rotated = pil.rotate(deg, expand=True)
    return cv2.cvtColor(np.array(rotated), cv2.COLOR_RGB2BGR)


def detect_face_mesh(img_bgr: np.ndarray, face_mesh_obj):
    """
    Run MediaPipe FaceMesh on a BGR image.
    Returns list of 468 landmarks as (x_px, y_px) or None if no face found.
    """
    h, w = img_bgr.shape[:2]
    rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh_obj.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = []
    for lm in result.multi_face_landmarks[0].landmark:
        landmarks.append((int(lm.x * w), int(lm.y * h)))
    return landmarks


def try_all_rotations(img_bgr: np.ndarray, face_mesh_obj):
    """
    Try 0/90/180/270 degree rotations.
    Validate upright orientation: landmark 10 (forehead) Y < landmark 152 (chin) Y.
    Returns (landmarks_in_original_coords, rotation_used) or (None, None).

    Landmark indices for orientation check (standard MediaPipe 468 model):
      10  = forehead top
      152 = chin / menton
    """
    print(f"    Trying rotations: ", end="", flush=True)

    for deg in ROTATIONS:
        rotated   = rotate_image_cv(img_bgr, deg)
        landmarks = detect_face_mesh(rotated, face_mesh_obj)
        print(f"{deg}° ", end="", flush=True)

        if landmarks is None:
            continue

        # Orientation check: forehead (10) should be above chin (152)
        forehead_y = landmarks[10][1]
        chin_y     = landmarks[152][1]
        if forehead_y >= chin_y:
            continue   # upside down or sideways — try next rotation

        print(f"-> detected at {deg}°")

        # Convert all 468 landmark coords back to ORIGINAL image space
        Hr, Wr = rotated.shape[:2]
        lm_orig = []
        for (px, py) in landmarks:
            if deg == 0:
                ox, oy = px, py
            elif deg == 90:
                ox = Wr - 1 - py
                oy = px
            elif deg == 180:
                ox = Wr - 1 - px
                oy = Hr - 1 - py
            elif deg == 270:
                ox = py
                oy = Hr - 1 - px
            lm_orig.append((int(ox), int(oy)))

        return lm_orig, deg

    print(f"-> all failed")
    return None, None


def draw_mesh(img_bgr: np.ndarray, landmarks: list) -> np.ndarray:
    """Draw 468 dots on the image."""
    out = img_bgr.copy()
    for (x, y) in landmarks:
        cv2.circle(out, (x, y), DOT_RADIUS, DOT_COLOR, -1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS ONE SCAN
# ─────────────────────────────────────────────────────────────────────────────

def process_scan(patient_id: str, scan_name: str, tif_path: Path,
                 face_mesh_obj, out_dir: Path,
                 overwrite: bool = False) -> dict:

    out_jpg  = out_dir / f"{scan_name}_mesh.jpg"
    out_json = out_dir / f"{scan_name}_mesh.json"

    if out_jpg.exists() and out_json.exists() and not overwrite:
        print(f"  ↷ {scan_name}  already done — skipping")
        return {"scan": scan_name, "status": "skipped"}

    print(f"  → {scan_name}")

    img_bgr = cv2.imread(str(tif_path))
    if img_bgr is None:
        print(f"    ✗ Could not load image")
        return {"scan": scan_name, "status": "load_failed"}

    # Detect
    landmarks, rotation = try_all_rotations(img_bgr, face_mesh_obj)

    if landmarks is None:
        print(f"    ✗ No face detected in any rotation")
        return {"scan": scan_name, "status": "no_face"}

    # Draw dots on original (unrotated) image
    mesh_img = draw_mesh(img_bgr, landmarks)
    cv2.imwrite(str(out_jpg), mesh_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Save JSON with all 468 coords
    h, w = img_bgr.shape[:2]
    data = {
        "scan":       scan_name,
        "patient":    patient_id,
        "image_w":    w,
        "image_h":    h,
        "rotation":   rotation,
        "n_landmarks": len(landmarks),
        "landmarks":  landmarks   # list of [x, y] for indices 0..467
    }
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"    ✓  {len(landmarks)} landmarks  |  rotation={rotation}°")
    print(f"       {out_jpg.name}")

    return {"scan": scan_name, "status": "success", "rotation": rotation}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch generate 468-point face mesh for all patient TIF images"
    )
    p.add_argument("--patient",   type=str, default=None,
                   help="Process only this patient (e.g. pat1)")
    p.add_argument("--dataset",   type=str, default=None,
                   help="Override dataset directory")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing outputs (default: skip)")
    return p.parse_args()


def main():
    args = parse_args()
    global DATASET_DIR, OUTPUT_DIR
    if args.dataset:
        DATASET_DIR = Path(args.dataset)
        OUTPUT_DIR  = DATASET_DIR / "face_mesh"

    if not DATASET_DIR.exists():
        print(f"Dataset not found: {DATASET_DIR}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Face Mesh Generator (468 landmarks)")
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Output  : {OUTPUT_DIR}")
    print(f"{'='*55}\n")

    scans = discover_scans(DATASET_DIR, patient_filter=args.patient)
    if not scans:
        print("No TIF files found.")
        sys.exit(1)

    print(f"  Found {len(scans)} scan(s)\n")

    # Initialise MediaPipe FaceMesh once — reuse for all images
    try:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
    except ImportError:
        print("Run: pip install mediapipe")
        sys.exit(1)

    face_mesh_obj = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.05,   # low threshold for burned/cylindrical images
        min_tracking_confidence=0.05,
    )

    report   = []
    prev_pat = None

    try:
        for patient_id, scan_name, tif_path in scans:

            if patient_id != prev_pat:
                print(f"{'─'*55}")
                print(f"  Patient: {patient_id.upper()}")
                prev_pat = patient_id

            pat_out = OUTPUT_DIR / patient_id
            pat_out.mkdir(parents=True, exist_ok=True)

            status = process_scan(
                patient_id, scan_name, tif_path,
                face_mesh_obj, pat_out,
                overwrite=args.overwrite
            )
            report.append(status)

    finally:
        face_mesh_obj.close()

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "face_mesh_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    ok     = sum(1 for r in report if r["status"] == "success")
    skip   = sum(1 for r in report if r["status"] == "skipped")
    failed = [r for r in report if r["status"] not in ("success", "skipped")]

    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Detected : {ok}")
    print(f"  Skipped  : {skip}")
    if failed:
        print(f"  Failed   : {len(failed)}")
        for r in failed:
            print(f"    {r['scan']} — {r['status']}")
    print(f"  Output   : {OUTPUT_DIR}")
    print(f"  Report   : {report_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()