#!/usr/bin/env python3
"""
frankfurt_step2_align.py
========================
Batch Frankfurt Plane alignment for all patients.
Reads front images + PLY files, detects landmarks, computes
Modified Frankfurt Plane transform, saves aligned PLY.

READS FROM:
  Dataset/frankfurt_aligned/pat1/pat1day0C_front.png  <- from step1
  Dataset/ply_files/pat1/pat1day0C.ply                <- original PLY

WRITES TO:
  Dataset/frankfurt_aligned/
    pat1/
        pat1day0C_front.png         <- already here from step1
        pat1day0C_debug.png         <- landmarks drawn on front image
        pat1day0C_aligned.ply       <- Frankfurt-aligned PLY
        pat1day0C_transform.npy     <- 4x4 transform matrix

USAGE:
  python frankfurt_step2_align.py
  python frankfurt_step2_align.py --patient pat1
  python frankfurt_step2_align.py --dataset "D:/NahidW/Dataset"
  python frankfurt_step2_align.py --overwrite

REQUIREMENTS:
  pip install numpy open3d mediapipe opencv-python
"""

import sys, os, re, json, argparse, urllib.request
import numpy as np
import cv2
import open3d as o3d
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR = Path("D:/NahidW/Dataset")
PLY_DIR     = DATASET_DIR / "ply_files"
OUTPUT_DIR  = DATASET_DIR / "frankfurt_aligned"

# Camera params — must match step1 exactly
YAW_DEG = 15.0
ZOOM    = 0.55
IMG_W   = 800
IMG_H   = 1000

# MediaPipe landmark indices
LM_SELLION   = 168
LM_L_TRAGION = 234
LM_R_TRAGION = 454
LM_PRONASALE = 4
LM_MENTON    = 152

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_FILE = "face_landmarker.task"

# ─────────────────────────────────────────────────────────────────────────────
#  FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

SCAN_RE = re.compile(
    r"^(?P<patient>pat\d+)day(?P<day>\d+)(?P<variant>[A-Z][A-Z0-9]?)_front\.png$",
    re.IGNORECASE
)


def discover_front_images(output_dir: Path, ply_dir: Path,
                           patient_filter=None) -> list:
    """
    Find all _front.png files in frankfurt_aligned/ and pair with PLY files.
    Returns list of (patient_id, scan_name, front_png_path, ply_path).
    """
    scans = []
    for f in sorted(output_dir.glob("*/*_front.png")):
        m = SCAN_RE.match(f.name)
        if not m:
            continue
        pid       = m.group("patient").lower()
        scan_name = f.name.replace("_front.png", "")
        if patient_filter and pid != patient_filter.lower():
            continue

        ply_path = ply_dir / pid / f"{scan_name}.ply"
        if not ply_path.exists():
            print(f"  ⚠ PLY not found for {scan_name}: {ply_path}")
            continue

        scans.append((pid, scan_name, f, ply_path))

    scans.sort(key=lambda x: (x[0], x[1]))
    return scans


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


def detect_landmarks_2d(image_path: Path, model_path: str):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W    = img_bgr.shape[:2]

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    )
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    with vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    if not result.face_landmarks:
        return None, None, img_bgr, H, W

    lms = result.face_landmarks[0]
    landmarks_px = {}
    for name, idx in [
        ("sellion",   LM_SELLION),
        ("l_tragion", LM_L_TRAGION),
        ("r_tragion", LM_R_TRAGION),
        ("pronasale", LM_PRONASALE),
        ("menton",    LM_MENTON),
    ]:
        lm = lms[idx]
        landmarks_px[name] = (int(lm.x * W), int(lm.y * H))

    return landmarks_px, lms, img_bgr, H, W


# ─────────────────────────────────────────────────────────────────────────────
#  3D BACK-PROJECTION + FRANKFURT TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────

def project_points(pts: np.ndarray, centroid: np.ndarray,
                   front_vec=None, zoom=None, img_w=None, img_h=None) -> np.ndarray:
    """
    Project 3D points to 2D using the same camera that step1 used.
    If render_meta is available (front_vec etc), uses those exact params.
    Falls back to default YAW_DEG if not.
    """
    if front_vec is None:
        yaw_rad   = np.radians(-YAW_DEG)
        front_vec = np.array([-np.cos(yaw_rad), -np.sin(yaw_rad), 0.0])
    if zoom   is None: zoom   = ZOOM
    if img_w  is None: img_w  = IMG_W
    if img_h  is None: img_h  = IMG_H

    front    = np.array(front_vec, dtype=np.float64)
    front   /= np.linalg.norm(front)
    world_up = np.array([0.0, 0.0, 1.0])
    right    = np.cross(front, world_up);  right /= np.linalg.norm(right)
    up       = np.cross(right, front);     up    /= np.linalg.norm(up)

    centered = pts - centroid
    screen_x = centered @ right
    screen_y = -centered @ up
    scale    = zoom * min(img_w, img_h)
    pixel_x  = (screen_x / scale * img_w + img_w / 2).astype(int)
    pixel_y  = (screen_y / scale * img_h + img_h / 2).astype(int)
    return np.stack([pixel_x, pixel_y], axis=1)


def backproject_to_3d(landmarks_px: dict, pts: np.ndarray,
                       centroid: np.ndarray, render_meta: dict = {}) -> dict:
    proj = project_points(
        pts, centroid,
        front_vec = render_meta.get("front_vec"),
        zoom      = render_meta.get("zoom"),
        img_w     = render_meta.get("img_w"),
        img_h     = render_meta.get("img_h"),
    )
    landmarks_3d = {}
    for name, (px, py) in landmarks_px.items():
        dists       = (proj[:, 0] - px) ** 2 + (proj[:, 1] - py) ** 2
        nearest_idx = np.argmin(dists)
        dist_px     = np.sqrt(dists[nearest_idx])
        if dist_px > 30:
            print(f"    ⚠ {name}: nearest 3D point is {dist_px:.1f}px away")
        landmarks_3d[name] = pts[nearest_idx].copy()
    return landmarks_3d


def compute_frankfurt_transform(landmarks_3d: dict):
    l_t = landmarks_3d["l_tragion"]
    r_t = landmarks_3d["r_tragion"]
    sel = landmarks_3d["sellion"]

    origin  = (l_t + r_t) / 2.0
    x_axis  = l_t - r_t;                x_axis /= np.linalg.norm(x_axis)
    temp_up = sel - origin;              temp_up /= np.linalg.norm(temp_up)
    y_axis  = np.cross(temp_up, x_axis); y_axis  /= np.linalg.norm(y_axis)
    z_axis  = np.cross(x_axis, y_axis);  z_axis  /= np.linalg.norm(z_axis)

    if np.dot(z_axis, temp_up) < 0:
        z_axis = -z_axis
        y_axis = np.cross(x_axis, z_axis)
        y_axis /= np.linalg.norm(y_axis)

    R         = np.stack([x_axis, y_axis, z_axis], axis=0)
    T         = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = -R @ origin
    return T, origin, x_axis, y_axis, z_axis


def validate_transform(T: np.ndarray, landmarks_3d: dict) -> bool:
    def tp(pt):
        return (T @ np.array([*pt, 1.0]))[:3]

    mid = tp((landmarks_3d["l_tragion"] + landmarks_3d["r_tragion"]) / 2)
    l_t = tp(landmarks_3d["l_tragion"])
    r_t = tp(landmarks_3d["r_tragion"])
    sel = tp(landmarks_3d["sellion"])

    z_diff = abs(l_t[2] - r_t[2])
    ok     = np.linalg.norm(mid) < 2.0 and z_diff < 5.0 and sel[2] > 0

    print(f"    Origin at    : {mid.round(2)}  (expect ~[0,0,0])")
    print(f"    Sellion Z    : {sel[2]:.2f}  (expect > 0)")
    print(f"    Tragion Z diff : {z_diff:.3f} mm  (expect ~0)")
    print(f"    Status       : {'✓ PASSED' if ok else '⚠ CHECK MANUALLY'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
#  DEBUG IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def save_debug_image(img_bgr: np.ndarray, lms_raw, H: int, W: int,
                      out_path: Path):
    debug = img_bgr.copy()
    for name, idx, color in [
        ("Sellion",   LM_SELLION,   (0,   255, 0)),
        ("L_Tragion", LM_L_TRAGION, (255, 100, 0)),
        ("R_Tragion", LM_R_TRAGION, (0,   100, 255)),
        ("Pronasale", LM_PRONASALE, (0,   255, 255)),
        ("Menton",    LM_MENTON,    (255, 0,   255)),
    ]:
        lm = lms_raw[idx]
        px = int(lm.x * W)
        py = int(lm.y * H)
        cv2.circle(debug, (px, py), 10, color, -1)
        cv2.circle(debug, (px, py), 11, (255, 255, 255), 1)
        cv2.putText(debug, name, (px+8, py-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(str(out_path), debug)


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS ONE SCAN
# ─────────────────────────────────────────────────────────────────────────────

def process_scan(patient_id: str, scan_name: str,
                 front_path: Path, ply_path: Path,
                 model_path: str, out_dir: Path,
                 overwrite: bool = False) -> dict:

    out_aligned   = out_dir / f"{scan_name}_aligned.ply"
    out_transform = out_dir / f"{scan_name}_transform.npy"
    out_debug     = out_dir / f"{scan_name}_debug.png"

    if out_aligned.exists() and not overwrite:
        print(f"  ↷ {scan_name}  already aligned — skipping")
        return {"scan": scan_name, "status": "skipped"}

    print(f"\n  ── {scan_name}")

    # 1. Detect 2D landmarks on front image
    print(f"    Detecting landmarks...")
    landmarks_px, lms_raw, img_bgr, img_H, img_W = \
        detect_landmarks_2d(front_path, model_path)

    if landmarks_px is None:
        # Try rotating the front image — face may be tilted in the render
        print(f"    No face at 0° — trying rotations...")
        from PIL import Image as PILImage
        img_orig = PILImage.open(str(front_path))
        for rot_deg in [90, 180, 270]:
            rotated     = np.array(img_orig.rotate(rot_deg, expand=True))
            rotated_bgr = cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)
            tmp_path    = front_path.parent / f"_tmp_rot{rot_deg}.png"
            cv2.imwrite(str(tmp_path), rotated_bgr)
            landmarks_px, lms_raw, img_bgr, img_H, img_W =                 detect_landmarks_2d(tmp_path, model_path)
            tmp_path.unlink()
            if landmarks_px is not None:
                print(f"    Found face at {rot_deg}° rotation")
                break
    if landmarks_px is None:
        print(f"    ✗ No face detected in any rotation")
        return {"scan": scan_name, "status": "no_face"}

    for name, (px, py) in landmarks_px.items():
        print(f"        {name:12s}: ({px}, {py})")

    save_debug_image(img_bgr, lms_raw, img_H, img_W, out_debug)

    # 2. Load PLY
    pcd      = o3d.io.read_point_cloud(str(ply_path))
    pts      = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)

    # Load render meta from step1 (exact camera params used for this scan)
    import json
    meta_path = front_path.parent / front_path.name.replace("_front.png", "_render_meta.json")
    render_meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            render_meta = json.load(f)
        print(f"    Using render meta: front_vec={[round(v,3) for v in render_meta['front_vec']]}")
    else:
        print(f"    No render meta found — using default camera params")

    # 3. Back-project to 3D
    print(f"    Back-projecting to 3D...")
    landmarks_3d = backproject_to_3d(landmarks_px, pts, centroid,
                                      render_meta=render_meta)
    for name, pt in landmarks_3d.items():
        print(f"        {name:12s}: [{pt[0]:7.2f}, {pt[1]:7.2f}, {pt[2]:7.2f}] mm")

    # 4. Compute Frankfurt transform
    print(f"    Computing Frankfurt transform...")
    T, origin, x_axis, y_axis, z_axis = compute_frankfurt_transform(landmarks_3d)

    # 5. Validate
    print(f"    Validation:")
    ok = validate_transform(T, landmarks_3d)

    # 6. Apply and save
    pcd.transform(T)
    o3d.io.write_point_cloud(str(out_aligned), pcd)
    np.save(str(out_transform), T)

    size_mb = out_aligned.stat().st_size / 1e6
    print(f"    ✓ Aligned PLY  : {out_aligned.name}  ({size_mb:.1f} MB)")
    print(f"    ✓ Transform    : {out_transform.name}")
    print(f"    ✓ Debug image  : {out_debug.name}")

    return {"scan": scan_name, "status": "success",
            "validation_passed": ok, "output": str(out_aligned)}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Step 2: Frankfurt Plane alignment for all patients"
    )
    p.add_argument("--patient",   type=str, default=None)
    p.add_argument("--dataset",   type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    global DATASET_DIR, PLY_DIR, OUTPUT_DIR
    if args.dataset:
        DATASET_DIR = Path(args.dataset)
        PLY_DIR     = DATASET_DIR / "ply_files"
        OUTPUT_DIR  = DATASET_DIR / "frankfurt_aligned"

    if not OUTPUT_DIR.exists():
        print(f"frankfurt_aligned/ not found: {OUTPUT_DIR}")
        print("Run frankfurt_step1_render.py first.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Frankfurt Step 2 — Plane Alignment")
    print(f"  Front images : {OUTPUT_DIR}")
    print(f"  PLY input    : {PLY_DIR}")
    print(f"  Output       : {OUTPUT_DIR}")
    print(f"{'='*55}")

    print(f"\n  Checking MediaPipe model...")
    model_path = ensure_model()

    scans = discover_front_images(OUTPUT_DIR, PLY_DIR,
                                   patient_filter=args.patient)
    if not scans:
        print("No _front.png files found. Run frankfurt_step1_render.py first.")
        sys.exit(1)

    print(f"\n  Found {len(scans)} scan(s)\n")

    report   = []
    prev_pat = None

    for patient_id, scan_name, front_path, ply_path in scans:
        if patient_id != prev_pat:
            print(f"{'─'*55}")
            print(f"  Patient: {patient_id.upper()}")
            prev_pat = patient_id

        pat_out = OUTPUT_DIR / patient_id
        pat_out.mkdir(parents=True, exist_ok=True)

        status = process_scan(
            patient_id, scan_name, front_path, ply_path,
            model_path, pat_out, overwrite=args.overwrite
        )
        report.append(status)

    # Save report
    with open(OUTPUT_DIR / "alignment_report.json", "w") as f:
        import json
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.bool_):   return bool(obj)
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super().default(obj)
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    ok       = sum(1 for r in report if r["status"] == "success")
    skip     = sum(1 for r in report if r["status"] == "skipped")
    passed   = sum(1 for r in report if r.get("validation_passed"))
    failed   = [r for r in report if r["status"] not in ("success","skipped")]

    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Aligned         : {ok}")
    print(f"  Validation pass : {passed}/{ok}")
    print(f"  Skipped         : {skip}")
    if failed:
        print(f"  Failed          : {len(failed)}")
        for r in failed:
            print(f"    {r['scan']} — {r['status']}")
    print(f"\n  Output : {OUTPUT_DIR}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()