#!/usr/bin/env python3
"""
frankfurt_step2_align.py
========================
Frankfurt Plane alignment for all patients.

READS:
    Dataset/frankfurt_aligned/pat1/*_icp_front.png   <- from step1
    Dataset/3d_scans/ply/pat1/*_icp.ply              <- ICP-aligned PLY

WRITES:
    Dataset/frankfurt_aligned/pat1/
        *_debug.png          <- landmarks drawn on front image
        *_frankfurt.ply      <- Frankfurt-aligned PLY
        *_transform.npy      <- 4x4 transform matrix

COORDINATE SYSTEM (ICP-aligned files):
    Y = up, face points toward +Z
    Camera for render: front=[0,0,1], up=[0,1,0]
    Back-projection uses matching axes.

USAGE:
    python frankfurt_step2_align.py
    python frankfurt_step2_align.py --patient pat1
    python frankfurt_step2_align.py --root D:/NahidW
    python frankfurt_step2_align.py --overwrite
"""

import sys, re, json, argparse, urllib.request
import numpy as np
import cv2
import open3d as o3d
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pathlib import Path


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
ZOOM  = 0.55       # must match step1
IMG_W = 800
IMG_H = 1000

# MediaPipe landmark indices
LM_SELLION   = 168
LM_L_TRAGION = 234
LM_R_TRAGION = 454
LM_PRONASALE = 4
LM_MENTON    = 152

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_FILE = "face_landmarker.task"


# ─────────────────────────────────────────────
#  AUTO-DETECT ROOT
# ─────────────────────────────────────────────
def find_nahidw_root(script_path: Path) -> Path:
    candidate = script_path.resolve().parent
    while True:
        if (candidate / "Dataset").is_dir():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(
                "Cannot find 'Dataset' folder. Use --root D:/NahidW")
        candidate = parent


# ─────────────────────────────────────────────
#  DISCOVER — pair front.png with its _icp.ply
# ─────────────────────────────────────────────
def discover_scans(fa_root: Path, ply_root: Path,
                   patient_filter=None) -> list:
    """
    Finds all *_oriented_icp_front.png in frankfurt_aligned/patX/
    and pairs each with its *_oriented_icp.ply in 3d_scans/ply/patX/.
    These are the pre-oriented + ICP-aligned files — not raw originals.
    """
    scans = []
    for pat_dir in sorted(fa_root.iterdir()):
        if not pat_dir.is_dir():
            continue
        pid = pat_dir.name
        if patient_filter and pid.lower() != patient_filter.lower():
            continue
        for png in sorted(pat_dir.glob("*_oriented_icp_front.png")):
            # stem of PLY = stem of PNG minus "_front"
            ply_stem = png.stem.replace("_front", "")
            ply_path = ply_root / pid / f"{ply_stem}.ply"
            if not ply_path.exists():
                print(f"  ⚠ PLY not found: {ply_path}")
                continue
            scans.append((pid, ply_stem, png, ply_path))
    scans.sort(key=lambda x: (x[0], x[1]))
    return scans


# ─────────────────────────────────────────────
#  MEDIAPIPE MODEL
# ─────────────────────────────────────────────
def ensure_model(script_dir: Path) -> str:
    model_path = script_dir / MODEL_FILE
    if not model_path.exists():
        print(f"  Downloading MediaPipe model (~30 MB)...")
        urllib.request.urlretrieve(MODEL_URL, str(model_path))
        print(f"  Saved: {model_path}")
    return str(model_path)


# ─────────────────────────────────────────────
#  DETECT 2D LANDMARKS
# ─────────────────────────────────────────────
def detect_landmarks_2d(image_path: Path, model_path: str):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W     = img_bgr.shape[:2]

    options  = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    )
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    with vision.FaceLandmarker.create_from_options(options) as det:
        result = det.detect(mp_image)

    if not result.face_landmarks:
        return None, None, img_bgr, H, W

    lms = result.face_landmarks[0]
    landmarks_px = {}
    for name, idx in [("sellion",   LM_SELLION),
                       ("l_tragion", LM_L_TRAGION),
                       ("r_tragion", LM_R_TRAGION),
                       ("pronasale", LM_PRONASALE),
                       ("menton",    LM_MENTON)]:
        lm = lms[idx]
        landmarks_px[name] = (int(lm.x * W), int(lm.y * H))

    return landmarks_px, lms, img_bgr, H, W


# ─────────────────────────────────────────────
#  BACK-PROJECTION  (2D pixel → 3D point)
#
#  Matches the render in step1 exactly:
#    camera front = [0, 0, 1]  (camera at +Z)
#    up           = [0, 1, 0]  (Y is up)
#
#  Derived axes:
#    view_dir = -front = [0, 0, -1]
#    right    = cross(view_dir, up) = [1, 0, 0]
#    screen_y increases downward in image → flip Y
#
#  Projection:
#    screen_x = (pt - centroid) · right  =  Δx
#    screen_y = (pt - centroid) · (-up)  = -Δy
#    pixel_x  = screen_x/scale * W + W/2
#    pixel_y  = screen_y/scale * H + H/2
# ─────────────────────────────────────────────
def project_to_pixels(pts: np.ndarray,
                       centroid: np.ndarray) -> np.ndarray:
    scale    = ZOOM * min(IMG_W, IMG_H)
    centered = pts - centroid
    px_x     = ( centered[:, 0] / scale * IMG_W + IMG_W / 2).astype(int)
    px_y     = (-centered[:, 1] / scale * IMG_H + IMG_H / 2).astype(int)
    return np.stack([px_x, px_y], axis=1)


def backproject_to_3d(landmarks_px: dict,
                       pts: np.ndarray,
                       centroid: np.ndarray) -> dict:
    proj         = project_to_pixels(pts, centroid)
    landmarks_3d = {}
    for name, (px, py) in landmarks_px.items():
        dists = (proj[:, 0] - px) ** 2 + (proj[:, 1] - py) ** 2
        idx   = np.argmin(dists)
        dist  = np.sqrt(dists[idx])
        if dist > 30:
            print(f"    ⚠  {name}: nearest 3D point {dist:.1f}px away")
        landmarks_3d[name] = pts[idx].copy()
    return landmarks_3d


# ─────────────────────────────────────────────
#  FRANKFURT TRANSFORM
#  Modified Frankfurt Plane:
#    l_tragion + r_tragion + sellion
#  Origin  = midpoint of tragions
#  X axis  = right → left  (R tragion → L tragion)
#  Y axis  = superior (up)
#  Z axis  = anterior (forward, toward face)
# ─────────────────────────────────────────────
def compute_frankfurt_transform(landmarks_3d: dict):
    l_t = landmarks_3d["l_tragion"]
    r_t = landmarks_3d["r_tragion"]
    sel = landmarks_3d["sellion"]

    origin = (l_t + r_t) / 2.0

    # Y axis = Frankfurt plane normal (must point UP, +Y)
    # Plane defined by 3 points: L_tragion, R_tragion, sellion
    plane_n = np.cross(l_t - r_t, sel - r_t)
    plane_n /= np.linalg.norm(plane_n)
    if plane_n[1] < 0:       # ensure superior (+Y) direction
        plane_n = -plane_n
    y_axis = plane_n

    # X axis = interaural line, orthogonalised against Y
    x_axis = l_t - r_t
    x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
    x_axis /= np.linalg.norm(x_axis)

    # Z axis = X × Y, must point forward (+Z, toward face)
    z_axis = np.cross(x_axis, y_axis)
    if z_axis[2] < 0:        # flip X and recompute Z if backward
        x_axis = -x_axis
        z_axis = np.cross(x_axis, y_axis)

    R         = np.stack([x_axis, y_axis, z_axis], axis=0)
    T         = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = -R @ origin
    return T


def validate_transform(T: np.ndarray, landmarks_3d: dict) -> bool:
    def tp(pt):
        return (T @ np.append(pt, 1.0))[:3]

    mid = tp((landmarks_3d["l_tragion"] + landmarks_3d["r_tragion"]) / 2)
    l_t = tp(landmarks_3d["l_tragion"])
    r_t = tp(landmarks_3d["r_tragion"])
    sel = tp(landmarks_3d["sellion"])

    z_diff = abs(l_t[2] - r_t[2])
    ok     = np.linalg.norm(mid) < 5.0 and z_diff < 5.0 and sel[2] > 0

    print(f"    Origin at      : {mid.round(2)}  (expect ~[0,0,0])")
    print(f"    Sellion Z      : {sel[2]:.2f}  (expect > 0)")
    print(f"    Tragion Z diff : {z_diff:.2f} mm  (expect ~0)")
    print(f"    Status         : {'✓ PASSED' if ok else '⚠ CHECK MANUALLY'}")
    return ok


# ─────────────────────────────────────────────
#  DEBUG IMAGE
# ─────────────────────────────────────────────
def save_debug_image(img_bgr: np.ndarray, lms_raw,
                     H: int, W: int, out_path: Path):
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
        cv2.putText(debug, name, (px + 8, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(str(out_path), debug)


# ─────────────────────────────────────────────
#  PROCESS ONE SCAN
# ─────────────────────────────────────────────
def process_scan(pid: str, stem: str,
                 front_path: Path, ply_path: Path,
                 model_path: str, out_dir: Path,
                 overwrite: bool) -> dict:

    out_ply   = out_dir / f"{stem}_frankfurt.ply"
    out_debug = out_dir / f"{stem}_debug.png"  # temp, deleted after

    if out_ply.exists() and not overwrite:
        print(f"    skip (exists): {out_ply.name}")
        return {"scan": stem, "status": "skipped"}

    print(f"\n  ── {stem}")

    # 1. Detect 2D landmarks
    print(f"    Detecting landmarks...")
    landmarks_px, lms_raw, img_bgr, img_H, img_W = \
        detect_landmarks_2d(front_path, model_path)

    if landmarks_px is None:
        print(f"    ✗ No face detected in front image")
        return {"scan": stem, "status": "no_face"}

    for name, (px, py) in landmarks_px.items():
        print(f"        {name:12s}: pixel ({px}, {py})")

    save_debug_image(img_bgr, lms_raw, img_H, img_W, out_debug)

    # 2. Load PLY
    pcd      = o3d.io.read_point_cloud(str(ply_path))
    pts      = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)

    # 3. Back-project pixels → 3D
    print(f"    Back-projecting to 3D...")
    landmarks_3d = backproject_to_3d(landmarks_px, pts, centroid)
    for name, pt in landmarks_3d.items():
        print(f"        {name:12s}: [{pt[0]:7.2f}, {pt[1]:7.2f}, {pt[2]:7.2f}] mm")

    # 4. Frankfurt transform
    print(f"    Computing Frankfurt transform...")
    T = compute_frankfurt_transform(landmarks_3d)

    # 5. Validate
    print(f"    Validation:")
    ok = validate_transform(T, landmarks_3d)

    # 6. Apply + save
    pcd.transform(T)
    o3d.io.write_point_cloud(str(out_ply), pcd)

    # Remove debug image — only keep the frankfurt PLY
    if out_debug.exists():
        out_debug.unlink()

    print(f"    ✓ Frankfurt PLY  → {out_ply.name}  "
          f"({out_ply.stat().st_size/1e6:.1f} MB)")

    return {"scan": stem, "status": "success",
            "validation_passed": bool(ok), "output": str(out_ply)}


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Frankfurt Step 2: landmark detection + plane alignment")
    parser.add_argument("--root",      default=None)
    parser.add_argument("--patient",   default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.root:
        root = Path(args.root).expanduser().resolve()
        print(f"Root (manual): {root}")
    else:
        root = find_nahidw_root(Path(__file__))
        print(f"Root (auto-detected): {root}")

    ply_root = root / "Dataset" / "3d_scans" / "ply"
    fa_root  = root / "Dataset" / "frankfurt_aligned"

    if not fa_root.exists():
        print(f"frankfurt_aligned/ not found. Run step1 first.")
        return

    print(f"\n{'='*55}")
    print(f"  Frankfurt Step 2 — Plane Alignment")
    print(f"  Front images : {fa_root}")
    print(f"  PLY input    : {ply_root}")
    print(f"  Output       : {fa_root}")
    print(f"{'='*55}")

    model_path = ensure_model(Path(__file__).parent)

    scans = discover_scans(fa_root, ply_root, patient_filter=args.patient)
    if not scans:
        print("No *_icp_front.png files found. Run step1 first.")
        return

    print(f"\nFound {len(scans)} scan(s)\n")

    report   = []
    prev_pat = None
    for pid, stem, front_path, ply_path in scans:
        if pid != prev_pat:
            print(f"{'─'*55}")
            print(f"  Patient: {pid}")
            prev_pat = pid
        out_dir = fa_root / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        r = process_scan(pid, stem, front_path, ply_path,
                         model_path, out_dir, overwrite=args.overwrite)
        report.append(r)

    with open(fa_root / "alignment_report.json", "w") as f:
        json.dump(report, f, indent=2)

    ok     = sum(1 for r in report if r["status"] == "success")
    skip   = sum(1 for r in report if r["status"] == "skipped")
    passed = sum(1 for r in report if r.get("validation_passed"))
    failed = [r for r in report if r["status"] not in ("success", "skipped")]

    print(f"\n{'='*55}")
    print(f"  Done!")
    print(f"  Aligned         : {ok}")
    print(f"  Validation pass : {passed}/{ok}")
    print(f"  Skipped         : {skip}")
    if failed:
        print(f"  Failed          : {len(failed)}")
        for r in failed:
            print(f"    {r['scan']} — {r['status']}")
    print(f"\n  Output : {fa_root}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()