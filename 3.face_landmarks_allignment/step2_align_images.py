"""
step2_align_images.py
=====================
Reads landmark JSONs from step1 and aligns every image so that
all 5 landmarks land at fixed standard pixel positions.

HOW IT WORKS:
  For each scan:
    1. Load the _landmarks.json from step1
    2. From the landmark positions, figure out the correct face orientation:
         - sellion should be ABOVE menton (Y axis: sellion_y < menton_y)
         - l_tragion should be LEFT of r_tragion (X axis: lt_x < rt_x)
         - sellion should be between the two tragions horizontally
         - pronasale should be between sellion and menton vertically
       If these aren't satisfied -> rotate landmarks 90/180/270 until they are
    3. Compute a similarity transform (rotation + scale + translation)
       that maps the (now correctly oriented) detected positions
       to the fixed TEMPLATE positions
    4. Apply that transform to the image -> save

TEMPLATE POSITIONS (on 600x700 output canvas):
  sellion   -> (300, 180)   centre top — nose bridge
  pronasale -> (300, 270)   centre — nose tip
  menton    -> (300, 440)   centre bottom — chin
  l_tragion -> (145, 235)   left side — left ear
  r_tragion -> (455, 235)   right side — right ear

READS FROM:
  Dataset/landmarks_raw/<patient>/<scan>_landmarks.json

WRITES TO:
  Dataset/landmark_aligned/<patient>/
      <scan>_aligned.tif
      <scan>_aligned_preview.png
      <scan>_alignment_info.json   <- which rotation was applied + final errors

USAGE:
  python step2_align_images.py
  python step2_align_images.py --patient pat1
  python step2_align_images.py --patient pat1 --show
  python step2_align_images.py --dataset "D:/NahidW/Dataset"

REQUIREMENTS:
  pip install numpy pillow opencv-python matplotlib
"""

import re, sys, json, argparse, warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR   = Path("D:/NahidW/Dataset")
LANDMARKS_DIR = DATASET_DIR / "landmarks_raw"
OUTPUT_DIR    = DATASET_DIR / "landmark_aligned"

OUT_W = 600
OUT_H = 700

# ── Standard template: where every landmark MUST land in output image ─────
# These are fixed for all patients, all timepoints.
# Changing these numbers moves the face position on the canvas.
TEMPLATE = {
    "sellion":   np.array([300, 180], dtype=np.float32),  # centre, upper
    "pronasale": np.array([300, 270], dtype=np.float32),  # centre, nose tip
    "menton":    np.array([300, 440], dtype=np.float32),  # centre, chin
    "l_tragion": np.array([145, 235], dtype=np.float32),  # left ear
    "r_tragion": np.array([455, 235], dtype=np.float32),  # right ear
}

LM_COLOR = {
    "sellion":   (0,   220, 0),
    "pronasale": (220, 220, 0),
    "menton":    (220, 0,   220),
    "l_tragion": (0,   120, 255),
    "r_tragion": (255, 120, 0),
}

LANDMARK_ORDER = ["sellion", "pronasale", "menton", "l_tragion", "r_tragion"]


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD LANDMARK JSONs
# ─────────────────────────────────────────────────────────────────────────────

def load_all_landmarks(landmarks_dir: Path, patient_filter=None) -> dict:
    """
    Scan landmarks_raw/ for all _landmarks.json files.
    Returns { patient_id: [ {scan_data...}, ... ] } sorted by day+variant.
    """
    patients = {}
    pattern  = "**/*_landmarks.json" if not patient_filter else \
               f"{patient_filter}/*_landmarks.json"

    for f in sorted(landmarks_dir.glob(pattern)):
        with open(f) as fp:
            data = json.load(fp)
        pid = data.get("patient", f.parent.name)
        if patient_filter and pid != patient_filter.lower():
            continue
        data["_json_path"] = str(f)
        patients.setdefault(pid, []).append(data)

    for pid in patients:
        patients[pid].sort(key=lambda d: (d.get("day", 0), d.get("variant", "")))

    return patients


# ─────────────────────────────────────────────────────────────────────────────
#  ORIENTATION CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

def is_correctly_oriented(lm: dict, img_w: int, img_h: int) -> bool:
    """
    Check if landmarks describe a face in normal upright orientation.
    Rules (all must pass):
      1. sellion_y < menton_y           (nose bridge above chin)
      2. l_tragion_x < r_tragion_x     (left ear on left side)
      3. sellion_y < pronasale_y < menton_y  (nose tip between bridge and chin)
      4. l_tragion_x < sellion_x < r_tragion_x  (nose bridge between ears)
    """
    sel = lm["sellion"]
    men = lm["menton"]
    pro = lm["pronasale"]
    lt  = lm["l_tragion"]
    rt  = lm["r_tragion"]

    r1 = sel[1] < men[1]                        # sellion above menton
    r2 = lt[0]  < rt[0]                         # left ear on left
    r3 = sel[1] < pro[1] < men[1]               # nose tip between
    r4 = lt[0]  < sel[0] < rt[0]               # sellion between ears

    return r1 and r2 and r3 and r4


def rotate_landmarks(lm: dict, deg: int, img_w: int, img_h: int) -> dict:
    """
    Rotate landmark pixel coordinates by deg degrees
    (same rotation as PIL Image.rotate with expand=True).
    Returns new landmark dict in rotated image coords.
    """
    out = {}
    for name, (px, py) in lm.items():
        if deg == 0:
            nx, ny = px, py
        elif deg == 90:
            # PIL rotate(90) CCW: new_x = py, new_y = W-1-px
            nx, ny = py, img_w - 1 - px
        elif deg == 180:
            nx, ny = img_w - 1 - px, img_h - 1 - py
        elif deg == 270:
            # PIL rotate(270) CCW: new_x = H-1-py, new_y = px
            nx, ny = img_h - 1 - py, px
        out[name] = [int(nx), int(ny)]
    return out


def get_rotated_image_size(w: int, h: int, deg: int):
    if deg in (90, 270):
        return h, w   # width and height swap
    return w, h


def correct_orientation(lm_orig: dict, img_w: int, img_h: int):
    """
    Try rotating landmarks 0, 90, 180, 270 degrees until orientation is correct.
    Returns (corrected_landmarks, rotation_applied, new_img_w, new_img_h).
    If nothing works, returns best guess (180 degrees as fallback).
    """
    for deg in [0, 90, 180, 270]:
        rw, rh = get_rotated_image_size(img_w, img_h, deg)
        lm_rot = rotate_landmarks(lm_orig, deg, img_w, img_h)
        if is_correctly_oriented(lm_rot, rw, rh):
            return lm_rot, deg, rw, rh

    # Fallback: return 0 with a warning
    print(f"    ⚠ Could not auto-correct orientation — using 0° (check output!)")
    return lm_orig, 0, img_w, img_h


# ─────────────────────────────────────────────────────────────────────────────
#  SIMILARITY TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────

def compute_similarity_transform(landmarks: dict) -> np.ndarray:
    """
    Scale + translate only — NO rotation.

    Rotation was already handled by the orientation correction step.
    This step only adjusts:
      - Uniform scale: resize so the face is the same size across all images
      - Translation:   shift so the face centroid lands at the template centroid

    The scale is computed as the ratio of the template span to the
    detected span (using the bounding box of all 5 points), applied
    uniformly to preserve face proportions.

    Because every image maps to the same fixed TEMPLATE, all aligned
    images will have their 5 landmarks at the same pixel positions.
    """
    src = np.array([landmarks[n] for n in LANDMARK_ORDER], dtype=np.float64)
    dst = np.array([TEMPLATE[n]  for n in LANDMARK_ORDER], dtype=np.float64)

    N = len(src)

    # ── Uniform scale ─────────────────────────────────────────────────────────
    # Use the vertical span (sellion Y to menton Y) as the scale reference.
    # This is the most reliable measurement on a face (stable across expressions).
    src_sel = np.array(landmarks["sellion"],  dtype=np.float64)
    src_men = np.array(landmarks["menton"],   dtype=np.float64)
    dst_sel = np.array(TEMPLATE["sellion"],   dtype=np.float64)
    dst_men = np.array(TEMPLATE["menton"],    dtype=np.float64)

    src_span = np.linalg.norm(src_men - src_sel)
    dst_span = np.linalg.norm(dst_men - dst_sel)

    if src_span < 1e-6:
        raise RuntimeError("sellion and menton too close — check landmarks")

    scale = dst_span / src_span

    # ── Translation ───────────────────────────────────────────────────────────
    # After scaling around origin, compute where centroid lands vs template centroid
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)

    # Scale src centroid, then find translation to match dst centroid
    tx = dst_centroid[0] - scale * src_centroid[0]
    ty = dst_centroid[1] - scale * src_centroid[1]

    # ── Build 2x3 matrix (scale only, no rotation) ───────────────────────────
    M = np.array([
        [scale, 0.0,   tx],
        [0.0,   scale, ty],
    ], dtype=np.float32)

    print(f"    Transform: scale={scale:.3f}  translate=({tx:.1f}, {ty:.1f})  NO rotation")

    # ── Residuals ─────────────────────────────────────────────────────────────
    src_h  = np.hstack([src, np.ones((N, 1))])
    warped = (M @ src_h.T).T
    errors = np.linalg.norm(warped - dst, axis=1)
    for name, err in zip(LANDMARK_ORDER, errors):
        print(f"        {name:12s}: residual = {err:.1f} px")
    print(f"    Max residual: {errors.max():.1f} px")

    return M


def apply_transform(img_rgb: np.ndarray, M: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(
        img_rgb, M, (OUT_W, OUT_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )


def verify_alignment(landmarks: dict, M: np.ndarray) -> dict:
    """
    Apply M to each landmark and compare with template.
    Returns {name: {got: [x,y], target: [x,y], error_px: float}}.
    """
    results = {}
    for name in LANDMARK_ORDER:
        px, py = landmarks[name]
        pt = np.array([px, py, 1.0])
        warped = M @ pt
        gx, gy = int(round(warped[0])), int(round(warped[1]))
        tx, ty = int(TEMPLATE[name][0]), int(TEMPLATE[name][1])
        err = ((gx - tx)**2 + (gy - ty)**2) ** 0.5
        results[name] = {"got": [gx, gy], "target": [tx, ty], "error_px": round(err, 2)}
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def draw_landmarks_on_image(img_rgb: np.ndarray, landmarks: dict,
                             label="") -> np.ndarray:
    img = img_rgb.copy()
    for name, (px, py) in landmarks.items():
        r, g, b = LM_COLOR[name]
        cv2.circle(img, (px, py), 10, (r, g, b), -1)
        cv2.circle(img, (px, py), 11, (255, 255, 255), 1)
        cv2.putText(img, name, (px+10, py-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (r, g, b), 1, cv2.LINE_AA)
    return img


def draw_template_on_aligned(img_rgb: np.ndarray,
                              verification: dict) -> np.ndarray:
    img = img_rgb.copy()
    for name, info in verification.items():
        r, g, b = LM_COLOR[name]
        gx, gy = info["got"]
        tx, ty = info["target"]
        err    = info["error_px"]

        # Filled colored dot at actual warped landmark position
        cv2.circle(img, (gx, gy), 10, (r, g, b), -1)
        cv2.circle(img, (gx, gy), 11, (255, 255, 255), 1)

        # White cross at template target (where it SHOULD be)
        cv2.drawMarker(img, (tx, ty), (255, 255, 255),
                       cv2.MARKER_CROSS, 22, 1, cv2.LINE_AA)

        # Name label + error in pixels
        cv2.putText(img, name, (gx + 12, gy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (r, g, b), 1, cv2.LINE_AA)
        cv2.putText(img, f"{err:.1f}px", (gx + 12, gy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

        # Line from dot to cross if misaligned (makes offset visible)
        if err > 2:
            cv2.line(img, (gx, gy), (tx, ty), (100, 100, 100), 1, cv2.LINE_AA)

    return img


def show_result(scan_name: str, orig_rgb: np.ndarray,
                lm_orig: dict, rotated_rgb: np.ndarray,
                lm_rotated: dict, aligned_rgb: np.ndarray,
                verification: dict, rotation: int):
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(f"{scan_name}", color="white", fontsize=13, y=0.98)

    # Panel 1: original + detected landmarks
    axes[0].imshow(draw_landmarks_on_image(orig_rgb, lm_orig))
    axes[0].set_title("Original + detected landmarks\n(raw from step1)",
                      color="#ffaaaa", fontsize=10)
    axes[0].axis("off")

    # Panel 2: after orientation correction
    axes[1].imshow(draw_landmarks_on_image(rotated_rgb, lm_rotated))
    axes[1].set_title(f"Orientation corrected ({rotation}° rotation)\n"
                      "Face now upright, left=left, right=right",
                      color="#aaaaff", fontsize=10)
    axes[1].axis("off")

    # Panel 3: final aligned with error overlay
    axes[2].imshow(draw_template_on_aligned(aligned_rgb, verification))
    max_err = max(v["error_px"] for v in verification.values())
    quality = "excellent" if max_err < 5 else "good" if max_err < 15 else "check manually"
    axes[2].set_title(
        f"Final aligned ({OUT_W}x{OUT_H})\n"
        f"Cross=target, dot=actual  |  max error={max_err:.1f}px ({quality})",
        color="#aaffaa", fontsize=10
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.show(block=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS ONE SCAN
# ─────────────────────────────────────────────────────────────────────────────

def process_scan(scan_data: dict, out_dir: Path, show=False) -> dict:
    scan_name = scan_data["scan"]
    print(f"\n  ── {scan_name}  ({scan_data.get('label','')})")

    # Dataset/<patient>/<scan>/<scan>.tif
    patient  = scan_data["patient"]
    tif_path = DATASET_DIR / patient / scan_name / f"{scan_name}.tif"
    if not tif_path.exists():
        print(f"    ✗ TIF not found: {tif_path}")
        return {"scan": scan_name, "status": "tif_not_found"}

    aligned_tif  = out_dir / f"{scan_name}_aligned.tif"
    aligned_png  = out_dir / f"{scan_name}_aligned_preview.png"
    info_json    = out_dir / f"{scan_name}_alignment_info.json"

    img_w = scan_data["image_w"]
    img_h = scan_data["image_h"]
    lm_orig = {k: v for k, v in scan_data["landmarks"].items()}

    # ── Step A: correct orientation ───────────────────────────────────────────
    lm_corrected, rotation, rw, rh = correct_orientation(lm_orig, img_w, img_h)
    print(f"    Orientation: {rotation}° rotation applied")
    print(f"    Landmarks after correction:")
    for name, (px, py) in lm_corrected.items():
        tgt = TEMPLATE[name]
        print(f"        {name:12s}: ({px:4d},{py:4d})  ->  target ({int(tgt[0])},{int(tgt[1])})")

    # ── Step B: rotate image to match corrected landmarks ─────────────────────
    orig_rgb    = np.array(Image.open(tif_path).convert("RGB"))
    rotated_rgb = np.array(Image.fromarray(orig_rgb).rotate(rotation, expand=True))

    # ── Step C: compute similarity transform ──────────────────────────────────
    try:
        M = compute_similarity_transform(lm_corrected)
    except RuntimeError as e:
        print(f"    ✗ Transform failed: {e}")
        return {"scan": scan_name, "status": "transform_failed"}

    # ── Step D: apply transform -> aligned image ──────────────────────────────
    aligned_rgb = apply_transform(rotated_rgb, M)

    # ── Step E: verify ────────────────────────────────────────────────────────
    verification = verify_alignment(lm_corrected, M)
    errors = [v["error_px"] for v in verification.values()]
    max_err = max(errors)
    avg_err = sum(errors) / len(errors)
    print(f"    Alignment errors (px):  max={max_err:.1f}  avg={avg_err:.1f}")
    if max_err > 20:
        print(f"    ⚠ High error — check {scan_name}_aligned_preview.png")

    # ── Step F: save ──────────────────────────────────────────────────────────
    # Clean TIF — no overlays
    Image.fromarray(aligned_rgb).save(str(aligned_tif), format="TIFF")

    # Preview PNG — landmarks drawn at their final positions on the aligned image
    lm_final = {name: info["got"] for name, info in verification.items()}
    preview  = draw_landmarks_on_image(aligned_rgb, lm_final)
    Image.fromarray(preview).save(str(aligned_png), format="PNG")

    info = {
        "scan":         scan_name,
        "patient":      scan_data["patient"],
        "day":          scan_data["day"],
        "variant":      scan_data["variant"],
        "rotation_applied": rotation,
        "canvas_size":  f"{OUT_W}x{OUT_H}",
        "max_error_px": max_err,
        "avg_error_px": avg_err,
        "template":     {k: v.tolist() for k, v in TEMPLATE.items()},
        "verification": verification
    }
    with open(info_json, "w") as f:
        json.dump(info, f, indent=2)

    print(f"    Saved: {aligned_png.name}")

    # ── Step G: optional show ─────────────────────────────────────────────────
    if show:
        show_result(scan_name, orig_rgb, lm_orig,
                    rotated_rgb, lm_corrected,
                    aligned_rgb, verification, rotation)

    return {"scan": scan_name, "status": "success",
            "rotation": rotation, "max_error_px": max_err}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Step 2: Align face images using landmarks from step1"
    )
    p.add_argument("--patient",  type=str, default=None)
    p.add_argument("--show",     action="store_true",
                   help="Show 3-panel result for each scan")
    p.add_argument("--dataset",  type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    global DATASET_DIR, LANDMARKS_DIR, OUTPUT_DIR
    if args.dataset:
        DATASET_DIR   = Path(args.dataset)
        LANDMARKS_DIR = DATASET_DIR / "landmarks_raw"
        OUTPUT_DIR    = DATASET_DIR / "landmark_aligned"

    if not LANDMARKS_DIR.exists():
        print(f"landmarks_raw/ not found: {LANDMARKS_DIR}")
        print("Run step1_detect_landmarks.py first.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Step 2 — Face Image Alignment")
    print(f"  Landmarks from : {LANDMARKS_DIR}")
    print(f"  Output         : {OUTPUT_DIR}")
    print(f"{'='*60}")

    patients = load_all_landmarks(LANDMARKS_DIR,
                                   patient_filter=args.patient)
    if not patients:
        print("No landmark JSON files found. Run step1 first.")
        sys.exit(1)

    total = sum(len(v) for v in patients.values())
    print(f"\n  {len(patients)} patient(s), {total} scan(s)")

    report = []
    for patient_id, scans in patients.items():
        print(f"\n{'─'*60}")
        print(f"  Patient: {patient_id.upper()}  ({len(scans)} scans)")

        pat_out = OUTPUT_DIR / patient_id
        pat_out.mkdir(parents=True, exist_ok=True)

        for scan_data in scans:
            status = process_scan(scan_data, pat_out, show=args.show)
            report.append(status)

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "alignment_report.json", "w") as f:
        json.dump(report, f, indent=2)

    ok   = sum(1 for r in report if r["status"] == "success")
    fail = [r for r in report if r["status"] != "success"]

    print(f"\n{'='*60}")
    print(f"  DONE  {ok}/{len(report)} aligned")
    if fail:
        print(f"  Issues:")
        for r in fail:
            print(f"    {r['scan']} — {r['status']}")
    print(f"\n  Output : {OUTPUT_DIR}")
    print(f"  Next   : run color_normalization_tool.py on landmark_aligned/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()