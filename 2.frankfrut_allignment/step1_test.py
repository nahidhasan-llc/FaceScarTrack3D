#!/usr/bin/env python3
"""
frankfurt_step1_render.py
=========================
Batch render frontal face views from all PLY files in Dataset/ply_files/.
Output goes to Dataset/frankfurt_aligned/<patient>/

READS FROM:
  Dataset/ply_files/
    pat1/
        pat1day0C.ply
        pat1day28A.ply

WRITES TO:
  Dataset/frankfurt_aligned/
    pat1/
        pat1day0C_front.png
        pat1day28A_front.png

USAGE:
  python frankfurt_step1_render.py
  python frankfurt_step1_render.py --patient pat1
  python frankfurt_step1_render.py --dataset "D:/NahidW/Dataset"
  python frankfurt_step1_render.py --overwrite

REQUIREMENTS:
  pip install numpy open3d opencv-python
"""

import sys, os, re, argparse
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR = Path("D:/NahidW/Dataset")
PLY_DIR     = DATASET_DIR / "ply_files"
OUTPUT_DIR  = DATASET_DIR / "frankfurt_aligned"

# Render settings — must match step2 exactly
YAW_DEG = 15.0
ZOOM    = 0.55
IMG_W   = 800
IMG_H   = 1000

# ─────────────────────────────────────────────────────────────────────────────
#  FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

SCAN_RE = re.compile(
    r"^(?P<patient>pat\d+)day(?P<day>\d+)(?P<variant>[A-Z][A-Z0-9]?)\.ply$",
    re.IGNORECASE
)


def discover_plys(ply_dir: Path, patient_filter=None) -> list:
    scans = []
    for f in sorted(ply_dir.glob("*/*.ply")):
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
#  RENDER
# ─────────────────────────────────────────────────────────────────────────────

def score_face_image(img_rgb: np.ndarray) -> float:
    """
    Score how likely a rendered image is a frontal face view.
    Uses two signals:
      1. Skin-coloured pixels in the central 50% of the image
         (face should fill the centre, not the edges)
      2. Vertical symmetry — a frontal face is roughly symmetric
         left-right in the centre band
    Higher score = more likely to be the correct face-forward direction.
    """
    H, W = img_rgb.shape[:2]

    # Central region: middle 50% horizontally, middle 60% vertically
    cx0, cx1 = W // 4,     3 * W // 4
    cy0, cy1 = H // 5,     4 * H // 5
    centre   = img_rgb[cy0:cy1, cx0:cx1].astype(np.float32) / 255.0

    r, g, b = centre[:,:,0], centre[:,:,1], centre[:,:,2]

    # Skin pixel count in centre
    skin = ((r > 0.25) & (r > g * 1.05) & (g > b * 0.85) &
            (r - b > 0.06) & (r < 0.98))
    skin_score = float(skin.sum())

    # Left-right symmetry of skin pixels
    left_skin  = skin[:, :skin.shape[1]//2].sum()
    right_skin = skin[:, skin.shape[1]//2:].sum()
    total      = left_skin + right_skin + 1e-6
    symmetry   = 1.0 - abs(left_skin - right_skin) / total

    return skin_score * (0.5 + 0.5 * symmetry)


def find_up_vector(pts, cols, front_vec):
    """
    Find the true head-up direction from the 3D face point cloud.

    Strategy:
      The head vertical axis = the direction along which the face points
      have maximum spread PERPENDICULAR to the front_vec.

      We use PCA on the face points (top 30% radial), project out the
      front_vec component, and pick the remaining principal axis that
      is most aligned with world Z (since patients stand/sit upright).

      Then try both that axis and its opposite, pick the one that
      scores better (more skin in upper half vs lower half of render).
    """
    # Get face point subset
    xy_radius = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    face_mask = xy_radius >= np.percentile(xy_radius, 70)
    if cols is not None and len(cols) == len(pts):
        r, g, b   = cols[:, 0], cols[:, 1], cols[:, 2]
        skin_mask = (r > 0.2) & (r > g * 1.02) & (r - b > 0.05)
        combined  = face_mask & skin_mask
        if combined.sum() >= 300:
            face_mask = combined
    face_pts = pts[face_mask]

    # PCA on face points
    centred  = face_pts - face_pts.mean(axis=0)
    cov      = centred.T @ centred
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvecs columns are principal axes, sorted ascending eigenvalue
    # largest variance = eigvecs[:, 2], second = eigvecs[:, 1]

    # Project out the front_vec from each principal axis
    # so we only consider axes perpendicular to the camera direction
    candidates = []
    for i in [2, 1, 0]:
        axis = eigvecs[:, i].copy()
        axis -= np.dot(axis, front_vec) * front_vec   # remove front component
        n = np.linalg.norm(axis)
        if n > 0.1:
            axis /= n
            # Prefer axes more aligned with world Z
            z_alignment = abs(axis[2])
            candidates.append((z_alignment, axis))

    candidates.sort(key=lambda x: -x[0])
    up_candidate = candidates[0][1]

    # Ensure it points generally upward (positive Z preferred)
    if up_candidate[2] < 0:
        up_candidate = -up_candidate

    return up_candidate


def detect_face_orientation(pts, cols, pcd, centroid):
    """
    Find the correct front_vec AND up_vec for rendering.

    Step 1: Find face-forward direction (front_vec) using max radial points.
            Test both the candidate and its opposite — pick the face side.
    Step 2: Find the true head-up direction using PCA on face points.
            Test both up and its opposite — pick whichever puts more
            skin pixels in the upper half of the rendered image.

    This handles any scan orientation stored by the Cyberware scanner.
    Returns (front_vec, up_vec)
    """
    # ── Step 1: front direction ───────────────────────────────────────────────
    xy_radius = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    threshold = np.percentile(xy_radius, 80)
    face_mask = xy_radius >= threshold
    if cols is not None and len(cols) == len(pts):
        r, g, b   = cols[:, 0], cols[:, 1], cols[:, 2]
        skin_mask = (r > 0.25) & (r > g * 1.05) & (g > b * 0.9) & (r - b > 0.08)
        combined  = face_mask & skin_mask
        if combined.sum() >= 200:
            face_mask = combined

    face_centre  = pts[face_mask].mean(axis=0)
    candidate    = face_centre - centroid
    candidate[2] = 0.0
    norm = np.linalg.norm(candidate)
    candidate = candidate / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    opposite  = -candidate.copy()

    # Test both with default up=[0,0,1] to find correct face side
    default_up = np.array([0.0, 0.0, 1.0])
    best_front = candidate
    best_score = -1.0
    for vec, label in [(candidate, "front_cand"), (opposite, "front_opp")]:
        img = render_front_view(pcd, centroid, vec, default_up, ZOOM, 300, 375)
        sc  = score_face_image(img)
        yaw = np.degrees(np.arctan2(vec[1], vec[0]))
        print(f"    {label}: yaw={yaw:+.1f}  score={sc:.0f}")
        if sc > best_score:
            best_score = sc
            best_front = vec.copy()

    # ── Step 2: up direction ──────────────────────────────────────────────────
    up_candidate = find_up_vector(pts, cols, best_front)
    up_opposite  = -up_candidate.copy()

    best_up    = up_candidate
    best_up_sc = -1.0
    for up, label in [(up_candidate, "up_cand"), (up_opposite, "up_opp")]:
        img = render_front_view(pcd, centroid, best_front, up, ZOOM, 300, 375)
        # Score: more skin in upper 50% than lower 50% = face upright
        H = img.shape[0]
        top_score = score_face_image(img[:H//2])
        bot_score = score_face_image(img[H//2:])
        sc = top_score - bot_score * 0.3   # upright face has more in top
        print(f"    {label}: score={sc:.0f}  (top={top_score:.0f} bot={bot_score:.0f})")
        if sc > best_up_sc:
            best_up_sc = sc
            best_up    = up.copy()

    yaw_final = np.degrees(np.arctan2(best_front[1], best_front[0]))
    print(f"    Final: yaw={yaw_final:+.1f}  up=[{best_up[0]:.2f},{best_up[1]:.2f},{best_up[2]:.2f}]")
    return best_front, best_up


def render_front_view(pcd, centroid, front_vec, up_vec, zoom, W, H) -> np.ndarray:
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=W, height=H)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size       = 2.0
    opt.background_color = np.array([0.0, 0.0, 0.0])
    ctr = vis.get_view_control()
    ctr.set_lookat(centroid.tolist())
    ctr.set_up(up_vec.tolist())
    ctr.set_front(front_vec.tolist())
    ctr.set_zoom(zoom)
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    return (img * 255).astype(np.uint8)


def process_ply(patient_id, scan_name, ply_path, out_dir,
                overwrite=False, yaw_override=None):
    import json

    out_png  = out_dir / f"{scan_name}_front.png"
    out_meta = out_dir / f"{scan_name}_render_meta.json"

    if out_png.exists() and not overwrite:
        print(f"  ↷ {scan_name}_front.png  already exists — skipping")
        return {"scan": scan_name, "status": "skipped"}

    print(f"  → {scan_name}")

    try:
        pcd      = o3d.io.read_point_cloud(str(ply_path))
        pts      = np.asarray(pcd.points)
        cols     = np.asarray(pcd.colors) if pcd.has_colors() else None
        centroid = pts.mean(axis=0)
        print(f"    Points: {len(pts):,}  centroid: {centroid.round(1)}")

        if yaw_override is not None:
            yaw_rad   = np.radians(-yaw_override)
            front_vec = np.array([-np.cos(yaw_rad), -np.sin(yaw_rad), 0.0])
            up_vec    = np.array([0.0, 0.0, 1.0])
            print(f"    Yaw override: {yaw_override}deg")
        else:
            front_vec, up_vec = detect_face_orientation(pts, cols, pcd, centroid)

        img_rgb = render_front_view(pcd, centroid, front_vec, up_vec,
                                    ZOOM, IMG_W, IMG_H)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Fill point cloud gaps so MediaPipe detects face more reliably
        gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gap_mask = (gray < 8).astype(np.uint8) * 255
        img_bgr  = cv2.inpaint(img_bgr, gap_mask, inpaintRadius=4,
                                flags=cv2.INPAINT_TELEA)
        img_bgr  = cv2.convertScaleAbs(img_bgr, alpha=1.4, beta=20)

        cv2.imwrite(str(out_png), img_bgr)

        # Save render meta so step2 can use exact same camera params
        meta = {
            "scan":      scan_name,
            "front_vec": front_vec.tolist(),
            "up_vec":    up_vec.tolist(),
            "centroid":  centroid.tolist(),
            "zoom":      ZOOM,
            "img_w":     IMG_W,
            "img_h":     IMG_H,
        }
        with open(out_meta, "w") as f:
            json.dump(meta, f, indent=2)

        yaw_deg = np.degrees(np.arctan2(front_vec[1], front_vec[0]))
        print(f"    Saved: {out_png.name}  yaw={yaw_deg:.1f}deg")
        return {"scan": scan_name, "status": "success",
                "output": str(out_png), "yaw": round(yaw_deg, 2)}

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"scan": scan_name, "status": "failed", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Step 1: Render frontal views from PLY files"
    )
    p.add_argument("--patient",   type=str, default=None)
    p.add_argument("--dataset",   type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--yaw",       type=float, default=None,
                   help="Override auto-detected yaw for all scans")
    return p.parse_args()


def main():
    args = parse_args()
    global DATASET_DIR, PLY_DIR, OUTPUT_DIR
    if args.dataset:
        DATASET_DIR = Path(args.dataset)
        PLY_DIR     = DATASET_DIR / "ply_files"
        OUTPUT_DIR  = DATASET_DIR / "frankfurt_aligned"

    if not PLY_DIR.exists():
        print(f"PLY folder not found: {PLY_DIR}")
        print("Run convert_all_to_ply.py first.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Frankfurt Step 1 — Render Frontal Views")
    print(f"  PLY input : {PLY_DIR}")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"{'='*55}\n")

    scans = discover_plys(PLY_DIR, patient_filter=args.patient)
    if not scans:
        print("No PLY files found.")
        sys.exit(1)

    print(f"  Found {len(scans)} PLY file(s)\n")

    report   = []
    prev_pat = None

    for patient_id, scan_name, ply_path in scans:
        if patient_id != prev_pat:
            print(f"{'─'*55}")
            print(f"  Patient: {patient_id.upper()}")
            prev_pat = patient_id

        pat_out = OUTPUT_DIR / patient_id
        pat_out.mkdir(parents=True, exist_ok=True)

        status = process_ply(patient_id, scan_name, ply_path,
                             pat_out, overwrite=args.overwrite,
                             yaw_override=args.yaw)
        report.append(status)

    ok     = sum(1 for r in report if r["status"] == "success")
    skip   = sum(1 for r in report if r["status"] == "skipped")
    failed = [r for r in report if r["status"] == "failed"]

    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Rendered : {ok}")
    print(f"  Skipped  : {skip}")
    if failed:
        print(f"  Failed   : {len(failed)}")
        for r in failed:
            print(f"    {r['scan']} — {r['error']}")
    print(f"\n  Output   : {OUTPUT_DIR}")
    print(f"  Next     : run frankfurt_step2_align.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()