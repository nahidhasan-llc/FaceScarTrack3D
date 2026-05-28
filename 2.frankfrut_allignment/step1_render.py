#!/usr/bin/env python3
"""
frankfurt_step1_render.py
=========================
Render frontal face views from ICP-aligned PLY files.

READS:   D:/NahidW/Dataset/3d_scans/ply/pat1/*_icp.ply
WRITES:  D:/NahidW/Dataset/frankfurt_aligned/pat1/*_front.png

USAGE:
    python frankfurt_step1_render.py
    python frankfurt_step1_render.py --patient pat1
    python frankfurt_step1_render.py --root D:/NahidW
    python frankfurt_step1_render.py --overwrite
"""

import argparse
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
ZOOM  = 0.55
IMG_W = 800
IMG_H = 1000


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
#  DISCOVER ICP PLY FILES
#  Matches *_icp.ply (catches both _aligned_icp
#  and _aligned_oriented_icp naming)
# ─────────────────────────────────────────────
def discover_icp_plys(ply_root: Path, patient_filter=None) -> list:
    scans = []
    for pat_dir in sorted(ply_root.iterdir()):
        if not pat_dir.is_dir():
            continue
        pid = pat_dir.name
        if patient_filter and pid.lower() != patient_filter.lower():
            continue
        for f in sorted(pat_dir.glob("*_oriented_icp.ply")):
            if "merged" in f.stem:
                continue
            scans.append((pid, f.stem, f))
    return scans


# ─────────────────────────────────────────────
#  RENDER FRONTAL VIEW
#  Coordinate system of ICP-aligned files:
#    Y = up, face points toward +Z
#    Camera at +Z looking back at the face
# ─────────────────────────────────────────────
def render_front(pcd: o3d.geometry.PointCloud,
                 centroid: np.ndarray,
                 zoom: float,
                 W: int, H: int) -> np.ndarray:
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=W, height=H)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size       = 2.0
    opt.background_color = np.array([0.0, 0.0, 0.0])

    ctr = vis.get_view_control()
    ctr.set_lookat(centroid.tolist())
    ctr.set_front([0.0, 0.0,  1.0])   # camera at +Z, looking at face
    ctr.set_up(   [0.0, 1.0,  0.0])   # Y is up
    ctr.set_zoom(zoom)

    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    return (img * 255).astype(np.uint8)


# ─────────────────────────────────────────────
#  ENHANCE IMAGE
# ─────────────────────────────────────────────
def enhance(img_rgb: np.ndarray) -> np.ndarray:
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gap_mask = (gray < 8).astype(np.uint8) * 255
    img_bgr  = cv2.inpaint(img_bgr, gap_mask,
                            inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    img_bgr  = cv2.convertScaleAbs(img_bgr, alpha=1.4, beta=20)
    return img_bgr


# ─────────────────────────────────────────────
#  PROCESS ONE PLY
# ─────────────────────────────────────────────
def process_one(pid: str, stem: str, ply_path: Path,
                out_dir: Path, overwrite: bool) -> dict:

    out_png = out_dir / f"{stem}_front.png"

    if out_png.exists() and not overwrite:
        print(f"    skip (exists): {out_png.name}")
        return {"scan": stem, "status": "skipped"}

    print(f"    {stem}", end="  ", flush=True)

    try:
        pcd      = o3d.io.read_point_cloud(str(ply_path))
        pts      = np.asarray(pcd.points)
        centroid = pts.mean(axis=0)

        img_rgb = render_front(pcd, centroid, ZOOM, IMG_W, IMG_H)
        img_bgr = enhance(img_rgb)

        cv2.imwrite(str(out_png), img_bgr)
        print(f"✓  {out_png.name}")
        return {"scan": stem, "status": "ok", "output": str(out_png)}

    except Exception as e:
        print(f"✗  {e}")
        return {"scan": stem, "status": "failed", "error": str(e)}


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
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
    out_root = root / "Dataset" / "frankfurt_aligned"

    print(f"\n{'='*55}")
    print(f"  Frankfurt Step 1 — Render Frontal Views")
    print(f"  Input  : {ply_root}")
    print(f"  Output : {out_root}")
    print(f"{'='*55}")

    scans = discover_icp_plys(ply_root, patient_filter=args.patient)
    if not scans:
        print("\nNo *_icp.ply files found. Run face_icp_nahid.py first.")
        return

    print(f"\nFound {len(scans)} ICP-aligned scan(s)\n")

    results  = []
    prev_pat = None
    for pid, stem, ply_path in scans:
        if pid != prev_pat:
            print(f"  Patient: {pid}")
            prev_pat = pid
        out_dir = out_root / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        results.append(
            process_one(pid, stem, ply_path, out_dir, overwrite=args.overwrite))

    ok     = sum(1 for r in results if r["status"] == "ok")
    skip   = sum(1 for r in results if r["status"] == "skipped")
    failed = [r for r in results if r["status"] == "failed"]

    print(f"\n{'='*55}")
    print(f"  Done — rendered: {ok}  skipped: {skip}  failed: {len(failed)}")
    if failed:
        for r in failed:
            print(f"    ✗ {r['scan']} — {r['error']}")
    print(f"\n  Images : {out_root}")
    print(f"  Next   : run frankfurt_step2_align.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()