#!/usr/bin/env python3
"""
align_3d_front.py
=================
Align all 3D face scans to front-facing orientation.

FIXED ROTATIONS (same for all scans):
  Step 1: -90° around X axis
  Step 2: +90° around Y axis

FINE CORRECTION (per-scan, optional):
  Add to FINE_YAW_CONFIG after reviewing snapshots.
  Positive = rotate face left, Negative = rotate face right.

READS FROM:  Dataset/3d_scans/ply/pat1/pat1day0C.ply
WRITES TO:
  Dataset/3d_scans/ply/pat1/pat1day0C_aligned.ply
  Dataset/3d_scans/ply/pat1/pat1day0C_front.png
  Dataset/3d_scans/obj/pat1/pat1day0C_aligned.obj

USAGE:
  python align_3d_front.py
  python align_3d_front.py --patient pat1
  python align_3d_front.py --show
  python align_3d_front.py --overwrite
  python align_3d_front.py --fine-yaw 15.0

REQUIREMENTS:
  pip install numpy open3d opencv-python
"""

import sys, re, argparse, time
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR = Path("D:/NahidW/Dataset")
SCANS_DIR   = DATASET_DIR / "3d_scans"
PLY_DIR     = SCANS_DIR / "ply"
OBJ_DIR     = SCANS_DIR / "obj"

IMG_W = 800
IMG_H = 1000
ZOOM  = 0.55

FINE_YAW_CONFIG = {
    # "pat1day0C":   -12.0,
    # "pat1day28A":   -3.0,
    # "pat1day28C2": -35.0,
}


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
        if "_aligned" in f.stem:
            continue
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
#  ROTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_transform(rx=0.0, ry=0.0, rz=0.0) -> np.ndarray:
    def Rx(d):
        r = np.radians(d)
        return np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    def Ry(d):
        r = np.radians(d)
        return np.array([[np.cos(r),0,np.sin(r)],[0,1,0],[-np.sin(r),0,np.cos(r)]])
    def Rz(d):
        r = np.radians(d)
        return np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]])
    R = Rz(rz) @ Ry(ry) @ Rx(rx)
    T = np.eye(4)
    T[:3,:3] = R
    return T


# ─────────────────────────────────────────────────────────────────────────────
#  RENDER SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────

def render_snapshot(pcd, out_path: Path):
    """
    Render front-face snapshot using matplotlib scatter.
    No Open3D window — fully deterministic, saves directly to file.
    After alignment face points toward +Y, Z is up.
    Project onto XZ plane, keep front half only.
    """
    import matplotlib
    matplotlib.use('Agg')   # no display needed
    import matplotlib.pyplot as plt

    pts  = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)   # float 0..1

    # Sort back-to-front along Z so face draws on top of back of head
    order = np.argsort(pts[:, 2])
    pts   = pts[order]
    cols  = cols[order]

    # Brighten colors
    cols_bright = np.clip(cols * 1.3, 0, 1)
    cols_uint8  = (cols_bright * 255).astype(np.uint8)

    # Project XY to pixel coordinates
    x = pts[:, 0]
    y = pts[:, 1]

    pad = 0.02
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    ix = ((x - x_min) / (x_max - x_min) * (1-2*pad) + pad) * (IMG_W - 1)
    iy = ((y - y_min) / (y_max - y_min) * (1-2*pad) + pad) * (IMG_H - 1)
    ix = ix.astype(int).clip(0, IMG_W - 1)
    iy = iy.astype(int).clip(0, IMG_H - 1)

    # Paint into image — each point fills a NxN block to remove gaps
    import cv2 as _cv2
    img  = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    N    = 3   # block size — increase if still gaps
    half = N // 2
    for i in range(len(pts)):
        y0 = max(0, iy[i] - half)
        y1 = min(IMG_H, iy[i] + half + 1)
        x0 = max(0, ix[i] - half)
        x1 = min(IMG_W, ix[i] + half + 1)
        img[y0:y1, x0:x1] = cols_uint8[i]

    # Flip vertically so head is up
    img = np.flipud(img)

    img_bgr = _cv2.cvtColor(img, _cv2.COLOR_RGB2BGR)
    _cv2.imwrite(str(out_path), img_bgr)


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE OBJ
# ─────────────────────────────────────────────────────────────────────────────

def save_obj(pcd, out_path: Path):
    pts    = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# {out_path.stem}\n# {len(pts):,} points\n\n")
        rgb = colors / 255.0
        for i in range(len(pts)):
            f.write(f"v {pts[i,0]:.4f} {pts[i,1]:.4f} {pts[i,2]:.4f} "
                    f"{rgb[i,0]:.4f} {rgb[i,1]:.4f} {rgb[i,2]:.4f}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS ONE SCAN
# ─────────────────────────────────────────────────────────────────────────────

def process_scan(patient_id, scan_name, ply_path,
                 fine_yaw_override=None, overwrite=False, show=False):

    ply_out = ply_path.parent / f"{scan_name}_aligned.ply"
    obj_out = OBJ_DIR / patient_id / f"{scan_name}_aligned.obj"
    png_out = ply_path.parent / f"{scan_name}_front.png"

    if ply_out.exists() and png_out.exists() and not overwrite:
        print(f"  ↷ {scan_name}  already done — skipping")
        return {"scan": scan_name, "status": "skipped"}

    print(f"  → {scan_name}")

    try:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        print(f"    Points: {np.asarray(pcd.points).shape[0]:,}")

        # Fixed rotations
        T = make_transform(rx=-90.0, ry=90.0)
        pcd.transform(T)
        print(f"    Applied: Rx=-90°, Ry=+90°")

        # Fine correction
        fine = fine_yaw_override
        if fine is None:
            fine = FINE_YAW_CONFIG.get(scan_name, 0.0)
        if fine != 0.0:
            pcd.transform(make_transform(ry=fine))
            print(f"    Fine Ry: {fine:+.1f}°")

        # Save
        o3d.io.write_point_cloud(str(ply_out), pcd)
        save_obj(pcd, obj_out)
        render_snapshot(pcd, png_out)
        print(f"    ✓ {ply_out.name}")
        print(f"    ✓ {obj_out.name}")
        print(f"    ✓ {png_out.name}")

        if show:
            print(f"    Opening viewer — close to continue...")
            o3d.visualization.draw_geometries(
                [pcd], window_name=scan_name, width=900, height=700)

        return {"scan": scan_name, "status": "success", "fine_ry": fine}

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"scan": scan_name, "status": "failed", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--patient",   type=str,   default=None)
    p.add_argument("--dataset",   type=str,   default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--show",      action="store_true")
    p.add_argument("--fine-yaw",  type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    global DATASET_DIR, SCANS_DIR, PLY_DIR, OBJ_DIR
    if args.dataset:
        DATASET_DIR = Path(args.dataset)
        SCANS_DIR   = DATASET_DIR / "3d_scans"
        PLY_DIR     = SCANS_DIR / "ply"
        OBJ_DIR     = SCANS_DIR / "obj"

    if not PLY_DIR.exists():
        print(f"PLY folder not found: {PLY_DIR}")
        print("Run convert_to_3d.py first.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  3D Face Alignment")
    print(f"  Fixed: Rx=-90°, Ry=+90°")
    print(f"  Input : {PLY_DIR}")
    print(f"{'='*55}\n")

    scans = discover_plys(PLY_DIR, patient_filter=args.patient)
    if not scans:
        print("No PLY files found.")
        sys.exit(1)

    print(f"  Found {len(scans)} scan(s)\n")

    report = []
    prev_pat = None
    for patient_id, scan_name, ply_path in scans:
        if patient_id != prev_pat:
            print(f"{'─'*55}")
            print(f"  Patient: {patient_id.upper()}")
            prev_pat = patient_id
        status = process_scan(patient_id, scan_name, ply_path,
                               fine_yaw_override=args.fine_yaw,
                               overwrite=args.overwrite,
                               show=args.show)
        report.append(status)

    ok   = sum(1 for r in report if r["status"] == "success")
    skip = sum(1 for r in report if r["status"] == "skipped")
    fail = [r for r in report if r["status"] == "failed"]

    print(f"\n{'='*55}")
    print(f"  DONE  {ok} aligned, {skip} skipped")
    if fail:
        for r in fail:
            print(f"  ✗ {r['scan']} — {r['error']}")
    print(f"\n  Review _front.png snapshots in {PLY_DIR}")
    print(f"  Add fine corrections to FINE_YAW_CONFIG if needed")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()