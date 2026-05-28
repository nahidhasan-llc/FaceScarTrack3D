#!/usr/bin/env python3
"""
pre_orient.py
=============
Run this BEFORE face_icp_nahid.py.

Fixes the yaw (horizontal rotation) of each scan so the face
points exactly toward +Z before ICP alignment starts.

Detection method: GEOMETRIC only — no color.
  Nose tip = median of top 2% Z points in central X strip, upper face.
  This is the most reliable method — works regardless of skin color,
  bindi color, burn scars, or expression.

Overwrites *_aligned.ply files in place.
Saves *_aligned_preorient_backup.ply so you can always recover.

USAGE:
    python pre_orient.py                   # all patients
    python pre_orient.py --patient pat1    # one patient only
    python pre_orient.py --root D:/NahidW  # manual root
    python pre_orient.py --no-backup       # skip backup files
"""

import argparse, re
import numpy as np
import open3d as o3d
from pathlib import Path


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
# Central X strip half-width around face centroid (mm)
# Nose tip is found within this strip
NOSE_X_HALF = 60

# Minimum Y to be considered "upper face" (above neck/chin)
NOSE_Y_MIN  = 280   # above mouth even when open (mouth Y≈230-260)

# Use median of top N% of Z points — robust to outliers
NOSE_Z_TOP_PCT = 98


# ─────────────────────────────────────────────
#  HELPERS
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


def parse_scan_name(filename: str) -> dict | None:
    stem = Path(filename).stem
    m = re.match(r"^(pat\d+)day(\d+)([A-Za-z]\d*)(?:_.+)?$", stem, re.IGNORECASE)
    if not m:
        return None
    return {"patient": m.group(1), "day": int(m.group(2)),
            "camera": m.group(3)}


def discover_patients(root: Path) -> dict[str, list[Path]]:
    scan_root = root / "Dataset" / "3d_scans" / "ply"
    patients  = {}
    for pat_dir in sorted(scan_root.iterdir()):
        if not pat_dir.is_dir():
            continue
        plys = [f for f in sorted(pat_dir.glob("*_aligned.ply"))
                if "_icp"      not in f.stem
                and "merged"   not in f.stem
                and "backup"   not in f.stem
                and "oriented" not in f.stem
                and parse_scan_name(f.name)]
        if plys:
            patients[pat_dir.name] = plys
    return patients


# ─────────────────────────────────────────────
#  NOISE FLOOR AUTO-DETECT
# ─────────────────────────────────────────────
def noise_floor(pts: np.ndarray) -> float:
    counts, edges = np.histogram(pts[:, 1], bins=100)
    total = len(pts)
    for i, c in enumerate(counts):
        if c > total * 0.01:
            return float(edges[i])
    return float(pts[:, 1].min())


# ─────────────────────────────────────────────
#  GEOMETRIC NOSE TIP
#  Central X strip, upper face, highest Z cluster.
#  Pure geometry — no color, works for any patient.
# ─────────────────────────────────────────────
def find_nose_tip(pts: np.ndarray) -> np.ndarray | None:
    thresh    = noise_floor(pts)
    face_pts  = pts[pts[:, 1] >= thresh]
    if len(face_pts) < 100:
        return None

    cx = face_pts[:, 0].mean()

    central = face_pts[
        (face_pts[:, 0] >= cx - NOSE_X_HALF) &
        (face_pts[:, 0] <= cx + NOSE_X_HALF) &
        (face_pts[:, 1] >= NOSE_Y_MIN)
    ]
    if len(central) < 50:
        return None

    z_thresh = np.percentile(central[:, 2], NOSE_Z_TOP_PCT)
    top_pts  = central[central[:, 2] >= z_thresh]
    return top_pts.mean(axis=0)


# ─────────────────────────────────────────────
#  COMPUTE YAW
#  Angle between centroid→nose_tip (projected on XZ)
#  and the +Z axis. Positive = face turned right.
# ─────────────────────────────────────────────
def compute_yaw(pts: np.ndarray, nose_tip: np.ndarray) -> float | None:
    thresh   = noise_floor(pts)
    centroid = pts[pts[:, 1] >= thresh].mean(axis=0)

    fwd    = nose_tip - centroid
    fwd_xz = np.array([fwd[0], 0.0, fwd[2]])
    norm   = np.linalg.norm(fwd_xz)
    if norm < 5.0:
        return None                    # nose too close to centroid — unreliable

    fwd_xz /= norm
    return float(np.degrees(np.arctan2(fwd_xz[0], fwd_xz[2])))


# ─────────────────────────────────────────────
#  APPLY YAW ROTATION around Y axis
#  Rotates around face centroid (not world origin)
# ─────────────────────────────────────────────
def apply_yaw(pcd: o3d.geometry.PointCloud,
              pts: np.ndarray,
              yaw_deg: float) -> o3d.geometry.PointCloud:
    thresh   = noise_floor(pts)
    centroid = pts[pts[:, 1] >= thresh].mean(axis=0)

    angle = np.radians(-yaw_deg)     # counter-rotate to cancel the offset
    c, s  = np.cos(angle), np.sin(angle)
    R     = np.array([[ c, 0, s],
                      [ 0, 1, 0],
                      [-s, 0, c]])

    T_to_o   = np.eye(4); T_to_o[:3, 3]   = -centroid
    T_rot    = np.eye(4); T_rot[:3, :3]   =  R
    T_from_o = np.eye(4); T_from_o[:3, 3] =  centroid
    T        = T_from_o @ T_rot @ T_to_o

    out = o3d.geometry.PointCloud(pcd)
    out.transform(T)
    return out


# ─────────────────────────────────────────────
#  PROCESS ONE FILE
# ─────────────────────────────────────────────
def process_one(path: Path) -> dict:
    print(f"    {path.name}", end="  ", flush=True)

    pcd      = o3d.io.read_point_cloud(str(path))
    pts      = np.asarray(pcd.points)

    nose_tip = find_nose_tip(pts)
    if nose_tip is None:
        print("⚠  nose tip not found — skipping")
        return {"file": path.name, "status": "failed"}

    yaw_deg = compute_yaw(pts, nose_tip)
    if yaw_deg is None:
        print("⚠  yaw unreliable — skipping")
        return {"file": path.name, "status": "failed"}

    if abs(yaw_deg) < 1.0:
        # Still save a copy so downstream scripts have a consistent _oriented.ply
        out_name = path.stem + "_oriented.ply"
        out_path = path.parent / out_name
        o3d.io.write_point_cloud(str(out_path), pcd)
        print(f"✓  already front-facing (yaw={yaw_deg:.2f}°)  →  {out_name}")
        return {"file": path.name, "status": "ok", "yaw_deg": round(yaw_deg, 2)}

    # Save as new file — input is never touched
    out_name = path.stem + "_oriented.ply"
    out_path = path.parent / out_name
    corrected = apply_yaw(pcd, pts, yaw_deg)
    o3d.io.write_point_cloud(str(out_path), corrected)
    print(f"✓  corrected {yaw_deg:+.1f}°  →  {out_name}")
    return {"file": path.name, "status": "corrected",
            "yaw_deg": round(yaw_deg, 2), "output": out_name}


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pre-orient face scans to front-facing before ICP")
    parser.add_argument("--root",      default=None)
    parser.add_argument("--patient",   default=None)

    args = parser.parse_args()

    if args.root:
        root = Path(args.root).expanduser().resolve()
        print(f"Root (manual): {root}")
    else:
        root = find_nahidw_root(Path(__file__))
        print(f"Root (auto-detected): {root}")

    all_patients = discover_patients(root)
    if not all_patients:
        print("No *_aligned.ply files found."); return

    if args.patient:
        pid = args.patient
        if pid not in all_patients:
            print(f"'{pid}' not found. Available: {list(all_patients.keys())}"); return
        all_patients = {pid: all_patients[pid]}

    print(f"\nFound {len(all_patients)} patient(s)")


    all_results = []
    for pid, paths in all_patients.items():
        print(f"\n{'='*50}")
        print(f"  Patient: {pid}  ({len(paths)} scans)")
        print(f"{'='*50}")
        for path in sorted(paths, key=lambda p: (
                parse_scan_name(p.name)["day"],
                parse_scan_name(p.name)["camera"])):
            r = process_one(path)
            all_results.append(r)

    corrected = [r for r in all_results if r["status"] == "corrected"]
    failed    = [r for r in all_results if r["status"] == "failed"]

    print(f"\n{'='*50}")
    print(f"  Done — corrected: {len(corrected)}  "
          f"unchanged: {len(all_results)-len(corrected)-len(failed)}  "
          f"failed: {len(failed)}")
    if failed:
        for r in failed: print(f"    ✗ {r['file']}")
    print(f"\n  Next: python face_icp_nahid.py")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()