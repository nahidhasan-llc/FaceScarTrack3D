#!/usr/bin/env python3
"""
pre_orient.py
=============
Run this BEFORE face_icp_nahid.py.

Fixes the yaw (horizontal rotation) of each scan so the face
points exactly toward +Z before ICP alignment starts.

Detection method: GEOMETRIC only — no color.
  Nose tip = median of top 2% Z points in central X strip, upper face.

USAGE:
    python pre_orient.py                   # all patients
    python pre_orient.py --patient pat1    # one patient only
    python pre_orient.py --root D:/NahidW  # manual root
"""

import argparse, re
import numpy as np
import open3d as o3d
from pathlib import Path


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
NOSE_X_HALF    = 60
NOSE_Y_MIN     = 280
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


def discover_patients(root: Path, fmt: str = "both") -> dict[str, list[Path]]:
    patients  = {}
    scan_root = root / "Dataset" / "3d_scans"

    # PLY: ply/<pat>/*_aligned.ply
    ply_root = scan_root / "ply"
    if fmt in ("ply", "both") and ply_root.exists():
        for pat_dir in sorted(ply_root.iterdir()):
            if not pat_dir.is_dir():
                continue
            files = [f for f in sorted(pat_dir.glob("*_aligned.ply"))
                     if "_icp"      not in f.stem
                     and "merged"   not in f.stem
                     and "backup"   not in f.stem
                     and "oriented" not in f.stem
                     and parse_scan_name(f.name)]
            if files:
                patients.setdefault(pat_dir.name, []).extend(files)

    # OBJ: obj/<pat>/*_aligned.obj
    obj_root = scan_root / "obj"
    if fmt in ("obj", "both") and obj_root.exists():
        for pat_dir in sorted(obj_root.iterdir()):
            if not pat_dir.is_dir():
                continue
            files = [f for f in sorted(pat_dir.glob("*_aligned.obj"))
                     if "_icp"      not in f.stem
                     and "merged"   not in f.stem
                     and "backup"   not in f.stem
                     and "oriented" not in f.stem
                     and parse_scan_name(f.name)]
            if files:
                patients.setdefault(pat_dir.name, []).extend(files)

    return patients


# ─────────────────────────────────────────────
#  OBJ READ / WRITE  (open3d cannot handle vertex-color OBJ)
# ─────────────────────────────────────────────
def _read_obj(path: Path) -> o3d.geometry.PointCloud:
    pts_list, col_list = [], []
    with open(path, "r", errors="replace") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            p = line.split()
            pts_list.append([float(p[1]), float(p[2]), float(p[3])])
            if len(p) >= 7:
                rgb = [float(p[4]), float(p[5]), float(p[6])]
                col_list.append([v / 255.0 if v > 1.0 else v for v in rgb])
            else:
                col_list.append([0.5, 0.5, 0.5])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts_list))
    pcd.colors = o3d.utility.Vector3dVector(np.array(col_list))
    return pcd


def _write_obj(pcd: o3d.geometry.PointCloud, out_path: Path):
    pts    = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# " + out_path.stem + "\n")
        f.write("# " + str(len(pts)) + " points\n\n")
        rgb = colors / 255.0
        for i in range(len(pts)):
            f.write("v %.4f %.4f %.4f %.4f %.4f %.4f\n" % (
                pts[i, 0], pts[i, 1], pts[i, 2],
                rgb[i, 0], rgb[i, 1], rgb[i, 2]))


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
# ─────────────────────────────────────────────
def find_nose_tip(pts: np.ndarray) -> np.ndarray | None:
    thresh   = noise_floor(pts)
    face_pts = pts[pts[:, 1] >= thresh]
    if len(face_pts) < 100:
        return None
    cx      = face_pts[:, 0].mean()
    central = face_pts[
        (face_pts[:, 0] >= cx - NOSE_X_HALF) &
        (face_pts[:, 0] <= cx + NOSE_X_HALF) &
        (face_pts[:, 1] >= NOSE_Y_MIN)
    ]
    if len(central) < 50:
        return None
    z_thresh = np.percentile(central[:, 2], NOSE_Z_TOP_PCT)
    return central[central[:, 2] >= z_thresh].mean(axis=0)


# ─────────────────────────────────────────────
#  COMPUTE YAW
# ─────────────────────────────────────────────
def compute_yaw(pts: np.ndarray, nose_tip: np.ndarray) -> float | None:
    thresh   = noise_floor(pts)
    centroid = pts[pts[:, 1] >= thresh].mean(axis=0)
    fwd      = nose_tip - centroid
    fwd_xz   = np.array([fwd[0], 0.0, fwd[2]])
    norm     = np.linalg.norm(fwd_xz)
    if norm < 5.0:
        return None
    fwd_xz /= norm
    return float(np.degrees(np.arctan2(fwd_xz[0], fwd_xz[2])))


# ─────────────────────────────────────────────
#  APPLY YAW ROTATION
# ─────────────────────────────────────────────
def apply_yaw(pcd: o3d.geometry.PointCloud,
              pts: np.ndarray,
              yaw_deg: float) -> o3d.geometry.PointCloud:
    thresh   = noise_floor(pts)
    centroid = pts[pts[:, 1] >= thresh].mean(axis=0)
    angle    = np.radians(-yaw_deg)
    c, s     = np.cos(angle), np.sin(angle)
    R        = np.array([[ c, 0, s], [0, 1, 0], [-s, 0, c]])
    T_to_o   = np.eye(4); T_to_o[:3, 3]  = -centroid
    T_rot    = np.eye(4); T_rot[:3, :3]  =  R
    T_from_o = np.eye(4); T_from_o[:3, 3] = centroid
    T        = T_from_o @ T_rot @ T_to_o
    out      = o3d.geometry.PointCloud(pcd)
    out.transform(T)
    return out


# ─────────────────────────────────────────────
#  PROCESS ONE FILE
# ─────────────────────────────────────────────
def process_one(path: Path, overwrite: bool = True) -> dict:
    print("    " + path.name, end="  ", flush=True)

    ext      = path.suffix
    out_path = path.parent / (path.stem + "_oriented" + ext)
    if out_path.exists() and not overwrite:
        print("  already exists — skipping")
        return {"file": path.name, "status": "skipped"}

    is_obj = path.suffix.lower() == ".obj"
    pcd    = _read_obj(path) if is_obj else o3d.io.read_point_cloud(str(path))
    pts    = np.asarray(pcd.points)

    nose_tip = find_nose_tip(pts)
    if nose_tip is None:
        print("  nose tip not found — skipping")
        return {"file": path.name, "status": "failed"}

    yaw_deg = compute_yaw(pts, nose_tip)
    if yaw_deg is None:
        print("  yaw unreliable — skipping")
        return {"file": path.name, "status": "failed"}

    out_name  = out_path.name
    corrected = apply_yaw(pcd, pts, yaw_deg) if abs(yaw_deg) >= 1.0 else pcd

    if is_obj:
        _write_obj(corrected, out_path)
    else:
        o3d.io.write_point_cloud(str(out_path), corrected)

    if abs(yaw_deg) < 1.0:
        print("  already front-facing (yaw=%.2f)  ->  %s" % (yaw_deg, out_name))
        return {"file": path.name, "status": "ok", "yaw_deg": round(yaw_deg, 2)}
    else:
        print("  corrected %+.1f  ->  %s" % (yaw_deg, out_name))
        return {"file": path.name, "status": "corrected",
                "yaw_deg": round(yaw_deg, 2), "output": out_name}


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pre-orient face scans to front-facing before ICP")
    parser.add_argument("--root",    default=None)
    parser.add_argument("--patient", default=None)
    parser.add_argument("--format",  default="both", choices=["ply", "obj", "both"],
                        help="Input format to process (default: both)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files (default: skip)")
    args = parser.parse_args()

    if args.root:
        root = Path(args.root).expanduser().resolve()
        print("Root (manual): " + str(root))
    else:
        root = find_nahidw_root(Path(__file__))
        print("Root (auto-detected): " + str(root))

    all_patients = discover_patients(root, fmt=args.format)
    if not all_patients:
        print("No *_aligned.ply / *_aligned.obj files found.")
        return

    if args.patient:
        pid = args.patient
        if pid not in all_patients:
            print("'" + pid + "' not found. Available: " + str(list(all_patients.keys())))
            return
        all_patients = {pid: all_patients[pid]}

    print("\nFound " + str(len(all_patients)) + " patient(s)")

    all_results = []
    for pid, paths in all_patients.items():
        print("\n" + "=" * 50)
        print("  Patient: " + pid + "  (" + str(len(paths)) + " scans)")
        print("=" * 50)
        for path in sorted(paths, key=lambda p: (
                parse_scan_name(p.name)["day"],
                parse_scan_name(p.name)["camera"])):
            r = process_one(path, overwrite=args.overwrite)
            all_results.append(r)

    corrected = [r for r in all_results if r["status"] == "corrected"]
    skipped   = [r for r in all_results if r["status"] == "skipped"]
    failed    = [r for r in all_results if r["status"] == "failed"]

    print("\n" + "=" * 50)
    print("  Done — corrected: %d  unchanged: %d  skipped: %d  failed: %d" % (
        len(corrected),
        len(all_results) - len(corrected) - len(skipped) - len(failed),
        len(skipped),
        len(failed)))
    if failed:
        for r in failed:
            print("    x " + r["file"])
    print("\n  Next: python face_icp_nahid.py")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()







# python step1_preorient.py                        # PLY + OBJ (default)
# python step1_preorient.py --format ply           # PLY only
# python step1_preorient.py --format obj           # OBJ only
# python step1_preorient.py --format obj --patient pat1   # OBJ, one patient

# python step1_preorient.py                   # skips existing outputs
# python step1_preorient.py --overwrite       # redoes everything
# python step1_preorient.py --format obj --overwrite   # redo OBJ only