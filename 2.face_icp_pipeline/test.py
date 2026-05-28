"""
Face ICP Alignment — NahidW Dataset
=====================================
Script location:
    D:/NahidW/Coding/2.face_icp_pipeline/face_icp_nahid.py

What this does (simple and correct):
    1. Clean  — remove background scatter below Y=100
    2. Scale  — rescale each moving scan to match reference face height
    3. ICP    — point-to-plane ICP from identity (scans already front-facing,
                just need small rotation + translation correction)

That's it. No PCA, no FPFH, no coarse alignment, no ROI tricks.

Usage:
    python face_icp_nahid.py                   # all patients, auto root
    python face_icp_nahid.py --patient pat1    # one patient only
    python face_icp_nahid.py --root D:/NahidW  # manual root override
"""

import open3d as o3d
import numpy as np
import re, csv, argparse
from pathlib import Path


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

# Noise removal: scatter below this Y is tripod/background.
# Set to None to auto-detect from the data (recommended —
# works for any scanner/orientation). Set a fixed mm value
# only if auto-detection gives wrong results for your scanner.
NOISE_Y_MIN = None      # None = auto-detect

# Scale: inter-percentile span per axis, geometric mean across X/Y/Z
SCALE_PCT_HI = 95       # upper percentile
SCALE_PCT_LO = 5        # lower percentile

# Nose marker (bindi) detection — color thresholds in [0,1] RGB.
# Default detects a red dot. Change if your marker is a different colour,
# or set MARKER_COLOR = None to skip the snap step entirely.
MARKER_R_MIN = 0.5      # red channel minimum
MARKER_G_MAX = 0.3      # green channel maximum
MARKER_B_MAX = 0.3      # blue channel maximum
MARKER_CLUSTER_MM = 15  # search radius (mm) for density clustering

# ICP: five passes coarse → fine
# More passes = better rotation and tilt correction
ICP_SCALES = [
    (40.0, 100),   # gross translation
    (20.0, 150),   # coarse rotation
    (10.0, 200),   # medium tilt correction
    (5.0,  200),   # fine
    (2.0,  200),   # very fine surface lock
]

VOXEL_SIZE    = 2.0     # mm — downsampling for ICP
NORMAL_RADIUS = 6.0     # mm — normal estimation
OUTPUT_SUFFIX = "_icp"


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def parse_scan_name(filename: str) -> dict | None:
    stem = Path(filename).stem
    m = re.match(r"^(pat\d+)day(\d+)([A-Za-z]\d*)(?:_.+)?$", stem, re.IGNORECASE)
    if not m:
        return None
    return {"patient": m.group(1), "day": int(m.group(2)),
            "camera": m.group(3), "stem": stem}


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


def discover_patients(root: Path) -> dict[str, list[Path]]:
    scan_root = root / "Dataset" / "3d_scans" / "ply"
    if not scan_root.exists():
        raise FileNotFoundError(f"Not found: {scan_root}")
    patients = {}
    for pat_dir in sorted(scan_root.iterdir()):
        if not pat_dir.is_dir():
            continue
        plys = [f for f in sorted(pat_dir.glob("*_aligned.ply"))
                if OUTPUT_SUFFIX not in f.stem
                and "merged" not in f.stem
                and parse_scan_name(f.name)]
        if plys:
            patients[pat_dir.name] = plys
    return patients


# ─────────────────────────────────────────────
#  STEP 1 — LOAD
# ─────────────────────────────────────────────
def load_pcd(path: Path) -> o3d.geometry.PointCloud:
    mesh = o3d.io.read_triangle_mesh(str(path))
    pcd  = (mesh.sample_points_poisson_disk(40_000)
            if len(mesh.triangles) > 0
            else o3d.io.read_point_cloud(str(path)))
    if len(pcd.points) == 0:
        raise ValueError(f"No points in {path.name}")
    return pcd


# ─────────────────────────────────────────────
#  STEP 2 — CLEAN
#  Remove low-Y background scatter, keep face cluster only.
# ─────────────────────────────────────────────
def clean(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pts  = np.asarray(pcd.points)

    # Auto-detect noise floor: find the largest gap in Y density below the
    # main face cluster. The face is a dense blob; scatter sits below it
    # with a clear gap. Works for any scanner orientation/scale.
    if NOISE_Y_MIN is not None:
        y_thresh = NOISE_Y_MIN
    else:
        # Build a Y histogram with 100 bins, find the lowest bin that
        # contains > 1% of points — that's the bottom of the face cluster.
        y_vals = pts[:, 1]
        counts, edges = np.histogram(y_vals, bins=100)
        total = len(y_vals)
        # Walk up from the bottom until we hit a bin with >1% of points
        y_thresh = edges[0]
        for i, c in enumerate(counts):
            if c > total * 0.01:
                y_thresh = edges[i]
                break

    pcd  = pcd.select_by_index(np.where(pts[:, 1] >= y_thresh)[0])

    # Keep largest connected cluster (the face) using 3mm voxel / eps=20
    tmp    = pcd.voxel_down_sample(3.0)
    labels = np.array(tmp.cluster_dbscan(eps=20, min_points=5, print_progress=False))
    if labels.max() >= 0:
        cl  = np.asarray(tmp.points)[labels == np.bincount(labels[labels >= 0]).argmax()]
        PAD = 15
        p   = np.asarray(pcd.points)
        mask = (
            (p[:, 0] >= cl[:, 0].min() - PAD) & (p[:, 0] <= cl[:, 0].max() + PAD) &
            (p[:, 1] >= cl[:, 1].min() - PAD) & (p[:, 1] <= cl[:, 1].max() + PAD) &
            (p[:, 2] >= cl[:, 2].min() - PAD) & (p[:, 2] <= cl[:, 2].max() + PAD)
        )
        pcd = pcd.select_by_index(np.where(mask)[0])
    return pcd


# ─────────────────────────────────────────────
#  STEP 3 — SCALE
#  Measure vertical face span (forehead–chin).
#  Rescale uniformly around centroid so it matches reference.
# ─────────────────────────────────────────────
def face_scale(pcd: o3d.geometry.PointCloud) -> float:
    """Geometric mean of X/Y/Z inter-percentile spans.
    Captures scale in all three axes — fixes Z-depth nesting."""
    pts = np.asarray(pcd.points)
    spans = [np.percentile(pts[:, ax], SCALE_PCT_HI) -
             np.percentile(pts[:, ax], SCALE_PCT_LO) for ax in range(3)]
    return float(np.cbrt(spans[0] * spans[1] * spans[2]))


def rescale(pcd: o3d.geometry.PointCloud, factor: float) -> o3d.geometry.PointCloud:
    if abs(factor - 1.0) < 1e-5:
        return pcd
    pts      = np.asarray(pcd.points).copy()
    centroid = pts.mean(axis=0)
    pts      = centroid + (pts - centroid) * factor
    out      = o3d.geometry.PointCloud(pcd)
    out.points = o3d.utility.Vector3dVector(pts)
    return out


# ─────────────────────────────────────────────
#  STEP 4 — DOWNSAMPLE + NORMALS  (for ICP)
# ─────────────────────────────────────────────
def prepare(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    d = pcd.voxel_down_sample(VOXEL_SIZE)
    d.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=30))
    d.orient_normals_consistent_tangent_plane(k=15)
    return d


# ─────────────────────────────────────────────
#  STEP 5 — ICP  (point-to-plane, from identity)
# ─────────────────────────────────────────────
def icp(src, tgt, init_T, max_dist, max_iter):
    return o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=max_dist,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter, relative_fitness=1e-7, relative_rmse=1e-7)
    )



# ─────────────────────────────────────────────
#  STEP 6 — NOSE BINDI SNAP
#  Detects the red bindi mark by color in both scans,
#  then applies a final translation so they sit at exactly
#  the same 3D position. Works for any patient that has
#  a visible red mark on their nose.
# ─────────────────────────────────────────────
def find_bindi(pcd: o3d.geometry.PointCloud) -> np.ndarray | None:
    """
    Returns the 3D centroid of the red bindi mark, or None if not found.
    Detection: R > 0.5, G < 0.3, B < 0.3, then densest spatial cluster.
    """
    if not pcd.has_colors():
        return None
    pts    = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    mask = ((colors[:, 0] > MARKER_R_MIN) &
            (colors[:, 1] < MARKER_G_MAX) &
            (colors[:, 2] < MARKER_B_MAX))
    if mask.sum() < 5:
        return None

    red_pts = pts[mask]
    try:
        from scipy.spatial import KDTree
        tree   = KDTree(red_pts)
        counts = tree.query_ball_point(red_pts, r=MARKER_CLUSTER_MM, return_length=True)
        nearby = red_pts[counts > counts.max() * 0.5]
        if len(nearby) < 3:
            return None
        return nearby.mean(axis=0)
    except ImportError:
        # fallback: just use centroid of all red points
        return red_pts.mean(axis=0)


def bindi_snap(src_orig: o3d.geometry.PointCloud,
               tgt_orig: o3d.geometry.PointCloud,
               T_icp: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Apply T_icp to src, find bindi in both, compute residual translation,
    return corrected final transform and snap info.
    """
    # Apply current ICP transform to get the aligned src
    src_aligned = o3d.geometry.PointCloud(src_orig)
    src_aligned.transform(T_icp)

    src_bindi = find_bindi(src_aligned)
    tgt_bindi = find_bindi(tgt_orig)

    if src_bindi is None or tgt_bindi is None:
        missing = []
        if src_bindi is None: missing.append("source")
        if tgt_bindi is None: missing.append("target")
        print(f"    ⚠  Bindi not found in: {', '.join(missing)} — skipping snap")
        return T_icp, {"bindi_snap_mm": None}

    # Residual offset between bindi positions after ICP
    offset    = tgt_bindi - src_bindi
    offset_mm = np.linalg.norm(offset)
    print(f"    Bindi src: ({src_bindi[0]:.1f}, {src_bindi[1]:.1f}, {src_bindi[2]:.1f})")
    print(f"    Bindi tgt: ({tgt_bindi[0]:.1f}, {tgt_bindi[1]:.1f}, {tgt_bindi[2]:.1f})")
    print(f"    Bindi residual offset: {offset_mm:.2f} mm  → snapping")

    # Build translation-only correction
    T_snap        = np.eye(4)
    T_snap[:3, 3] = offset
    T_final       = T_snap @ T_icp

    return T_final, {"bindi_snap_mm": round(float(offset_mm), 2)}

# ─────────────────────────────────────────────
#  ALIGN ONE PAIR
# ─────────────────────────────────────────────
def align_pair(src_clean, tgt_clean, src_h, tgt_h):
    # Scale
    factor     = tgt_h / src_h
    src_scaled = rescale(src_clean, factor)

    # Downsample + normals
    src_d = prepare(src_scaled)
    tgt_d = prepare(tgt_clean)

    # Centroid pre-alignment: translate src centroid onto tgt centroid
    # This gives ICP a much closer starting position (pure translation, no rotation)
    src_c = np.asarray(src_d.points).mean(axis=0)
    tgt_c = np.asarray(tgt_d.points).mean(axis=0)
    T = np.eye(4)
    T[:3, 3] = tgt_c - src_c

    # Multi-scale ICP from centroid-aligned position
    for max_dist, max_iter in ICP_SCALES:
        r = icp(src_d, tgt_d, T, max_dist, max_iter)
        T = r.transformation

    # Compose: scale-around-centroid then ICP
    c            = np.asarray(src_clean.points).mean(axis=0)
    T_to_o       = np.eye(4); T_to_o[:3, 3]   = -c
    T_s          = np.eye(4); T_s[:3, :3]     *= factor
    T_from_o     = np.eye(4); T_from_o[:3, 3] =  c
    T_scale      = T_from_o @ T_s @ T_to_o
    T_final      = T @ T_scale

    # Final bindi snap — translate so red nose marks align exactly
    T_final, bindi_stats = bindi_snap(src_clean, tgt_clean, T_final)

    stats = {
        "scale_factor": round(factor,            5),
        "fitness":      round(float(r.fitness),  4),
        "rmse_mm":      round(float(r.inlier_rmse), 3),
    }
    stats.update(bindi_stats)
    return T_final, stats


# ─────────────────────────────────────────────
#  PROCESS ONE PATIENT
# ─────────────────────────────────────────────
def process_patient(patient_id: str, scan_paths: list[Path]) -> list[dict]:
    print(f"\n{'='*55}")
    print(f"  Patient: {patient_id}  ({len(scan_paths)} scans)")
    print(f"{'='*55}")

    scan_paths = sorted(scan_paths,
                        key=lambda p: (parse_scan_name(p.name)["day"],
                                       parse_scan_name(p.name)["camera"]))
    for p in scan_paths:
        info = parse_scan_name(p.name)
        tag  = "  ← reference" if p == scan_paths[0] else ""
        print(f"  day {info['day']:>3}  cam {info['camera']:<4}  {p.name}{tag}")

    # Load + clean
    print("\n  [1/3] Loading and cleaning...")
    originals, cleaned = [], []
    for path in scan_paths:
        orig = load_pcd(path)
        cl   = clean(orig)
        originals.append(orig)
        cleaned.append(cl)
        print(f"    {path.name}: {len(orig.points):,} → {len(cl.points):,} pts")

    # Measure scale
    print("\n  [2/3] Measuring face scale (3D geometric mean)...")
    heights = []
    for path, cl in zip(scan_paths, cleaned):
        h = face_scale(cl)
        heights.append(h)
        print(f"    {path.name}: scale={h:.1f}")

    ref_path = scan_paths[0]
    ref_info = parse_scan_name(ref_path.name)
    ref_h    = heights[0]
    print(f"\n  Reference: {ref_path.name}  (scale={ref_h:.1f})")

    # Save reference with _icp suffix — direct read/write, no processing
    ref_out_name = ref_path.stem + OUTPUT_SUFFIX + ".ply"
    ref_out_path = ref_path.parent / ref_out_name
    _ref_raw = o3d.io.read_point_cloud(str(ref_path))
    o3d.io.write_point_cloud(str(ref_out_path), _ref_raw)
    print(f"  Reference saved  → {ref_out_path}")

    results_log = [{
        "patient": patient_id, "file": ref_path.name,
        "day": ref_info["day"], "camera": ref_info["camera"],
        "role": "reference", "scale_factor": 1.0,
        "fitness": 1.0, "rmse_mm": 0.0,
        "output": ref_out_name,
    }]
    transforms = [np.eye(4)]

    # Align
    print("\n  [3/3] Scale + ICP alignment...")
    for i in range(1, len(scan_paths)):
        src_path = scan_paths[i]
        src_info = parse_scan_name(src_path.name)
        print(f"\n    {src_path.name}  →  {ref_path.name}")

        T_final, stats = align_pair(cleaned[i], cleaned[0], heights[i], heights[0])
        transforms.append(T_final)

        status = "✓ good" if stats["fitness"] > 0.80 else \
                 ("~ ok"  if stats["fitness"] > 0.55 else "⚠ low")
        print(f"    scale: {stats['scale_factor']:.5f}   "
              f"fitness: {stats['fitness']:.4f}   "
              f"RMSE: {stats['rmse_mm']:.3f} mm   {status}")

        out_name = src_path.stem + OUTPUT_SUFFIX + ".ply"
        out_path = src_path.parent / out_name
        aligned  = o3d.geometry.PointCloud(originals[i])
        aligned.transform(T_final)
        o3d.io.write_point_cloud(str(out_path), aligned)
        print(f"    Saved → {out_name}")

        results_log.append({
            "patient": patient_id, "file": src_path.name,
            "day": src_info["day"], "camera": src_info["camera"],
            "role": "aligned", **stats, "output": out_name,
        })

    # Merged
    print("\n  Building merged cloud...")
    merged = o3d.geometry.PointCloud()
    for orig, T in zip(originals, transforms):
        c = o3d.geometry.PointCloud(orig)
        merged += c.transform(T)
    merged = merged.voxel_down_sample(VOXEL_SIZE * 0.5)
    merged_path = scan_paths[0].parent / f"{patient_id}_merged_icp.ply"
    o3d.io.write_point_cloud(str(merged_path), merged)
    print(f"  Saved merged → {merged_path.name}  ({len(merged.points):,} pts)")

    return results_log


# ─────────────────────────────────────────────
#  CSV REPORT
# ─────────────────────────────────────────────
def save_report(results: list[dict], path: Path):
    fields = ["patient","file","day","camera","role",
              "scale_factor","fitness","rmse_mm","bindi_snap_mm","output"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\n  Report → {path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",    default=None)
    parser.add_argument("--patient", default=None)
    args = parser.parse_args()

    if args.root:
        root = Path(args.root).expanduser().resolve()
        print(f"Root (manual): {root}")
    else:
        root = find_nahidw_root(Path(__file__))
        print(f"Root (auto-detected): {root}")

    print("\nDiscovering patients and scans...")
    all_patients = discover_patients(root)
    if not all_patients:
        print("No scans found."); return

    if args.patient:
        pid = args.patient
        if pid not in all_patients:
            print(f"'{pid}' not found. Available: {list(all_patients.keys())}"); return
        all_patients = {pid: all_patients[pid]}

    print(f"\nFound {len(all_patients)} patient(s):")
    for pid, paths in all_patients.items():
        print(f"  {pid}: {len(paths)} scan(s)")

    all_results = []
    for pid, scan_paths in all_patients.items():
        try:
            all_results.extend(process_patient(pid, scan_paths))
        except Exception as e:
            print(f"\n  ERROR processing {pid}: {e}")
            import traceback; traceback.print_exc()

    if all_results:
        report_path = root / "Dataset" / "3d_scans" / "ply" / "alignment_report.csv"
        save_report(all_results, report_path)

    aligned = [r for r in all_results if r["role"] == "aligned"]
    if aligned:
        avg_f = np.mean([r["fitness"] for r in aligned])
        avg_r = np.mean([r["rmse_mm"] for r in aligned])
        print(f"\n{'='*55}")
        print(f"  Done!  {len(aligned)} scan(s) aligned")
        print(f"  Avg fitness: {avg_f:.4f}   Avg RMSE: {avg_r:.3f} mm")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()
