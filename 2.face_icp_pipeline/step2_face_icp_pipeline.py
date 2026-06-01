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

Usage:
    python face_icp_nahid.py                   # all patients, PLY only (default)
    python face_icp_nahid.py --format obj      # OBJ only
    python face_icp_nahid.py --format both     # PLY + OBJ
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

NOISE_Y_MIN    = None
SCALE_PCT_HI   = 95
SCALE_PCT_LO   = 5
MARKER_R_MIN   = 0.5
MARKER_G_MAX   = 0.3
MARKER_B_MAX   = 0.3
MARKER_CLUSTER_MM = 15
ICP_SCALES = [
    (40.0, 100),
    (20.0, 150),
    (10.0, 200),
    (5.0,  200),
    (2.0,  200),
]
VOXEL_SIZE    = 2.0
NORMAL_RADIUS = 6.0
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


def discover_patients(root: Path, fmt: str = "ply") -> dict[str, list[Path]]:
    """Find *_oriented.ply / *_oriented.obj files per patient."""
    scan_root = root / "Dataset" / "3d_scans"
    patients  = {}

    def _collect(base_dir, ext):
        if not base_dir.exists():
            return
        for pat_dir in sorted(base_dir.iterdir()):
            if not pat_dir.is_dir():
                continue
            # Prefer *_oriented files; fall back to *_aligned
            oriented = {f.stem.replace("_oriented", ""): f
                        for f in pat_dir.glob("*_oriented" + ext)
                        if "merged" not in f.stem}
            fallback = {f.stem: f
                        for f in pat_dir.glob("*_aligned" + ext)
                        if OUTPUT_SUFFIX not in f.stem
                        and "merged"  not in f.stem
                        and "backup"  not in f.stem}
            combined = {**fallback, **oriented}
            files = sorted(
                [f for f in combined.values() if parse_scan_name(f.name)],
                key=lambda f: f.name)
            if files:
                patients.setdefault(pat_dir.name, []).extend(files)

    if fmt in ("ply", "both"):
        _collect(scan_root / "ply", ".ply")
    if fmt in ("obj", "both"):
        _collect(scan_root / "obj", ".obj")

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
        f.write("# " + out_path.stem + "\n# " + str(len(pts)) + " points\n\n")
        rgb = colors / 255.0
        for i in range(len(pts)):
            f.write("v %.4f %.4f %.4f %.4f %.4f %.4f\n" % (
                pts[i,0], pts[i,1], pts[i,2],
                rgb[i,0], rgb[i,1], rgb[i,2]))


def _read_pcd(path: Path) -> o3d.geometry.PointCloud:
    """Read PLY or OBJ into a PointCloud."""
    if path.suffix.lower() == ".obj":
        return _read_obj(path)
    mesh = o3d.io.read_triangle_mesh(str(path))
    return (mesh.sample_points_poisson_disk(40_000)
            if len(mesh.triangles) > 0
            else o3d.io.read_point_cloud(str(path)))


def _write_pcd(pcd: o3d.geometry.PointCloud, out_path: Path):
    """Write PLY or OBJ from a PointCloud."""
    if out_path.suffix.lower() == ".obj":
        _write_obj(pcd, out_path)
    else:
        o3d.io.write_point_cloud(str(out_path), pcd)


# ─────────────────────────────────────────────
#  STEP 2 — CLEAN  ← unchanged
# ─────────────────────────────────────────────
def load_pcd(path: Path) -> o3d.geometry.PointCloud:
    pcd = _read_pcd(path)
    if len(pcd.points) == 0:
        raise ValueError(f"No points in {path.name}")
    return pcd


def clean(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    if NOISE_Y_MIN is not None:
        y_thresh = NOISE_Y_MIN
    else:
        y_vals = pts[:, 1]
        counts, edges = np.histogram(y_vals, bins=100)
        total    = len(y_vals)
        y_thresh = edges[0]
        for i, c in enumerate(counts):
            if c > total * 0.01:
                y_thresh = edges[i]
                break
    pcd    = pcd.select_by_index(np.where(pts[:, 1] >= y_thresh)[0])
    tmp    = pcd.voxel_down_sample(3.0)
    labels = np.array(tmp.cluster_dbscan(eps=20, min_points=5, print_progress=False))
    if labels.max() >= 0:
        cl   = np.asarray(tmp.points)[labels == np.bincount(labels[labels >= 0]).argmax()]
        PAD  = 15
        p    = np.asarray(pcd.points)
        mask = (
            (p[:, 0] >= cl[:, 0].min() - PAD) & (p[:, 0] <= cl[:, 0].max() + PAD) &
            (p[:, 1] >= cl[:, 1].min() - PAD) & (p[:, 1] <= cl[:, 1].max() + PAD) &
            (p[:, 2] >= cl[:, 2].min() - PAD) & (p[:, 2] <= cl[:, 2].max() + PAD)
        )
        pcd = pcd.select_by_index(np.where(mask)[0])
    return pcd


# ─────────────────────────────────────────────
#  STEP 3 — SCALE  ← unchanged
# ─────────────────────────────────────────────
def face_scale(pcd: o3d.geometry.PointCloud) -> float:
    pts   = np.asarray(pcd.points)
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
#  STEP 4 — DOWNSAMPLE + NORMALS  ← unchanged
# ─────────────────────────────────────────────
def prepare(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    d = pcd.voxel_down_sample(VOXEL_SIZE)
    d.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=30))
    d.orient_normals_consistent_tangent_plane(k=15)
    return d


# ─────────────────────────────────────────────
#  STEP 5 — ICP  ← unchanged
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
#  STEP 6 — BINDI SNAP  ← unchanged
# ─────────────────────────────────────────────
def find_bindi(pcd: o3d.geometry.PointCloud) -> np.ndarray | None:
    if not pcd.has_colors():
        return None
    pts    = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    mask   = ((colors[:, 0] > MARKER_R_MIN) &
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
        return red_pts.mean(axis=0)


def bindi_snap(src_orig, tgt_orig, T_icp):
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
    offset    = tgt_bindi - src_bindi
    offset_mm = np.linalg.norm(offset)
    print(f"    Bindi src: ({src_bindi[0]:.1f}, {src_bindi[1]:.1f}, {src_bindi[2]:.1f})")
    print(f"    Bindi tgt: ({tgt_bindi[0]:.1f}, {tgt_bindi[1]:.1f}, {tgt_bindi[2]:.1f})")
    print(f"    Bindi residual offset: {offset_mm:.2f} mm  → snapping")
    T_snap        = np.eye(4)
    T_snap[:3, 3] = offset
    return T_snap @ T_icp, {"bindi_snap_mm": round(float(offset_mm), 2)}


# ─────────────────────────────────────────────
#  ALIGN ONE PAIR  ← unchanged
# ─────────────────────────────────────────────
def align_pair(src_clean, tgt_clean, src_h, tgt_h):
    factor     = tgt_h / src_h
    src_scaled = rescale(src_clean, factor)
    src_d      = prepare(src_scaled)
    tgt_d      = prepare(tgt_clean)
    src_c      = np.asarray(src_d.points).mean(axis=0)
    tgt_c      = np.asarray(tgt_d.points).mean(axis=0)
    T          = np.eye(4)
    T[:3, 3]   = tgt_c - src_c
    for max_dist, max_iter in ICP_SCALES:
        r = icp(src_d, tgt_d, T, max_dist, max_iter)
        T = r.transformation
    c        = np.asarray(src_clean.points).mean(axis=0)
    T_to_o   = np.eye(4); T_to_o[:3, 3]   = -c
    T_s      = np.eye(4); T_s[:3, :3]     *= factor
    T_from_o = np.eye(4); T_from_o[:3, 3] =  c
    T_scale  = T_from_o @ T_s @ T_to_o
    T_final  = T @ T_scale
    T_final, bindi_stats = bindi_snap(src_clean, tgt_clean, T_final)
    stats = {
        "scale_factor": round(factor,           5),
        "fitness":      round(float(r.fitness), 4),
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
    ext      = ref_path.suffix                        # .ply or .obj
    print(f"\n  Reference: {ref_path.name}  (scale={ref_h:.1f})")

    # Save reference with _icp suffix
    ref_out_path = ref_path.parent / (ref_path.stem + OUTPUT_SUFFIX + ext)
    _write_pcd(load_pcd(ref_path), ref_out_path)
    print(f"  Reference saved  → {ref_out_path}")

    results_log = [{
        "patient": patient_id, "file": ref_path.name,
        "day": ref_info["day"], "camera": ref_info["camera"],
        "role": "reference", "scale_factor": 1.0,
        "fitness": 1.0, "rmse_mm": 0.0,
        "output": ref_out_path.name,
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

        status   = "✓ good" if stats["fitness"] > 0.80 else \
                   ("~ ok"  if stats["fitness"] > 0.55 else "⚠ low")
        print(f"    scale: {stats['scale_factor']:.5f}   "
              f"fitness: {stats['fitness']:.4f}   "
              f"RMSE: {stats['rmse_mm']:.3f} mm   {status}")

        out_path = src_path.parent / (src_path.stem + OUTPUT_SUFFIX + ext)
        aligned  = o3d.geometry.PointCloud(originals[i])
        aligned.transform(T_final)
        _write_pcd(aligned, out_path)
        print(f"    Saved → {out_path.name}")

        results_log.append({
            "patient": patient_id, "file": src_path.name,
            "day": src_info["day"], "camera": src_info["camera"],
            "role": "aligned", **stats, "output": out_path.name,
        })

    # Merged
    print("\n  Building merged cloud...")
    merged = o3d.geometry.PointCloud()
    for orig, T in zip(originals, transforms):
        c = o3d.geometry.PointCloud(orig)
        merged += c.transform(T)
    merged      = merged.voxel_down_sample(VOXEL_SIZE * 0.5)
    merged_path = ref_path.parent / f"{patient_id}_merged_icp{ext}"
    _write_pcd(merged, merged_path)
    print(f"  Saved merged → {merged_path.name}  ({len(merged.points):,} pts)")

    return results_log


# ─────────────────────────────────────────────
#  CSV REPORT
# ─────────────────────────────────────────────
def save_report(results: list[dict], root: Path, fmt: str):
    fields = ["patient","file","day","camera","role",
              "scale_factor","fitness","rmse_mm","bindi_snap_mm","output"]
    # Save report alongside the data — in ply/ for PLY, obj/ for OBJ
    folder = "obj" if fmt == "obj" else "ply"
    path   = root / "Dataset" / "3d_scans" / folder / "alignment_report.csv"
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
    parser.add_argument("--format",  default="both",
                        choices=["ply", "obj", "both"],
                        help="Input format to process (default: both)")
    args = parser.parse_args()

    if args.root:
        root = Path(args.root).expanduser().resolve()
        print(f"Root (manual): {root}")
    else:
        root = find_nahidw_root(Path(__file__))
        print(f"Root (auto-detected): {root}")

    # When format=both, run PLY and OBJ as two separate passes
    # so each gets its own merged file and report in the correct folder
    formats = ["ply", "obj"] if args.format == "both" else [args.format]

    for fmt in formats:
        print(f"\nDiscovering patients and scans [{fmt.upper()}]...")
        all_patients = discover_patients(root, fmt=fmt)
        if not all_patients:
            print(f"  No {fmt.upper()} scans found.")
            continue

        if args.patient:
            pid = args.patient
            if pid not in all_patients:
                print(f"  '{pid}' not found.")
                continue
            all_patients = {pid: all_patients[pid]}

        print(f"  Found {len(all_patients)} patient(s):")
        for pid, paths in all_patients.items():
            print(f"    {pid}: {len(paths)} scan(s)")

        all_results = []
        for pid, scan_paths in all_patients.items():
            try:
                all_results.extend(process_patient(pid, scan_paths))
            except Exception as e:
                print(f"\n  ERROR processing {pid}: {e}")
                import traceback; traceback.print_exc()

        if all_results:
            save_report(all_results, root, fmt)

        aligned = [r for r in all_results if r["role"] == "aligned"]
        if aligned:
            avg_f = sum(r["fitness"] for r in aligned) / len(aligned)
            avg_r = sum(r["rmse_mm"] for r in aligned) / len(aligned)
            print(f"\n{'='*55}")
            print(f"  Done [{fmt.upper()}]!  {len(aligned)} scan(s) aligned")
            print(f"  Avg fitness: {avg_f:.4f}   Avg RMSE: {avg_r:.3f} mm")
            print(f"{'='*55}")

if __name__ == "__main__":
    main()





# python face_icp_nahid.py                          # PLY + OBJ (default)
# python face_icp_nahid.py --format ply             # PLY only
# python face_icp_nahid.py --format obj             # OBJ only
# python face_icp_nahid.py --format obj --patient pat1   # OBJ, one patient
# python face_icp_nahid.py --root D:/NahidW         # manual root