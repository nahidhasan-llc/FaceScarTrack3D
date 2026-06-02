#!/usr/bin/env python3
"""
ply_to_tif.py
=============
Reverses the Cyberware cylindrical scan pipeline:
  PLY (XYZ + RGB point cloud)  →  unwrapped cylindrical TIF (color texture)

The original pipeline was:
  Cyberware range file  →  cylindrical coords  →  Cartesian XYZ  →  PLY
  TIF (NLT rows × NLG cols, height×angular) stored color at (tif_row, tif_col)

This script reverses that:
  PLY XYZ  →  cylindrical (theta, z)  →  pixel grid (NLG × NLT)  →  TIF

READS FROM:
  Dataset/3d_scans/ply/
    pat1/
        pat1day0C.ply
        pat1day28A.ply
    pat2/
        pat2day0A.ply

WRITES TO:
  Dataset/3d_scans/tif_unwrapped/
    pat1/
        pat1day0C.tif
        pat1day28A.tif
    pat2/
        pat2day0A.tif

USAGE:
  python ply_to_tif.py
  python ply_to_tif.py --patient pat1
  python ply_to_tif.py --ply_dir "D:/NahidW/Dataset/3d_scans/ply"
  python ply_to_tif.py --resolution 1024 2048
  python ply_to_tif.py --overwrite

REQUIREMENTS:
  pip install numpy pillow tqdm
"""

import sys, os, re, argparse, struct
import numpy as np
from pathlib import Path
from PIL import Image

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  (mirrors original scanner constants)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PLY_DIR = Path("D:/NahidW/Dataset/3d_scans/ply")
DEFAULT_OUT_DIR = Path("D:/NahidW/Dataset/3d_scans/tif_unwrapped")

# Output will be written to:
#   D:/NahidW/Dataset/3d_scans/tif_unwrapped/
#     pat1/
#       pat1day0C.tif
#       pat1day28A.tif
#     pat2/
#       pat2day0A.tif

# Original scanner physical dimensions (used only for aspect-ratio defaults)
SCANNER_HEIGHT_MM = 18 * 25.4   # 457.2 mm
SCANNER_RADIUS_MM =  9 * 25.4   # 228.6 mm

# Default output canvas size  (NLG=angular cols, NLT=height rows)
# Matches the original Cyberware resolution ~512 angular × 256 height
DEFAULT_NLG = 512   # angular (theta) columns → image width
DEFAULT_NLT = 256   # height  (z)    rows    → image height

SCAN_RE = re.compile(
    r"^(?P<patient>pat\d+)day(?P<day>\d+)(?P<variant>[A-Z][A-Z0-9]?)$",
    re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
#  PLY READER  (handles float32 and float64 XYZ, uchar RGB)
# ─────────────────────────────────────────────────────────────────────────────

def read_ply(path: Path):
    """
    Read a binary-little-endian PLY with XYZ + RGB per vertex.
    Returns (pts, colors):
      pts    : (N, 3) float64  — XYZ in original units (mm)
      colors : (N, 3) uint8   — RGB
    Supports property float / property double for XYZ.
    """
    with open(path, "rb") as f:
        raw = f.read()

    # ── Parse header ──────────────────────────────────────────────────────────
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:header_end].decode("ascii", errors="replace")
    lines  = [l.strip() for l in header.split("\n") if l.strip()]

    # Format check
    fmt_line = next(l for l in lines if l.startswith("format"))
    if "binary_little_endian" not in fmt_line:
        raise ValueError(f"Only binary_little_endian PLY supported. Got: {fmt_line}")

    # Vertex count
    n_verts = int(next(
        re.search(r"element vertex (\d+)", l).group(1)
        for l in lines if l.startswith("element vertex")
    ))

    # Properties
    props = [l.split() for l in lines if l.startswith("property")]
    # props[i] = ['property', type, name]

    TYPE_MAP = {
        "float":   ("f4", 4),
        "float32": ("f4", 4),
        "double":  ("f8", 8),
        "float64": ("f8", 8),
        "uchar":   ("u1", 1),
        "uint8":   ("u1", 1),
        "char":    ("i1", 1),
        "short":   ("i2", 2),
        "ushort":  ("u2", 2),
        "int":     ("i4", 4),
        "uint":    ("u4", 4),
    }

    dtype_fields = []
    for _, ptype, pname in props:
        dt_str, _ = TYPE_MAP[ptype]
        dtype_fields.append((pname, f"<{dt_str}"))

    dt = np.dtype(dtype_fields)

    # Sanity-check expected vs actual data size
    expected = n_verts * dt.itemsize
    actual   = len(raw) - header_end
    if expected != actual:
        raise ValueError(
            f"PLY data size mismatch: expected {expected} bytes, "
            f"found {actual} bytes for {n_verts} vertices with dtype {dt}"
        )

    data = np.frombuffer(raw[header_end:], dtype=dt)

    # Extract XYZ — look for x/y/z regardless of float vs double
    pts = np.column_stack([
        data["x"].astype(np.float64),
        data["y"].astype(np.float64),
        data["z"].astype(np.float64),
    ])

    # Extract RGB — field names may be red/green/blue or r/g/b
    def _get_color(d, names):
        for n in names:
            if n in d.dtype.names:
                return d[n]
        raise KeyError(f"None of {names} found in PLY properties")

    colors = np.column_stack([
        _get_color(data, ["red",   "r"]),
        _get_color(data, ["green", "g"]),
        _get_color(data, ["blue",  "b"]),
    ]).astype(np.uint8)

    return pts, colors


# ─────────────────────────────────────────────────────────────────────────────
#  CYLINDRICAL UNWRAP  (exact inverse of original load_scan)
# ─────────────────────────────────────────────────────────────────────────────

def ply_to_tif(pts: np.ndarray,
               colors: np.ndarray,
               nlg: int = DEFAULT_NLG,
               nlg_out: int = None,
               nlt_out: int = None) -> Image.Image:
    """
    Convert an XYZ+RGB point cloud back to an unwrapped cylindrical TIF.

    The original mapping was:
        X = r * cos(theta)    theta = col * (2π / NLG)
        Y = r * sin(theta)    z     = row * (height_mm / NLT)
        Z = z

        TIF row  = (NLT - 1) - int(z_index * NLT / NLT)   ← top-down flip
        TIF col  = int(theta_index * NLG_tif / NLG)

    Inverse:
        theta   = atan2(Y, X)            ∈ [-π, π]  → normalise to [0, 2π]
        z       = Z                      ∈ [z_min, z_max]

        col_tif = round(theta / (2π) * NLG_out)           (angular position)
        row_tif = round((1 - (z - z_min) / z_range) * (NLT_out - 1))
                  ↑ flipped because TIF is top-down, scanner is bottom-up

    Parameters
    ----------
    pts      : (N,3) XYZ array
    colors   : (N,3) RGB uint8 array
    nlg      : angular resolution of output image (columns)
    nlg_out  : alias for nlg (overrides)
    nlt_out  : height resolution of output image (rows); if None, auto from
               aspect ratio of coordinate ranges

    Returns
    -------
    PIL Image (RGB)
    """
    if nlg_out is not None:
        nlg = nlg_out

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # ── Cylindrical coordinates ────────────────────────────────────────────
    theta = np.arctan2(y, x)                 # [-π, π]
    theta = (theta + 2 * np.pi) % (2 * np.pi)  # [0, 2π)

    z_min, z_max = z.min(), z.max()
    z_range = z_max - z_min
    if z_range == 0:
        raise ValueError("All Z values are identical — cannot unwrap.")

    # ── Auto NLT from aspect ratio if not specified ────────────────────────
    if nlt_out is None:
        # Approximate circumference vs height ratio
        r_vals = np.sqrt(x**2 + y**2)
        r_mean = np.nanmedian(r_vals[r_vals > 0])
        circumference = 2 * np.pi * r_mean
        aspect = circumference / z_range
        nlt_out = max(64, int(nlg / aspect))
        print(f"    Auto NLT = {nlt_out}  "
              f"(r_median={r_mean:.1f}mm, circ={circumference:.1f}mm, "
              f"z_range={z_range:.1f}mm, aspect={aspect:.2f})")

    nlt = nlt_out

    # ── Map to pixel coordinates ───────────────────────────────────────────
    # col: angular position, wraps [0, NLG)
    col = (theta / (2 * np.pi) * nlg).astype(int) % nlg

    # row: height position, top-down flip (low Z → bottom of image → high row)
    row = np.round((1.0 - (z - z_min) / z_range) * (nlt - 1)).astype(int)
    row = np.clip(row, 0, nlt - 1)

    # ── Accumulate into canvas ────────────────────────────────────────────
    # Use float accumulation to handle multiple points mapping to same pixel
    canvas_sum   = np.zeros((nlt, nlg, 3), dtype=np.float64)
    canvas_count = np.zeros((nlt, nlg),    dtype=np.int32)

    np.add.at(canvas_sum,   (row, col), colors.astype(np.float64))
    np.add.at(canvas_count, (row, col), 1)

    # Average where multiple points hit the same pixel
    mask = canvas_count > 0
    canvas_rgb = np.zeros((nlt, nlg, 3), dtype=np.uint8)
    canvas_rgb[mask] = (canvas_sum[mask] /
                        canvas_count[mask, np.newaxis]).astype(np.uint8)

    coverage = mask.sum() / (nlt * nlg) * 100
    print(f"    Canvas: {nlg}×{nlt}  |  "
          f"Coverage: {coverage:.1f}%  |  "
          f"Max overlap: {canvas_count.max()} pts/px")

    return Image.fromarray(canvas_rgb, mode="RGB")


# ─────────────────────────────────────────────────────────────────────────────
#  FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_plys(ply_dir: Path, patient_filter=None) -> list:
    """
    Find all PLY files under ply_dir/<patient>/<scan>.ply
    Returns sorted list of (patient_id, scan_name, ply_path).
    """
    found = []
    for ply_path in sorted(ply_dir.glob("*/*.ply")):
        scan_name  = ply_path.stem
        patient_id = ply_path.parent.name.lower()
        m = SCAN_RE.match(scan_name)
        if not m:
            # Still process even if name doesn't match — just warn
            print(f"  ⚠ Unexpected scan name pattern: {scan_name} — including anyway")
        if patient_filter and patient_id != patient_filter.lower():
            continue
        found.append((patient_id, scan_name, ply_path))
    found.sort(key=lambda x: (x[0], x[1]))
    return found


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert PLY point clouds back to unwrapped cylindrical TIFs"
    )
    p.add_argument("--ply_dir",    type=str, default=None,
                   help=f"PLY input directory (default: {DEFAULT_PLY_DIR})")
    p.add_argument("--out_dir",    type=str, default=None,
                   help=f"TIF output directory (default: <ply_dir>/../tif_unwrapped)")
    p.add_argument("--patient",    type=str, default=None,
                   help="Process only this patient (e.g. pat1)")
    p.add_argument("--resolution", type=int, nargs=2, default=None,
                   metavar=("NLG", "NLT"),
                   help=f"Output image size: angular_cols height_rows "
                        f"(default: {DEFAULT_NLG} auto)")
    p.add_argument("--overwrite",  action="store_true",
                   help="Overwrite existing TIF files (default: skip)")
    return p.parse_args()


def main():
    args = parse_args()

    ply_dir = Path(args.ply_dir) if args.ply_dir else DEFAULT_PLY_DIR
    out_dir = (Path(args.out_dir) if args.out_dir
               else ply_dir.parent / "tif_unwrapped")

    nlg = DEFAULT_NLG
    nlt = None
    if args.resolution:
        nlg, nlt = args.resolution

    if not ply_dir.exists():
        print(f"PLY directory not found: {ply_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  PLY → Unwrapped Cylindrical TIF")
    print(f"  Input  : {ply_dir}")
    print(f"  Output : {out_dir}")
    print(f"  Canvas : {nlg} cols (angular) × {'auto' if nlt is None else nlt} rows (height)")
    print(f"{'='*60}")

    plys = discover_plys(ply_dir, patient_filter=args.patient)
    if not plys:
        print(f"\n  No PLY files found in {ply_dir}")
        print("  Expected structure: <ply_dir>/<patient>/<scan>.ply")
        sys.exit(1)

    print(f"\n  Found {len(plys)} PLY file(s)\n")

    report   = []
    prev_pat = None

    for patient_id, scan_name, ply_path in plys:

        if patient_id != prev_pat:
            print(f"{'─'*60}")
            print(f"  Patient: {patient_id.upper()}")
            prev_pat = patient_id

        tif_out = out_dir / patient_id / f"{scan_name}.tif"

        if tif_out.exists() and not args.overwrite:
            sz = tif_out.stat().st_size
            print(f"  ↷ {scan_name}  already exists ({sz/1e6:.1f}MB) — skipping")
            report.append({"scan": scan_name, "status": "skipped"})
            continue

        print(f"  → {scan_name}  ({ply_path.stat().st_size/1e6:.1f}MB PLY)")

        try:
            # 1. Read PLY
            pts, colors = read_ply(ply_path)
            print(f"    Loaded {len(pts):,} points  |  "
                  f"XYZ range: x={pts[:,0].min():.1f}…{pts[:,0].max():.1f}  "
                  f"y={pts[:,1].min():.1f}…{pts[:,1].max():.1f}  "
                  f"z={pts[:,2].min():.1f}…{pts[:,2].max():.1f}")

            # 2. Unwrap to TIF
            img = ply_to_tif(pts, colors, nlg_out=nlg, nlt_out=nlt)

            # 3. Save
            tif_out.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(tif_out), format="TIFF", compression="tiff_lzw")
            sz = tif_out.stat().st_size
            print(f"    ✓  Saved {img.width}×{img.height} TIF  ({sz/1e6:.1f}MB)  →  {tif_out}")
            report.append({"scan": scan_name, "status": "success",
                           "size": f"{img.width}×{img.height}"})

        except Exception as e:
            import traceback
            print(f"    ✗  {e}")
            traceback.print_exc()
            report.append({"scan": scan_name, "status": "failed", "error": str(e)})

    ok     = sum(1 for r in report if r["status"] == "success")
    skip   = sum(1 for r in report if r["status"] == "skipped")
    failed = [r for r in report if r["status"] == "failed"]

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Converted : {ok}")
    print(f"  Skipped   : {skip}  (use --overwrite to redo)")
    if failed:
        print(f"  Failed    : {len(failed)}")
        for r in failed:
            print(f"    {r['scan']} — {r['error']}")
    print(f"\n  Output    : {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()