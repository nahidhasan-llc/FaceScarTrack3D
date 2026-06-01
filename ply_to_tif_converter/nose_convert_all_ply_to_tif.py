#!/usr/bin/env python3
"""
ply_to_tif.py
=============
Converts forward-facing PLY or OBJ point clouds to unwrapped cylindrical TIFs.

Assumption (forward-facing convention):
    Height axis   = Y   (Y_min = chin,  Y_max = top of head)
    Nose direction = +Z  (face looks toward +Z)
    Radial plane  = XZ

    theta = atan2(X, Z)   ->  theta=0 at nose (+Z)
    shift by pi           ->  nose lands at exact centre column (col = width/2)

READS FROM:
  D:/NahidW/Dataset/3d_scans/ply/   (*.ply)
  D:/NahidW/Dataset/3d_scans/obj/   (*.obj)
    pat1/  pat1day0C.ply  pat1day28A.ply  ...
    pat2/  pat2day0A.ply  ...

WRITES TO:
  D:/NahidW/Dataset/3d_scans/tif_unwrapped/
    pat1/  pat1day0C.tif  pat1day28A.tif  ...
    pat2/  pat2day0A.tif  ...

USAGE:
  python ply_to_tif.py                        # all PLY files
  python ply_to_tif.py --format obj           # all OBJ files
  python ply_to_tif.py --format both          # PLY + OBJ
  python ply_to_tif.py --patient pat1         # one patient
  python ply_to_tif.py --width 512            # output width (default 512)
  python ply_to_tif.py --debug                # green centre line for verification
  python ply_to_tif.py --overwrite            # redo existing files

REQUIREMENTS:
  pip install numpy pillow
"""

import sys, re, argparse
import numpy as np
from pathlib import Path
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────

PLY_DIR = Path("D:/NahidW/Dataset/3d_scans/ply")
OBJ_DIR = Path("D:/NahidW/Dataset/3d_scans/obj")
OUT_DIR = Path("D:/NahidW/Dataset/3d_scans/tif_unwrapped")

DEFAULT_WIDTH = 512


# ─────────────────────────────────────────────────────────────────────────────
#  PLY READER
# ─────────────────────────────────────────────────────────────────────────────

def read_ply(path: Path):
    """
    Read binary-little-endian PLY with XYZ (float or double) + RGB (uchar).
    Returns:
        pts    : (N,3) float64
        colors : (N,3) uint8
    """
    with open(path, "rb") as f:
        raw = f.read()

    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header     = raw[:header_end].decode("ascii", errors="replace")
    lines      = [l.strip() for l in header.split("\n") if l.strip()]

    fmt = next(l for l in lines if l.startswith("format"))
    if "binary_little_endian" not in fmt:
        raise ValueError(f"Only binary_little_endian PLY supported. Got: {fmt}")

    n_verts = int(next(
        re.search(r"element vertex (\d+)", l).group(1)
        for l in lines if l.startswith("element vertex")
    ))

    TYPE_MAP = {
        "float":   ("f4", 4), "float32": ("f4", 4),
        "double":  ("f8", 8), "float64": ("f8", 8),
        "uchar":   ("u1", 1), "uint8":   ("u1", 1),
        "char":    ("i1", 1), "short":   ("i2", 2),
        "ushort":  ("u2", 2), "int":     ("i4", 4),
        "uint":    ("u4", 4),
    }
    props      = [l.split() for l in lines if l.startswith("property")]
    dtype_list = [(pname, f"<{TYPE_MAP[ptype][0]}") for _, ptype, pname in props]
    dt         = np.dtype(dtype_list)

    expected = n_verts * dt.itemsize
    actual   = len(raw) - header_end
    if expected != actual:
        raise ValueError(
            f"PLY size mismatch: expected {expected}B, got {actual}B "
            f"for {n_verts} vertices"
        )

    data = np.frombuffer(raw[header_end:], dtype=dt)

    pts = np.column_stack([
        data["x"].astype(np.float64),
        data["y"].astype(np.float64),
        data["z"].astype(np.float64),
    ])

    def _get(names):
        for n in names:
            if n in data.dtype.names:
                return data[n]
        raise KeyError(f"None of {names} found in PLY properties")

    colors = np.column_stack([
        _get(["red",   "r"]),
        _get(["green", "g"]),
        _get(["blue",  "b"]),
    ]).astype(np.uint8)

    return pts, colors


# ─────────────────────────────────────────────────────────────────────────────
#  OBJ READER
# ─────────────────────────────────────────────────────────────────────────────

def read_obj(path: Path):
    """
    Read OBJ file with vertex colors (v x y z r g b).
    Colors can be 0-1 float or 0-255 — auto-detected from range.
    Returns:
        pts    : (N,3) float64
        colors : (N,3) uint8
    """
    pts_list    = []
    colors_list = []

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 7:
                    # v x y z r g b
                    pts_list.append([float(parts[1]),
                                     float(parts[2]),
                                     float(parts[3])])
                    colors_list.append([float(parts[4]),
                                        float(parts[5]),
                                        float(parts[6])])
                elif len(parts) == 4:
                    # v x y z  (no colour — use grey)
                    pts_list.append([float(parts[1]),
                                     float(parts[2]),
                                     float(parts[3])])
                    colors_list.append([128.0, 128.0, 128.0])

    if not pts_list:
        raise ValueError("No vertices found in OBJ file.")

    pts    = np.array(pts_list,    dtype=np.float64)
    colors = np.array(colors_list, dtype=np.float64)

    # Auto-scale: if colours are in [0,1] range, multiply to [0,255]
    if colors.max() <= 1.0:
        colors = colors * 255.0

    return pts, colors.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  UNIFIED READER
# ─────────────────────────────────────────────────────────────────────────────

def read_pointcloud(path: Path):
    """Route to the correct reader based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return read_ply(path)
    elif suffix == ".obj":
        return read_obj(path)
    else:
        raise ValueError(f"Unsupported format '{suffix}'. Use .ply or .obj")


# ─────────────────────────────────────────────────────────────────────────────
#  UNWRAP
# ─────────────────────────────────────────────────────────────────────────────

def unwrap(pts: np.ndarray,
           colors: np.ndarray,
           nlg: int,
           debug: bool) -> Image.Image:
    """
    Unwrap a forward-facing point cloud to a cylindrical TIF.

    Convention:
        Y = height   (chin -> crown)
        Z = forward  (nose points toward +Z)
        X = sideways

        theta = atan2(X, Z)    theta=0 -> nose (+Z)
        shift by pi            nose    -> col = nlg/2 (centre)
        row   = 1 - norm(Y)    Y_max   -> row 0       (top of image)
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # ── Height ────────────────────────────────────────────────────────────
    h_min, h_max = y.min(), y.max()
    h_range      = h_max - h_min
    if h_range == 0:
        raise ValueError("All Y values identical — cannot unwrap.")

    # ── Angular coordinate ─────────────────────────────────────────────────
    theta         = (np.arctan2(x, z) + 2 * np.pi) % (2 * np.pi)
    theta_shifted = (theta + np.pi) % (2 * np.pi)   # nose -> centre col

    # ── Auto canvas height ─────────────────────────────────────────────────
    r     = np.sqrt(x**2 + z**2)
    r_med = np.median(r[r > 0])
    circ  = 2 * np.pi * r_med
    nlt   = max(64, int(nlg * h_range / circ))

    print(f"    Height axis : Y   range={h_range:.1f}mm")
    print(f"    Nose dir    : +Z  -> col {nlg//2} (centre)")
    print(f"    Canvas      : {nlg}x{nlt}  r_med={r_med:.1f}mm")

    # ── Pixel mapping ──────────────────────────────────────────────────────
    col = (theta_shifted / (2 * np.pi) * nlg).astype(int) % nlg
    row = np.round((1.0 - (y - h_min) / h_range) * (nlt - 1)).astype(int)
    row = np.clip(row, 0, nlt - 1)

    # ── Accumulate (average overlapping points) ────────────────────────────
    acc = np.zeros((nlt, nlg, 3), np.float64)
    cnt = np.zeros((nlt, nlg),    np.int32)
    np.add.at(acc, (row, col), colors.astype(np.float64))
    np.add.at(cnt, (row, col), 1)

    mask   = cnt > 0
    canvas = np.zeros((nlt, nlg, 3), np.uint8)
    canvas[mask] = (acc[mask] / cnt[mask, np.newaxis]).astype(np.uint8)

    cov = mask.sum() / (nlt * nlg) * 100
    print(f"    Coverage    : {cov:.1f}%   max overlap: {cnt.max()} pts/px")

    # ── Fill empty columns from nearest neighbour ──────────────────────────
    col_has_data = mask.any(axis=0)
    for ec in np.where(~col_has_data)[0]:
        for dist in range(1, nlg):
            left  = (ec - dist) % nlg
            right = (ec + dist) % nlg
            if col_has_data[left]:
                canvas[:, ec] = canvas[:, left]
                break
            if col_has_data[right]:
                canvas[:, ec] = canvas[:, right]
                break

    # ── Smooth sparse columns (< 20% of mean density) ─────────────────────
    col_count    = cnt.sum(axis=0)
    mean_density = col_count[col_has_data].mean() if col_has_data.any() else 1
    for sc in np.where((col_count > 0) & (col_count < mean_density * 0.20))[0]:
        left  = (sc - 1) % nlg
        right = (sc + 1) % nlg
        nbr   = ((canvas[:, left].astype(np.float32) +
                  canvas[:, right].astype(np.float32)) / 2)
        canvas[:, sc] = (canvas[:, sc] * 0.3 + nbr * 0.7).astype(np.uint8)

    # ── Debug: green centre line at nose position ──────────────────────────
    if debug:
        canvas[:, nlg // 2 - 1] = [0, 200, 0]
        canvas[:, nlg // 2]     = [0, 255, 0]
        canvas[:, nlg // 2 + 1] = [0, 200, 0]

    return Image.fromarray(canvas, "RGB")


# ─────────────────────────────────────────────────────────────────────────────
#  FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_files(fmt: str, patient_filter=None):
    """
    Find all .ply / .obj files under the relevant input directories.
    Searches one level deep: <dir>/<patient>/<scan>.ext
    If the same scan exists in both PLY and OBJ, PLY takes priority.
    Returns sorted list of (patient_id, scan_name, file_path).
    """
    # Use a dict keyed by (patient_id, scan_name) so duplicates resolve cleanly.
    # Collect OBJ first (lower priority), then PLY (overwrites OBJ if both exist).
    candidates = {}

    def _collect(base_dir, ext):
        if not base_dir.exists():
            print(f"  ! Directory not found, skipping: {base_dir}")
            return
        for fpath in sorted(base_dir.glob(f"*/*{ext}")):
            patient_id = fpath.parent.name.lower()
            if patient_filter and patient_id != patient_filter.lower():
                continue
            key = (patient_id, fpath.stem)
            existing = candidates.get(key)
            if existing is None or ext == ".ply":   # PLY always wins
                candidates[key] = fpath

    if fmt in ("obj", "both"):
        _collect(OBJ_DIR, ".obj")   # lower priority
    if fmt in ("ply", "both"):
        _collect(PLY_DIR, ".ply")   # higher priority, overwrites OBJ

    return [
        (patient_id, scan_name, fpath)
        for (patient_id, scan_name), fpath
        in sorted(candidates.items(), key=lambda kv: kv[0])
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Forward-facing PLY/OBJ -> unwrapped cylindrical TIF"
    )
    p.add_argument("--format",    type=str, default="both",
                   choices=["ply", "obj", "both"],
                   help="Input format to process (default: both)")
    p.add_argument("--patient",   type=str, default=None,
                   help="Process only this patient (e.g. pat1)")
    p.add_argument("--width",     type=int, default=DEFAULT_WIDTH,
                   help=f"Output image width in pixels (default: {DEFAULT_WIDTH})")
    p.add_argument("--debug",     action="store_true",
                   help="Draw green centre line at nose position for verification")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing TIF files")
    return p.parse_args()


def main():
    args = parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PLY/OBJ -> Unwrapped Cylindrical TIF")
    print(f"  Format : {args.format.upper()}")
    print(f"  PLY in : {PLY_DIR}")
    print(f"  OBJ in : {OBJ_DIR}")
    print(f"  Output : {OUT_DIR}")
    print(f"  Width  : {args.width}px  (height auto)")
    print(f"  Debug  : {'on -- green centre line at nose' if args.debug else 'off'}")
    print(f"{'='*60}")

    files = discover_files(args.format, patient_filter=args.patient)
    if not files:
        print(f"\n  No files found.")
        print(f"  PLY expected under : {PLY_DIR}/<patient>/<scan>.ply")
        print(f"  OBJ expected under : {OBJ_DIR}/<patient>/<scan>.obj")
        sys.exit(1)

    print(f"\n  Found {len(files)} file(s)\n")

    report, prev_pat = [], None

    for patient_id, scan_name, fpath in files:

        if patient_id != prev_pat:
            print(f"{'─'*60}")
            print(f"  Patient: {patient_id.upper()}")
            prev_pat = patient_id

        tif_out = OUT_DIR / patient_id / f"{scan_name}.tif"
        if tif_out.exists() and not args.overwrite:
            print(f"  skip  {scan_name}  ({tif_out.stat().st_size/1e6:.1f}MB) -- skipping")
            report.append({"scan": scan_name, "status": "skipped"})
            continue

        print(f"  -> {scan_name}  [{fpath.suffix.upper()}  {fpath.stat().st_size/1e6:.1f}MB]")
        try:
            pts, colors = read_pointcloud(fpath)
            print(f"    {len(pts):,} pts  "
                  f"x=[{pts[:,0].min():.0f}, {pts[:,0].max():.0f}]  "
                  f"y=[{pts[:,1].min():.0f}, {pts[:,1].max():.0f}]  "
                  f"z=[{pts[:,2].min():.0f}, {pts[:,2].max():.0f}]")

            img = unwrap(pts, colors, nlg=args.width, debug=args.debug)

            tif_out.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(tif_out), format="TIFF", compression="tiff_lzw")
            sz = tif_out.stat().st_size
            print(f"    ok  {img.width}x{img.height}px  "
                  f"({sz/1e6:.1f}MB)  ->  {tif_out.name}")
            report.append({"scan": scan_name, "status": "success"})

        except Exception as e:
            import traceback
            print(f"    fail  {e}")
            traceback.print_exc()
            report.append({"scan": scan_name, "status": "failed", "error": str(e)})

    ok     = sum(1 for r in report if r["status"] == "success")
    skip   = sum(1 for r in report if r["status"] == "skipped")
    failed = [r for r in report if r["status"] == "failed"]

    print(f"\n{'='*60}")
    print(f"  Converted : {ok}   Skipped : {skip}   Failed : {len(failed)}")
    if failed:
        for r in failed:
            print(f"    fail {r['scan']} -- {r['error']}")
    print(f"  Output    : {OUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()