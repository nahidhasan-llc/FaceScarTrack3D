#!/usr/bin/env python3
"""
obj_to_tif.py — OBJ + texture → unwrapped cylindrical TIF
──────────────────────────────────────────────────────────
★  CHANGE THESE LINES WHEN SWITCHING FILES  ★
"""

INPUT_OBJ  = r"D:\NahidW\Dataset\test_data\aligned_face\Pasha_guard_head.obj"
OUTPUT_TIF = r"D:\NahidW\Dataset\test_data\aligned_face\Pasha_guard_head_unwrapped.tif"

NLG          = 2048   # output width  (angular columns) — higher = more detail
NLT          = None   # output height (None = auto from geometry)
FRONT_AXIS   = 'z'    # axis the face points along
UP_AXIS      = 'y'    # vertical axis
FRONT_OFFSET = 180    # degrees: 180 = face centred in image

# ── Sampling density ─────────────────────────────────────────────
# Controls how densely each triangle is sampled from the texture.
# Higher = better quality, slower. 3 is a good default.
# 1 = fast/sparse,  3 = good quality,  6 = very dense (slow)
SAMPLE_DENSITY = 3

# ─────────────────────────────────────────────────────────────────
import sys, time
import numpy as np
from pathlib import Path
from PIL import Image

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "-q"])
    from scipy.ndimage import distance_transform_edt


# ── OBJ + texture loader ─────────────────────────────────────────

def load_obj(obj_path: Path):
    obj_dir = obj_path.parent
    verts, uvs, faces = [], [], []
    tex_img = None

    with open(obj_path, "r", errors="replace") as f:
        for raw in f:
            tok = raw.strip().split()
            if not tok: continue
            t = tok[0].lower()
            if t == "mtllib":
                tex_img = load_texture(obj_dir / " ".join(tok[1:]), obj_dir)
            elif t == "v":
                verts.append([float(tok[1]), float(tok[2]), float(tok[3])])
            elif t == "vt":
                uvs.append([float(tok[1]), float(tok[2])])
            elif t == "f":
                face = []
                for entry in tok[1:]:
                    p = entry.split("/")
                    vi = int(p[0]) - 1
                    ti = int(p[1]) - 1 if len(p) > 1 and p[1] else -1
                    face.append((vi, ti))
                for i in range(1, len(face) - 1):
                    faces.append([face[0], face[i], face[i+1]])

    pts   = np.array(verts, dtype=np.float64)
    uvs   = np.array(uvs,   dtype=np.float64) if uvs else np.zeros((0, 2))
    faces = np.array([[(vi, ti) for vi, ti in tri] for tri in faces], dtype=np.int32)

    print(f"  Vertices : {len(pts):,}   Faces : {len(faces):,}   UVs : {len(uvs):,}")
    if tex_img is not None:
        print(f"  Texture  : {tex_img.shape[1]}×{tex_img.shape[0]} px  ← full detail available")
    else:
        print("  ⚠ No texture found — output will be grey")

    return pts, uvs, faces, tex_img


def load_texture(mtl_path: Path, obj_dir: Path):
    if not mtl_path.exists():
        print(f"  ⚠ MTL not found: {mtl_path.name}")
        return None
    with open(mtl_path, "r", errors="replace") as f:
        for line in f:
            tok = line.strip().split()
            if tok and tok[0].lower() == "map_kd":
                tex_path = obj_dir / " ".join(tok[1:])
                if not tex_path.exists():
                    for c in obj_dir.glob("*"):
                        if c.name.lower() == tex_path.name.lower():
                            tex_path = c; break
                if tex_path.exists():
                    return np.array(Image.open(tex_path).convert("RGB"))
                else:
                    print(f"  ⚠ Texture file not found: {tex_path.name}")
    return None


# ── Main unwrap ───────────────────────────────────────────────────

def unwrap(pts, uvs, faces, tex, nlg, nlt,
           front_axis, up_axis, offset_deg, sample_density):

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    up_i    = axis_map[up_axis.lower()]
    front_i = axis_map[front_axis.lower()]
    side_i  = 3 - up_i - front_i

    up_v    = pts[:, up_i]
    front_v = pts[:, front_i]
    side_v  = pts[:, side_i]

    theta = np.arctan2(side_v, front_v)
    theta = (theta + np.deg2rad(offset_deg) + 2*np.pi) % (2*np.pi)
    z_min, z_max = up_v.min(), up_v.max()
    z_range = z_max - z_min

    if nlt is None:
        r_med  = np.nanmedian(np.sqrt(front_v**2 + side_v**2))
        aspect = (2 * np.pi * r_med) / z_range
        nlt    = max(64, int(nlg / aspect))
        print(f"  Auto NLT = {nlt}  (r_med={r_med:.1f}, z_range={z_range:.1f})")

    # Per-vertex cylindrical coords normalised [0,1]
    col_n = theta / (2 * np.pi)
    row_n = 1.0 - (up_v - z_min) / z_range

    vi_all = faces[:, :, 0]   # (F, 3)
    ti_all = faces[:, :, 1]

    fc = col_n[vi_all]         # (F, 3) cylindrical col
    fr = row_n[vi_all]         # (F, 3) cylindrical row
    fu = uvs[ti_all, 0]        # (F, 3) U
    fv = uvs[ti_all, 1]        # (F, 3) V

    # Skip faces that cross the seam (span > 25% of width)
    col_span = fc.max(axis=1) - fc.min(axis=1)
    valid    = col_span < 0.25
    print(f"  Valid faces : {valid.sum():,} / {len(faces):,}  (rest cross seam)")

    fc = fc[valid]; fr = fr[valid]
    fu = fu[valid]; fv = fv[valid]

    # Estimate pixel footprint per triangle → how many samples to take
    pw = np.maximum(np.abs(fc[:,0]-fc[:,1]), np.abs(fc[:,1]-fc[:,2])) * nlg
    ph = np.maximum(np.abs(fr[:,0]-fr[:,1]), np.abs(fr[:,1]-fr[:,2])) * nlt
    n_samp = np.clip((pw * ph * sample_density).astype(int), 6, 500)

    print(f"  Avg samples/triangle : {n_samp.mean():.0f}  "
          f"(total samples ≈ {n_samp.sum():,})")

    # Texture dimensions
    if tex is not None:
        TH, TW = tex.shape[:2]
    else:
        TH = TW = 1
        tex = np.full((1, 1, 3), 180, dtype=np.uint8)

    canvas_sum = np.zeros((nlt, nlg, 3), dtype=np.float32)
    canvas_cnt = np.zeros((nlt, nlg),    dtype=np.float32)

    print("  Rasterising triangles (sampling texture)...")
    t0 = time.time()
    np.random.seed(0)

    F = len(fc)
    REPORT = max(1, F // 5)

    for i in range(F):
        ns = n_samp[i]
        # Uniform random barycentric coords
        r1 = np.random.rand(ns); r2 = np.random.rand(ns)
        sq = np.sqrt(r1)
        a  = 1 - sq
        b  = sq * (1 - r2)
        c  = sq * r2

        # Interpolate cylindrical + UV
        sc = a*fc[i,0] + b*fc[i,1] + c*fc[i,2]
        sr = a*fr[i,0] + b*fr[i,1] + c*fr[i,2]
        su = a*fu[i,0] + b*fu[i,1] + c*fu[i,2]
        sv = a*fv[i,0] + b*fv[i,1] + c*fv[i,2]

        # Output pixel coords
        pc = (sc * nlg).astype(int) % nlg
        pr = np.clip((sr * nlt).astype(int), 0, nlt - 1)

        # Texture pixel coords (flip V axis)
        tpx = np.clip((su * (TW-1)).astype(int), 0, TW-1)
        tpy = np.clip(((1-sv) * (TH-1)).astype(int), 0, TH-1)

        rgb = tex[tpy, tpx].astype(np.float32)
        np.add.at(canvas_sum, (pr, pc), rgb)
        np.add.at(canvas_cnt, (pr, pc), 1)

        if i % REPORT == 0:
            print(f"    {i:>6}/{F}  ({i/F*100:.0f}%)  {time.time()-t0:.1f}s")

    print(f"  Done in {time.time()-t0:.1f}s")

    filled = canvas_cnt > 0
    cov    = filled.sum() / (nlt * nlg) * 100
    print(f"  Coverage : {cov:.1f}%")

    result = np.zeros((nlt, nlg, 3), dtype=np.uint8)
    result[filled] = (canvas_sum[filled] /
                      canvas_cnt[filled, None]).clip(0, 255).astype(np.uint8)

    # Inpaint any remaining gaps with nearest-neighbour
    if (~filled).any():
        print(f"  Inpainting {(~filled).sum():,} remaining gap pixels...")
        _, (nr, nc) = distance_transform_edt(~filled, return_indices=True)
        result = result[nr, nc]

    return Image.fromarray(result, "RGB"), nlt


# ── Main ─────────────────────────────────────────────────────────

def main():
    obj_path = Path(INPUT_OBJ)
    out_path = Path(OUTPUT_TIF)

    if not obj_path.exists():
        print(f"✗ OBJ not found: {obj_path.resolve()}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Input   : {obj_path.name}")
    print(f"  Output  : {out_path}")
    print(f"  Canvas  : {NLG} × {'auto' if NLT is None else NLT}")
    print(f"  Up={UP_AXIS.upper()}  Front={FRONT_AXIS.upper()}  "
          f"Offset={FRONT_OFFSET}°  Density={SAMPLE_DENSITY}")
    print(f"{'='*60}\n")

    pts, uvs, faces, tex = load_obj(obj_path)

    img, nlt_used = unwrap(pts, uvs, faces, tex,
                           nlg=NLG, nlt=NLT,
                           front_axis=FRONT_AXIS, up_axis=UP_AXIS,
                           offset_deg=FRONT_OFFSET,
                           sample_density=SAMPLE_DENSITY)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="TIFF", compression="tiff_lzw")
    sz = out_path.stat().st_size
    print(f"\n  ✓  {img.width}×{img.height} TIF saved  ({sz/1e6:.2f} MB)")
    print(f"     → {out_path}\n")


if __name__ == "__main__":
    main()