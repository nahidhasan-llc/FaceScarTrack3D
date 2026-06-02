#!/usr/bin/env python3
"""
ply_to_tif.py — PLY mesh → unwrapped cylindrical TIF
══════════════════════════════════════════════════════
★  CHANGE THESE TWO LINES PER FILE  ★
"""

INPUT_PLY  = r"D:\NahidW\Dataset\test_data\aligned_face\PLY_Pasha.ply"
OUTPUT_TIF = r"D:\NahidW\Dataset\test_data\aligned_face\PLY_Pasha_unwrapped.tif"

# ── Resolution ───────────────────────────────────────────────────
NLG = 2048
NLT = None    # None = auto

# ── Alignment ────────────────────────────────────────────────────
# If your PLY is the same head as an OBJ but in a different coordinate
# frame, set this to the OBJ path. The PLY will be auto-aligned to
# match the OBJ orientation before unwrapping.
# Set to None to skip alignment.
ALIGN_TO_OBJ = None
# Example: ALIGN_TO_OBJ = r"D:\NahidW\Dataset\test_data\aligned_face\Pasha_guard_head.obj"

# ── Manual axis overrides ─────────────────────────────────────────
# Only needed if NOT using ALIGN_TO_OBJ. Leave as None to auto-detect.
UP_AXIS      = None
FRONT_AXIS   = None
FLIP_Y       = None
FRONT_OFFSET = None

# ── No-color fallback ─────────────────────────────────────────────
NO_COLOR_MODE = 'normals'   # 'normals' or 'flat'
SKIN_COLOR    = (210, 175, 155)

# ═════════════════════════════════════════════════════════════════
import sys, re, struct, time
import numpy as np
from pathlib import Path
from PIL import Image

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "-q"])
    from scipy.ndimage import distance_transform_edt


# ─────────────────────────────────────────────────────────────────
#  PLY READER
# ─────────────────────────────────────────────────────────────────

TYPE_MAP = {
    "float":  ("f4", 4), "float32": ("f4", 4),
    "double": ("f8", 8), "float64": ("f8", 8),
    "uchar":  ("u1", 1), "uint8":   ("u1", 1),
    "char":   ("i1", 1), "int8":    ("i1", 1),
    "short":  ("i2", 2), "ushort":  ("u2", 2),
    "int":    ("i4", 4), "uint":    ("u4", 4),
    "int32":  ("i4", 4), "uint32":  ("u4", 4),
}

def read_ply(path: Path):
    with open(path, "rb") as f:
        raw = f.read()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:header_end].decode("ascii", errors="replace")
    lines  = [l.strip() for l in header.split("\n") if l.strip()]
    fmt    = next(l for l in lines if l.startswith("format"))
    is_ascii = "ascii" in fmt
    endian   = ">" if "binary_big_endian" in fmt else "<"

    elements = {}; cur = None
    for line in lines:
        if line.startswith("element"):
            p = line.split(); cur = p[1]
            elements[cur] = {"count": int(p[2]), "props": []}
        elif line.startswith("property") and cur:
            elements[cur]["props"].append(line.split())

    pos = header_end; parsed = {}
    for elem_name, elem in elements.items():
        n = elem["count"]; props = elem["props"]
        list_p = [p for p in props if "list" in p]
        reg_p  = [p for p in props if "list" not in p]
        if list_p:
            cnt_type=list_p[0][2]; idx_type=list_p[0][3]
            cnt_size=TYPE_MAP[cnt_type][1]; idx_size=TYPE_MAP[idx_type][1]
            cnt_char=np.dtype(endian+TYPE_MAP[cnt_type][0]).char
            idx_char=np.dtype(endian+TYPE_MAP[idx_type][0]).char
            indices=[]
            for _ in range(n):
                cnt=struct.unpack_from(endian+cnt_char,raw,pos)[0]; pos+=cnt_size
                idx=struct.unpack_from(endian+str(cnt)+idx_char,raw,pos); pos+=cnt*idx_size
                indices.append(idx)
            parsed[elem_name]=indices
        elif is_ascii:
            text_lines=raw[pos:].decode("ascii",errors="replace").split("\n")
            rows=[]
            for line in text_lines:
                if len(rows)>=n: break
                vals=line.strip().split()
                if vals: rows.append([float(v) for v in vals])
            parsed[elem_name]={p[2]:np.array([r[i] for r in rows[:n]]) for i,p in enumerate(reg_p)}
            pos+=sum(len(l.encode())+1 for l in text_lines[:n])
        else:
            dt=np.dtype([(p[2],endian+TYPE_MAP[p[1]][0]) for p in reg_p])
            data=np.frombuffer(raw[pos:pos+n*dt.itemsize],dtype=dt); pos+=n*dt.itemsize
            parsed[elem_name]={name:data[name] for name in data.dtype.names}

    vd  = parsed["vertex"]
    pts = np.column_stack([vd["x"].astype(np.float64),
                           vd["y"].astype(np.float64),
                           vd["z"].astype(np.float64)])
    colors = None
    for rk,gk,bk in [("red","green","blue"),("r","g","b")]:
        if rk in vd and gk in vd and bk in vd:
            colors = np.column_stack([vd[rk],vd[gk],vd[bk]]).astype(np.uint8)
            break
    faces = None
    if "face" in parsed:
        tris=[]
        for idx in parsed["face"]:
            if   len(idx)==3: tris.append(idx)
            elif len(idx)==4:
                tris.append((idx[0],idx[1],idx[2])); tris.append((idx[0],idx[2],idx[3]))
        faces=np.array(tris,dtype=np.int32) if tris else None

    print(f"  Points : {len(pts):,}")
    print(f"  Faces  : {len(faces):,}" if faces is not None else "  Faces  : none")
    print(f"  RGB    : {'yes' if colors is not None else 'no — will shade from normals'}")
    return pts, colors, faces


# ─────────────────────────────────────────────────────────────────
#  OBJ READER (vertices only, for alignment)
# ─────────────────────────────────────────────────────────────────

def read_obj_verts(path: Path):
    verts = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            tok = line.strip().split()
            if tok and tok[0] == 'v':
                verts.append([float(tok[1]), float(tok[2]), float(tok[3])])
    return np.array(verts, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────
#  PCA ALIGNMENT  — align PLY coords to match OBJ coords
# ─────────────────────────────────────────────────────────────────

def align_to_reference(pts, ref_pts):
    """
    Rotate and translate pts so its coordinate frame matches ref_pts.
    Uses PCA to find principal axes of both clouds, then maps one to the other.
    """
    # Centre both
    pts_c   = pts     - pts.mean(axis=0)
    ref_c   = ref_pts - ref_pts.mean(axis=0)

    def pca_axes(p):
        cov = np.cov(p.T)
        vals, vecs = np.linalg.eigh(cov)
        return vecs[:, ::-1]   # largest eigenvalue first

    pts_axes = pca_axes(pts_c)
    ref_axes = pca_axes(ref_c)

    # Rotation: from PLY axes to OBJ axes
    R = ref_axes @ pts_axes.T

    # Fix sign ambiguity — make sure axes point in same direction
    # by checking dot product of transformed vs reference
    pts_rot = (R @ pts_c.T).T
    for i in range(3):
        if pts_rot[:, i].mean() * ref_c[:, i].mean() < 0:
            R[i, :] *= -1
    pts_rot = (R @ pts_c.T).T

    print(f"  Alignment rotation applied")
    print(f"  PLY range after align: "
          f"X={pts_rot[:,0].min():.1f}..{pts_rot[:,0].max():.1f}  "
          f"Y={pts_rot[:,1].min():.1f}..{pts_rot[:,1].max():.1f}  "
          f"Z={pts_rot[:,2].min():.1f}..{pts_rot[:,2].max():.1f}")
    return pts_rot


# ─────────────────────────────────────────────────────────────────
#  NORMAL-BASED SHADING
# ─────────────────────────────────────────────────────────────────

def shade_from_normals(pts, faces, skin_color):
    if faces is None or len(faces) == 0:
        return np.full((len(pts), 3), skin_color, dtype=np.uint8)
    v0=pts[faces[:,0]]; v1=pts[faces[:,1]]; v2=pts[faces[:,2]]
    fn=np.cross(v1-v0,v2-v0)
    fn/=np.maximum(np.linalg.norm(fn,axis=1,keepdims=True),1e-10)
    vn=np.zeros_like(pts)
    for i in range(3): np.add.at(vn,faces[:,i],fn)
    vn/=np.maximum(np.linalg.norm(vn,axis=1,keepdims=True),1e-10)
    light=np.array([0.1,0.3,0.9],dtype=np.float64); light/=np.linalg.norm(light)
    intensity=0.4+0.6*np.clip(vn@light,0,1)
    skin=np.array(skin_color,dtype=np.float32)
    return (intensity[:,None]*skin[None,:]).clip(0,255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────
#  AUTO AXIS DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_axes(pts, faces):
    names=['x','y','z']
    lo=pts.min(axis=0); hi=pts.max(axis=0)
    ranges=hi-lo; centre=(lo+hi)/2

    by_range=np.argsort(ranges)[::-1]
    up_i=int(by_range[0]); cands=[int(by_range[1]),int(by_range[2])]

    scores=[]
    for ci in cands:
        top=abs(pts[pts[:,ci]>np.percentile(pts[:,ci],95),ci].mean()-centre[ci])
        bot=abs(pts[pts[:,ci]<np.percentile(pts[:,ci], 5),ci].mean()-centre[ci])
        scores.append(max(top,bot)/ranges[ci])
    front_i=cands[int(np.argmax(scores))]
    side_i =cands[int(np.argmin(scores))]

    vn=_vertex_normals(pts,faces)
    up_v=pts[:,up_i]
    y_lo,y_hi=np.percentile(up_v,20),np.percentile(up_v,80)
    mid=(up_v>y_lo)&(up_v<y_hi)
    mean_angle=np.arctan2(vn[mid,side_i].mean(),vn[mid,front_i].mean())
    offset=float(np.degrees(np.pi-mean_angle))%360

    flip=False  # default; set FLIP_Y=True if upside down

    return (names[up_i],names[front_i],names[side_i],
            up_i,front_i,side_i,offset,flip,
            {'ranges':{n:round(float(r),2) for n,r in zip('xyz',ranges)},
             'up_axis':names[up_i],'front_axis':names[front_i],
             'side_axis':names[side_i],'front_offset':round(offset,1),'flip_y':flip})


def _vertex_normals(pts,faces):
    if faces is None or len(faces)==0: return np.zeros_like(pts)
    v0=pts[faces[:,0]]; v1=pts[faces[:,1]]; v2=pts[faces[:,2]]
    fn=np.cross(v1-v0,v2-v0); fn/=np.maximum(np.linalg.norm(fn,axis=1,keepdims=True),1e-10)
    vn=np.zeros_like(pts)
    for i in range(3): np.add.at(vn,faces[:,i],fn)
    vn/=np.maximum(np.linalg.norm(vn,axis=1,keepdims=True),1e-10)
    return vn


# ─────────────────────────────────────────────────────────────────
#  CYLINDRICAL UNWRAP + INPAINTING
# ─────────────────────────────────────────────────────────────────

def unwrap(pts, colors, nlg, nlt, up_i, front_i, side_i, offset_deg, flip_y):
    # Centre on cylinder axis — prevents angular distortion/stretching
    pts = pts.copy()
    pts[:, front_i] -= pts[:, front_i].mean()
    pts[:, side_i]  -= pts[:, side_i].mean()

    up_v=pts[:,up_i]; front_v=pts[:,front_i]; side_v=pts[:,side_i]
    theta=np.arctan2(side_v,front_v)
    theta=(theta+np.deg2rad(offset_deg)+2*np.pi)%(2*np.pi)
    z_min,z_max=up_v.min(),up_v.max(); z_range=z_max-z_min

    if nlt is None:
        r_all=np.sqrt(front_v**2+side_v**2)
        r_med=np.nanmedian(r_all)
        nlt=max(64,int(nlg/(2*np.pi*r_med/z_range)))
        print(f"  Auto NLT = {nlt}  (r_med={r_med:.1f}, z_range={z_range:.1f})")

    col_i=(theta/(2*np.pi)*nlg).astype(int)%nlg
    norm=(up_v-z_min)/z_range
    if flip_y: row_i=np.clip((norm*(nlt-1)).astype(int),0,nlt-1)
    else:      row_i=np.clip(((1-norm)*(nlt-1)).astype(int),0,nlt-1)

    canvas_sum=np.zeros((nlt,nlg,3),dtype=np.float64)
    canvas_cnt=np.zeros((nlt,nlg),dtype=np.int32)
    np.add.at(canvas_sum,(row_i,col_i),colors.astype(np.float64))
    np.add.at(canvas_cnt,(row_i,col_i),1)
    filled=canvas_cnt>0
    print(f"  Vertex coverage : {filled.sum()/(nlt*nlg)*100:.1f}%")
    result=np.zeros((nlt,nlg,3),dtype=np.uint8)
    result[filled]=(canvas_sum[filled]/canvas_cnt[filled,None]).astype(np.uint8)
    print("  Inpainting gaps...")
    t0=time.time()
    _,(nr,nc)=distance_transform_edt(~filled,return_indices=True)
    result=result[nr,nc]
    print(f"  Done in {time.time()-t0:.2f}s")
    return Image.fromarray(result,"RGB"), nlt


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    in_path  = Path(INPUT_PLY)
    out_path = Path(OUTPUT_TIF)

    if not in_path.exists():
        print(f"✗  PLY not found: {in_path.resolve()}"); sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Input  : {in_path.name}")
    print(f"  Output : {out_path}")
    print(f"  Canvas : {NLG} × {'auto' if NLT is None else NLT}")
    print(f"{'='*60}\n")

    pts, colors, faces = read_ply(in_path)

    # Align to OBJ if provided
    if ALIGN_TO_OBJ is not None:
        obj_path = Path(ALIGN_TO_OBJ)
        if not obj_path.exists():
            print(f"  ⚠ OBJ not found: {obj_path} — skipping alignment")
        else:
            print(f"  Aligning PLY to OBJ: {obj_path.name}")
            ref_pts = read_obj_verts(obj_path)
            pts = align_to_reference(pts, ref_pts)

    # Colors
    if colors is None:
        if NO_COLOR_MODE == 'normals' and faces is not None:
            print("  Generating shading from normals...")
            colors = shade_from_normals(pts, faces, SKIN_COLOR)
        else:
            colors = np.full((len(pts),3), SKIN_COLOR, dtype=np.uint8)

    # Axes
    _,_,_,auto_up_i,auto_front_i,auto_side_i,auto_offset,auto_flip,info = detect_axes(pts,faces)
    axis_map={'x':0,'y':1,'z':2}
    up_axis    = UP_AXIS      if UP_AXIS      is not None else info['up_axis']
    front_axis = FRONT_AXIS   if FRONT_AXIS   is not None else info['front_axis']
    flip_y     = FLIP_Y       if FLIP_Y       is not None else info['flip_y']
    offset     = FRONT_OFFSET if FRONT_OFFSET is not None else info['front_offset']
    up_i    = axis_map[up_axis.lower()]
    front_i = axis_map[front_axis.lower()]
    side_i  = 3-up_i-front_i

    print(f"\n  ── Config ──────────────────────────────────────────")
    for k,v in info.items():
        ov=((k=='up_axis' and UP_AXIS is not None)or(k=='front_axis' and FRONT_AXIS is not None)
            or(k=='flip_y' and FLIP_Y is not None)or(k=='front_offset' and FRONT_OFFSET is not None))
        print(f"    {k:20s}: {v}{' ← override' if ov else ''}")
    print(f"\n  ── Using: up={up_axis}  front={front_axis}  flip={flip_y}  offset={offset}°\n")

    img,_ = unwrap(pts,colors,NLG,NLT,up_i,front_i,side_i,offset,flip_y)

    out_path.parent.mkdir(parents=True,exist_ok=True)
    img.save(str(out_path),format="TIFF",compression="tiff_lzw")
    sz=out_path.stat().st_size
    print(f"\n  ✓  {img.width}×{img.height} TIF  ({sz/1e6:.2f} MB)")
    print(f"     → {out_path}")
    print(f"\n  If upside down : FLIP_Y = {not flip_y}")
    print(f"  If face off-centre: FRONT_OFFSET = {(offset+180)%360:.0f}\n")


if __name__ == "__main__":
    main()