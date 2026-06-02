# 3D Scan → Unwrapped Cylindrical TIF

Converts aligned 3D head scans (OBJ or PLY) into unwrapped cylindrical TIF images. The head is projected onto a cylinder, then unrolled flat — like peeling a label off a bottle.

```
3D head scan  →  cylindrical projection  →  flat TIF image
  (.obj/.ply)                                 (colour or shaded)
```

---

## Files in this folder

```
obj_to_tif.py   — OBJ + texture → full-colour TIF
ply_to_tif.py   — PLY mesh      → shaded greyscale TIF
README.md       — this file
```

---

## Requirements

```bash
pip install numpy pillow scipy
```

---

## obj_to_tif.py

### What it does
Loads an OBJ file with its MTL material and texture image, rasterises every triangle by sampling the texture at each pixel, then projects the result cylindrically. Produces a high-quality full-colour TIF.

### What you need
- The `.obj` file
- The `.mtl` file (same folder)
- The texture image referenced in the MTL (e.g. `map_Kd texture.png`) in the same folder

### How to run
1. Open `obj_to_tif.py`
2. Change the two paths at the top:
```python
INPUT_OBJ  = r"D:\your\path\head.obj"
OUTPUT_TIF = r"D:\your\path\head_unwrapped.tif"
```
3. Run:
```bash
python obj_to_tif.py
```

### Config options
```python
NLG          = 2048   # output width in pixels — higher = more detail, slower
NLT          = None   # output height — None = auto-calculated from geometry
FRONT_AXIS   = 'z'    # which axis the face points along ('x', 'y', or 'z')
UP_AXIS      = 'y'    # which axis is vertical
FRONT_OFFSET = 180    # degrees — rotates face left/right in image
SAMPLE_DENSITY = 3    # texture sampling quality: 1=fast, 3=good, 6=max
```

### Output quality
Since the texture is sampled at full resolution (e.g. 4096×4096), the output is high quality with real skin color and detail. Runtime is typically 10–30 seconds depending on `SAMPLE_DENSITY` and mesh size.

### If orientation looks wrong
| Problem | Fix |
|---|---|
| Face off-centre | Change `FRONT_OFFSET` by ±45° steps |
| Head sideways | Swap `UP_AXIS` and `FRONT_AXIS` |

---

## ply_to_tif.py

### What it does
Loads a PLY mesh, auto-detects the head orientation, centres it on the cylinder axis, then projects cylindrically. Since PLY files usually have no texture, color is generated from face normals (diffuse shading) giving a realistic 3D-shaded look.

### What you need
- Just the `.ply` file — no texture needed

### How to run
1. Open `ply_to_tif.py`
2. Change the two paths at the top:
```python
INPUT_PLY  = r"D:\your\path\scan.ply"
OUTPUT_TIF = r"D:\your\path\scan_unwrapped.tif"
```
3. Run:
```bash
python ply_to_tif.py
```

### Config options
```python
NLG          = 2048   # output width in pixels
NLT          = None   # output height — None = auto

# Leave as None to auto-detect. Override only if result looks wrong:
UP_AXIS      = None   # 'x', 'y', or 'z'
FRONT_AXIS   = None   # 'x', 'y', or 'z'
FLIP_Y       = None   # True = flip upside down
FRONT_OFFSET = None   # degrees — shift face left/right

# Alignment to an OBJ (optional):
ALIGN_TO_OBJ = None   # path to .obj — aligns PLY to match OBJ coordinate frame

# Color when PLY has no RGB:
NO_COLOR_MODE = 'normals'       # 'normals' = shaded, 'flat' = plain color
SKIN_COLOR    = (210, 175, 155) # base skin tone RGB
```

### What the script auto-detects
Every run the script prints what it found:
```
── Auto-detected ───────────────────────────────────
  up_axis             : y
  front_axis          : z
  front_offset        : 184.0
  flip_y              : False
```

### If orientation looks wrong
The script prints the suggested fix at the end of every run:
```
If upside down    : FLIP_Y = True
If face off-centre: FRONT_OFFSET = 4
```
Just paste those lines into the config at the top and re-run.

### PLY with no color
If the PLY has no vertex RGB, the output will be a shaded skin-tone render — correct shape and proportions, but no real color. The dark areas are shadows from the face geometry (eye sockets, nose underside, etc.), not missing data.

To get real color from a PLY, the scan would need to include vertex RGB or a companion texture file.

### ALIGN_TO_OBJ
If you have both the OBJ and PLY of the same head and they are in different coordinate frames (different rotation/translation), set:
```python
ALIGN_TO_OBJ = r"D:\your\path\head.obj"
```
The script will auto-align the PLY to match the OBJ's orientation using PCA, so both produce the same layout.

---

## Understanding the output

```
┌──────────────────────────────────────────┐
│  back of head  │  face  │  back of head  │
│   left ear ←  nose  → right ear          │
│                                           │
│  top of image = top of head               │
│  bottom       = chin/neck                 │
└──────────────────────────────────────────┘
```

- The face is centred in the image
- Both ears appear — one on each side of the face
- The back of the head fills the left and right edges
- The seam (where the cylinder was cut) runs down the back of the head

---

## Comparison

| | `obj_to_tif.py` | `ply_to_tif.py` |
|---|---|---|
| Input | `.obj` + `.mtl` + texture | `.ply` only |
| Color | Real photo texture | Normal-based shading |
| Quality | High — full texture detail | Geometry only |
| Runtime | 10–30s | 2–5s |
| Axes | Manual config | Auto-detected |
| Works without texture | No | Yes (shaded) |
