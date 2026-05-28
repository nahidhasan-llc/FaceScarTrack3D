"""
step2_align_images.py
=====================
Reads landmark JSONs from step1 and aligns every front-view PNG
so all 5 landmarks land at fixed standard positions.

READS FROM:
    Dataset/landmarks_raw/pat1/
        pat1day0C_aligned_oriented_icp_landmarks.json
    Dataset/frankfurt_aligned/pat1/
        pat1day0C_aligned_oriented_icp_front.png

WRITES TO:
    Dataset/landmark_aligned/pat1/
        pat1day0C_aligned_oriented_icp_aligned.png
        pat1day0C_aligned_oriented_icp_aligned_preview.png

TEMPLATE (standard landmark positions on 600x700 canvas):
    sellion   → (300, 180)
    pronasale → (300, 270)
    menton    → (300, 440)
    l_tragion → (145, 235)
    r_tragion → (455, 235)

USAGE:
    python step2_align_images.py
    python step2_align_images.py --patient pat1
    python step2_align_images.py --show
"""

import sys, json, argparse, warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OUT_W = 600
OUT_H = 700

TEMPLATE = {
    "sellion":   np.array([300, 180], dtype=np.float32),
    "pronasale": np.array([300, 270], dtype=np.float32),
    "menton":    np.array([300, 440], dtype=np.float32),
    "l_tragion": np.array([145, 235], dtype=np.float32),
    "r_tragion": np.array([455, 235], dtype=np.float32),
}

LM_COLOR = {
    "sellion":   (0,   220, 0),
    "pronasale": (220, 220, 0),
    "menton":    (220, 0,   220),
    "l_tragion": (0,   120, 255),
    "r_tragion": (255, 120, 0),
}

LANDMARK_ORDER = ["sellion", "pronasale", "menton", "l_tragion", "r_tragion"]


# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-DETECT ROOT
# ─────────────────────────────────────────────────────────────────────────────
def find_nahidw_root(script_path: Path) -> Path:
    candidate = script_path.resolve().parent
    while True:
        if (candidate / "Dataset").is_dir():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError("Cannot find 'Dataset'. Use --root D:/NahidW")
        candidate = parent


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD LANDMARK JSONs
# ─────────────────────────────────────────────────────────────────────────────
def load_landmarks(lm_root: Path, patient_filter=None) -> list:
    """Returns list of landmark dicts sorted by patient/scan."""
    scans = []
    for f in sorted(lm_root.glob("**/*_landmarks.json")):
        with open(f) as fp:
            data = json.load(fp)
        pid = data.get("patient", f.parent.name)
        if patient_filter and pid.lower() != patient_filter.lower():
            continue
        data["_json_path"] = str(f)
        scans.append(data)
    return sorted(scans, key=lambda d: (d.get("patient",""), d.get("scan","")))


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSFORM: scale + translate only (no rotation — face already front-facing)
# ─────────────────────────────────────────────────────────────────────────────
def compute_transform(landmarks: dict) -> np.ndarray:
    """
    Scale uniformly using sellion-to-menton span as reference,
    then translate centroid to template centroid.
    No rotation — the ICP pre-orient step already made faces front-facing.
    """
    src = np.array([landmarks[n] for n in LANDMARK_ORDER], dtype=np.float64)
    dst = np.array([TEMPLATE[n]  for n in LANDMARK_ORDER], dtype=np.float64)

    # Scale from vertical span
    src_span = np.linalg.norm(
        np.array(landmarks["menton"]) - np.array(landmarks["sellion"]))
    dst_span = np.linalg.norm(TEMPLATE["menton"] - TEMPLATE["sellion"])

    if src_span < 1.0:
        raise ValueError("sellion and menton too close — check landmarks")

    scale = dst_span / src_span

    # Translation: align scaled centroid to template centroid
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    tx = dst_c[0] - scale * src_c[0]
    ty = dst_c[1] - scale * src_c[1]

    M = np.array([[scale, 0.0,   tx],
                  [0.0,   scale, ty]], dtype=np.float32)

    # Print residuals
    N      = len(src)
    src_h  = np.hstack([src, np.ones((N, 1))])
    warped = (M @ src_h.T).T
    errors = np.linalg.norm(warped - dst, axis=1)
    for name, err in zip(LANDMARK_ORDER, errors):
        print(f"        {name:12s}: residual {err:.1f} px")

    return M


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS ONE SCAN
# ─────────────────────────────────────────────────────────────────────────────
def process_one(data: dict, fa_root: Path, out_dir: Path, show=False) -> dict:
    stem    = data["scan"]
    pid     = data["patient"]
    lm      = {k: v for k, v in data["landmarks"].items()}

    # Source PNG: frankfurt_aligned/patX/<stem>_front.png
    png_path = fa_root / pid / f"{stem}_front.png"
    if not png_path.exists():
        print(f"    ✗ PNG not found: {png_path}")
        return {"scan": stem, "status": "png_not_found"}

    out_png     = out_dir / f"{stem}_aligned.png"
    out_preview = out_dir / f"{stem}_aligned_preview.png"

    print(f"\n    {stem}")

    img_bgr = cv2.imread(str(png_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        M = compute_transform(lm)
    except ValueError as e:
        print(f"    ✗ {e}")
        return {"scan": stem, "status": "failed"}

    aligned = cv2.warpAffine(img_rgb, M, (OUT_W, OUT_H),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))

    # Compute final landmark positions on aligned image
    errors = []
    lm_final = {}
    for name in LANDMARK_ORDER:
        px, py = lm[name]
        pt     = M @ np.array([px, py, 1.0])
        gx, gy = int(round(pt[0])), int(round(pt[1]))
        tx, ty = int(TEMPLATE[name][0]), int(TEMPLATE[name][1])
        err    = ((gx-tx)**2 + (gy-ty)**2) ** 0.5
        errors.append(err)
        lm_final[name] = [gx, gy]

    max_err = max(errors)
    print(f"    Max residual: {max_err:.1f} px  "
          f"{'✓' if max_err < 15 else '⚠ check output'}")

    # Save clean aligned PNG
    cv2.imwrite(str(out_png), cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))

    # Save preview with landmarks drawn
    preview = aligned.copy()
    for name, (gx, gy) in lm_final.items():
        r, g, b = LM_COLOR[name]
        cv2.circle(preview, (gx, gy), 8, (r, g, b), -1)
        cv2.circle(preview, (gx, gy), 9, (255, 255, 255), 1)
        cv2.putText(preview, name, (gx+10, gy-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (r, g, b), 1)
    cv2.imwrite(str(out_preview), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

    print(f"    Saved → {out_png.name}")

    if show:
        fig, axes = plt.subplots(1, 2, figsize=(12, 7))
        fig.patch.set_facecolor("#0d1117")
        axes[0].imshow(img_rgb); axes[0].set_title("Input", color="white"); axes[0].axis("off")
        axes[1].imshow(preview); axes[1].set_title(f"Aligned  (max err={max_err:.1f}px)",
                                                     color="white"); axes[1].axis("off")
        plt.suptitle(stem, color="white"); plt.tight_layout(); plt.show(block=True)

    return {"scan": stem, "status": "success", "max_error_px": round(max_err, 2)}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",    default=None)
    parser.add_argument("--patient", default=None)
    parser.add_argument("--show",    action="store_true")
    args = parser.parse_args()

    if args.root:
        root = Path(args.root).expanduser().resolve()
    else:
        root = find_nahidw_root(Path(__file__))
    print(f"Root: {root}")

    lm_root  = root / "Dataset" / "landmarks_raw"
    fa_root  = root / "Dataset" / "frankfurt_aligned"
    out_root = root / "Dataset" / "landmark_aligned"

    if not lm_root.exists():
        print("landmarks_raw/ not found. Run step1_detect_landmarks.py first.")
        return

    print(f"\n{'='*55}")
    print(f"  Step 2 — Image Alignment")
    print(f"  Landmarks : {lm_root}")
    print(f"  Images    : {fa_root}")
    print(f"  Output    : {out_root}")
    print(f"{'='*55}")

    scans = load_landmarks(lm_root, patient_filter=args.patient)
    if not scans:
        print("No landmark JSONs found."); return

    print(f"\nFound {len(scans)} scan(s)\n")

    results  = []
    prev_pat = None
    for data in scans:
        pid = data.get("patient", "?")
        if pid != prev_pat:
            print(f"  Patient: {pid}")
            prev_pat = pid
        out_dir = out_root / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        results.append(process_one(data, fa_root, out_dir, show=args.show))

    with open(out_root / "alignment_report.json", "w") as f:
        json.dump(results, f, indent=2)

    ok     = sum(1 for r in results if r["status"] == "success")
    failed = [r for r in results if r["status"] != "success"]

    print(f"\n{'='*55}")
    print(f"  Done — {ok}/{len(results)} aligned")
    if failed:
        for r in failed: print(f"    ✗ {r['scan']} — {r['status']}")
    print(f"\n  Output : {out_root}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()