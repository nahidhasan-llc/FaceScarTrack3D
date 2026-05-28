"""
step1_detect_landmarks.py
=========================
Detect 5 face landmarks on the rendered front-view PNGs
from the ICP-aligned scans.

READS FROM:
    Dataset/frankfurt_aligned/pat1/
        pat1day0C_aligned_oriented_icp_front.png
        pat1day28A_aligned_oriented_icp_front.png
        pat1day28C2_aligned_oriented_icp_front.png

WRITES TO:
    Dataset/landmarks_raw/pat1/
        pat1day0C_aligned_oriented_icp_landmarks.json
        pat1day0C_aligned_oriented_icp_landmarks.png

USAGE:
    python step1_detect_landmarks.py
    python step1_detect_landmarks.py --patient pat1
    python step1_detect_landmarks.py --manual pat1day0C_aligned_oriented_icp
    python step1_detect_landmarks.py --redetect
"""

import sys, json, argparse, urllib.request, warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# MediaPipe landmark indices
LANDMARK_NAMES = {
    "sellion":   168,   # nose bridge
    "pronasale": 4,     # nose tip
    "menton":    152,   # chin
    "l_tragion": 234,   # left ear
    "r_tragion": 454,   # right ear
}

LM_COLOR = {
    "sellion":   (0,   220, 0),
    "pronasale": (220, 220, 0),
    "menton":    (220, 0,   220),
    "l_tragion": (0,   120, 255),
    "r_tragion": (255, 120, 0),
}

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_FILE = "face_landmarker.task"


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
#  DISCOVER front PNGs
# ─────────────────────────────────────────────────────────────────────────────
def discover_front_pngs(fa_root: Path, patient_filter=None) -> list:
    """
    Find all *_oriented_icp_front.png in frankfurt_aligned/patX/
    Returns list of (patient_id, stem, png_path)
    """
    scans = []
    for pat_dir in sorted(fa_root.iterdir()):
        if not pat_dir.is_dir():
            continue
        pid = pat_dir.name
        if patient_filter and pid.lower() != patient_filter.lower():
            continue
        for f in sorted(pat_dir.glob("*_oriented_icp_front.png")):
            scans.append((pid, f.stem.replace("_front", ""), f))
    return scans


# ─────────────────────────────────────────────────────────────────────────────
#  MEDIAPIPE MODEL
# ─────────────────────────────────────────────────────────────────────────────
def ensure_model(script_dir: Path) -> str:
    model_path = script_dir / MODEL_FILE
    if not model_path.exists():
        print(f"  Downloading MediaPipe model (~30 MB)...")
        urllib.request.urlretrieve(MODEL_URL, str(model_path))
    return str(model_path)


# ─────────────────────────────────────────────────────────────────────────────
#  MEDIAPIPE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def mediapipe_detect(img_rgb: np.ndarray, model_path: str) -> dict | None:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
    except ImportError:
        print("Run: pip install mediapipe"); sys.exit(1)

    H, W = img_rgb.shape[:2]
    opts = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.05,
        min_face_presence_confidence=0.05,
        min_tracking_confidence=0.05,
    )
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    with vision.FaceLandmarker.create_from_options(opts) as det:
        res = det.detect(mp_img)

    if not res.face_landmarks:
        return None

    lms = res.face_landmarks[0]
    return {name: [int(lms[idx].x * W), int(lms[idx].y * H)]
            for name, idx in LANDMARK_NAMES.items()}


def is_valid(lm: dict) -> bool:
    """Check landmarks describe a correctly oriented front-facing face."""
    sel = lm["sellion"];  men = lm["menton"]
    pro = lm["pronasale"]
    lt  = lm["l_tragion"]; rt  = lm["r_tragion"]
    return (sel[1] < men[1]               # sellion above menton
            and lt[0]  < rt[0]            # left ear left of right
            and sel[1] < pro[1] < men[1]  # nose tip between
            and lt[0]  < sel[0] < rt[0])  # sellion between ears


# ─────────────────────────────────────────────────────────────────────────────
#  MANUAL CLICK FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def manual_click(img_rgb: np.ndarray, scan_name: str) -> dict | None:
    order = list(LANDMARK_NAMES.keys())
    state = {"lm": {}, "i": 0}
    dots, lbls = [], []

    fig, ax = plt.subplots(figsize=(10, 12))
    fig.patch.set_facecolor("#0d1117")
    ax.imshow(img_rgb); ax.axis("off")

    instructions = ("Click landmarks IN ORDER:\n\n" +
                    "\n".join(f"  {i+1}. {n}" for i, n in enumerate(order)) +
                    "\n\nRight-click = undo")
    fig.text(0.01, 0.5, instructions, color="white", fontsize=9,
             va="center", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.8))
    plt.subplots_adjust(left=0.18)

    def set_title():
        i = state["i"]
        if i < len(order):
            ax.set_title(f"{scan_name}\nClick: {order[i].upper()}  ({i+1}/{len(order)})",
                         color="white", fontsize=11)
        else:
            ax.set_title("All done! Close window to continue.",
                         color="#aaffaa", fontsize=12)
        fig.canvas.draw()

    def on_click(event):
        if event.inaxes != ax: return
        if event.button == 1 and state["i"] < len(order):
            name = order[state["i"]]
            px, py = int(event.xdata), int(event.ydata)
            state["lm"][name] = [px, py]
            clr = [c/255 for c in LM_COLOR[name]]
            dots.append(ax.plot(px, py, 'o', color=clr, ms=11, mec='white', mew=1.5)[0])
            lbls.append(ax.text(px+8, py-8, name, color=clr, fontsize=8, fontweight='bold'))
            state["i"] += 1; set_title()
        elif event.button == 3 and state["i"] > 0:
            state["i"] -= 1
            state["lm"].pop(order[state["i"]], None)
            dots.pop().remove(); lbls.pop().remove(); set_title()

    fig.canvas.mpl_connect('button_press_event', on_click)
    set_title(); plt.show(block=True)
    return state["lm"] if len(state["lm"]) == len(order) else None


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
def save_landmark_image(img_rgb: np.ndarray, landmarks: dict, path: Path):
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for name, (px, py) in landmarks.items():
        r, g, b = LM_COLOR[name]
        cv2.circle(img, (px, py), 10, (b, g, r), -1)
        cv2.circle(img, (px, py), 11, (255, 255, 255), 1)
        cv2.putText(img, name, (px+12, py-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (b, g, r), 1)
    cv2.imwrite(str(path), img)


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS ONE SCAN
# ─────────────────────────────────────────────────────────────────────────────
def process_one(pid: str, stem: str, png_path: Path,
                model_path: str, out_dir: Path,
                force_manual=False, redetect=False) -> dict:

    json_path = out_dir / f"{stem}_landmarks.json"
    img_path  = out_dir / f"{stem}_landmarks.png"

    if json_path.exists() and not redetect and not force_manual:
        print(f"    skip (cached): {stem}")
        return {"scan": stem, "status": "cached"}

    print(f"    {stem}", end="  ", flush=True)

    img_bgr = cv2.imread(str(png_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W    = img_rgb.shape[:2]

    landmarks = None
    method    = "mediapipe"

    if not force_manual:
        lm = mediapipe_detect(img_rgb, model_path)
        if lm and is_valid(lm):
            landmarks = lm
            print(f"✓ mediapipe")
        else:
            print(f"⚠ mediapipe failed → opening manual click")

    if landmarks is None:
        landmarks = manual_click(img_rgb, stem)
        if landmarks is None:
            print(f"✗ skipped")
            return {"scan": stem, "status": "skipped"}
        method = "manual"

    for name, (px, py) in landmarks.items():
        print(f"        {name:12s}: ({px}, {py})")

    # Save
    data = {
        "scan": stem, "patient": pid,
        "image_w": W, "image_h": H,
        "method": method,
        "source_png": str(png_path),
        "landmarks": landmarks,
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    save_landmark_image(img_rgb, landmarks, img_path)

    return {"scan": stem, "status": "success", "method": method}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",     default=None)
    parser.add_argument("--patient",  default=None)
    parser.add_argument("--manual",   default=None,
                        help="Force manual click for this scan stem")
    parser.add_argument("--redetect", action="store_true")
    args = parser.parse_args()

    if args.root:
        root = Path(args.root).expanduser().resolve()
    else:
        root = find_nahidw_root(Path(__file__))
    print(f"Root: {root}")

    fa_root  = root / "Dataset" / "frankfurt_aligned"
    out_root = root / "Dataset" / "landmarks_raw"

    print(f"\n{'='*55}")
    print(f"  Step 1 — Landmark Detection")
    print(f"  Input  : {fa_root}")
    print(f"  Output : {out_root}")
    print(f"{'='*55}")

    scans = discover_front_pngs(fa_root, patient_filter=args.patient)
    if not scans:
        print("No *_oriented_icp_front.png found.")
        print("Run frankfurt_step1_render.py first.")
        return

    print(f"\nFound {len(scans)} scan(s)\n")
    model_path = ensure_model(Path(__file__).parent)

    results  = []
    prev_pat = None
    for pid, stem, png_path in scans:
        if pid != prev_pat:
            print(f"  Patient: {pid}")
            prev_pat = pid
        out_dir = out_root / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        force = bool(args.manual and args.manual.lower() == stem.lower())
        results.append(
            process_one(pid, stem, png_path, model_path, out_dir,
                        force_manual=force, redetect=args.redetect))

    ok     = sum(1 for r in results if r["status"] in ("success","cached"))
    failed = [r for r in results if r["status"] == "skipped"]
    print(f"\n{'='*55}")
    print(f"  Done — {ok}/{len(results)} processed")
    if failed:
        print(f"  Needs manual click:")
        for r in failed:
            print(f"    python step1_detect_landmarks.py --manual {r['scan']}")
    print(f"\n  Output : {out_root}")
    print(f"  Next   : python step2_align_images.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()