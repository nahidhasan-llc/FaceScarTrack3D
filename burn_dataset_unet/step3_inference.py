"""
STEP 3 — Inference & Edge Overlay
===================================
Loads the trained UNet, runs inference on a burned face image,
and draws precise burn-area edges on the original image.

Usage:
    # Single image
    python step3_inference.py --image path/to/face.jpg

    # Folder of images
    python step3_inference.py --folder path/to/images/ --output results/

    # Webcam / video
    python step3_inference.py --video path/to/video.mp4

    # Interactive: draw bounding box to focus inference
    python step3_inference.py --image face.jpg --interactive
"""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = "checkpoints/best_model.pth"
IMG_SIZE           = 512
THRESHOLD          = 0.45     # lower = more sensitive, higher = more precise
EDGE_COLOR         = (0, 255, 80)   # bright green edges
FILL_ALPHA         = 0.25           # burn area fill transparency
FILL_COLOR         = (0, 60, 255)   # blue-red fill for burn area


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get("cfg", {"encoder": "resnet34"})

    model = smp.Unet(
        encoder_name=cfg.get("encoder", "resnet34"),
        encoder_weights=None,        # we load weights from checkpoint
        in_channels=3,
        classes=1,
        activation=None,
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    best_dice = ckpt.get("best_dice", "N/A")
    print(f"✅ Model loaded | Encoder: {cfg.get('encoder')} | Best Dice: {best_dice:.4f}")
    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────
def get_transform(img_size=IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ── Core inference ────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_mask(model, image_rgb, device, transform, threshold=THRESHOLD):
    """
    Args:
        image_rgb: np.ndarray (H, W, 3) in RGB
    Returns:
        binary_mask: np.ndarray (H, W) uint8, values 0 or 255
        prob_map:    np.ndarray (H, W) float32 0–1
    """
    h, w = image_rgb.shape[:2]
    tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    logits   = model(tensor)
    prob_map = torch.sigmoid(logits)[0, 0].cpu().numpy()
    prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)

    binary_mask = (prob_map > threshold).astype(np.uint8) * 255
    return binary_mask, prob_map


def clean_mask(binary_mask, kernel_size=9):
    """Morphological cleanup: remove speckle, close small gaps."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask   = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask,        cv2.MORPH_OPEN,  kernel)
    return mask


# ── Visualization ─────────────────────────────────────────────────────────────
def draw_burn_edges(
    image_bgr,
    binary_mask,
    prob_map=None,
    edge_color=EDGE_COLOR,
    fill_alpha=FILL_ALPHA,
    fill_color=FILL_COLOR,
    edge_thickness=2,
    show_probability=False,
):
    """
    Draws burn area with:
      - Filled semi-transparent overlay on burn region
      - Precise contour edges
      - Optional probability heatmap
    Returns annotated BGR image.
    """
    result = image_bgr.copy()
    h, w   = result.shape[:2]

    # ── Semi-transparent fill ──────────────────────────────────────────────
    if fill_alpha > 0:
        overlay           = result.copy()
        burn_pixels       = binary_mask > 127
        overlay[burn_pixels] = fill_color
        result = cv2.addWeighted(overlay, fill_alpha, result, 1 - fill_alpha, 0)

    # ── Contour edges ──────────────────────────────────────────────────────
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, edge_color, edge_thickness)

    # ── Probability heatmap (optional) ────────────────────────────────────
    if show_probability and prob_map is not None:
        heatmap = cv2.applyColorMap(
            (prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap = cv2.resize(heatmap, (w, h))
        result  = cv2.addWeighted(result, 0.7, heatmap, 0.3, 0)

    # ── Stats overlay ──────────────────────────────────────────────────────
    burn_pct = (binary_mask > 127).sum() / (h * w) * 100
    n_regions = len(contours)

    cv2.putText(result, f"Burn area: {burn_pct:.1f}%",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(result, f"Regions: {n_regions}",
                (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return result, contours


# ── Single image pipeline ─────────────────────────────────────────────────────
def process_image(model, image_path, device, transform, output_path=None, show=False, interactive=False):
    orig_bgr  = cv2.imread(str(image_path))
    if orig_bgr is None:
        print(f"ERROR: Cannot load {image_path}")
        return

    image_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # ── Interactive ROI selection ──────────────────────────────────────────
    if interactive:
        print("Draw a box around the face / burn area, then press ENTER or SPACE")
        roi = cv2.selectROI("Select burn region", orig_bgr, fromCenter=False)
        cv2.destroyWindow("Select burn region")
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            # Run inference on cropped region, paste result back
            crop_rgb  = image_rgb[y:y+rh, x:x+rw]
            mask_crop, prob_crop = predict_mask(model, crop_rgb, device, transform)
            mask_crop = clean_mask(mask_crop)

            binary_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            binary_mask[y:y+rh, x:x+rw] = cv2.resize(mask_crop, (rw, rh), interpolation=cv2.INTER_NEAREST)
            prob_map = np.zeros(image_rgb.shape[:2], dtype=np.float32)
            prob_map[y:y+rh, x:x+rw] = cv2.resize(prob_crop, (rw, rh))
        else:
            binary_mask, prob_map = predict_mask(model, image_rgb, device, transform)
            binary_mask = clean_mask(binary_mask)
    else:
        binary_mask, prob_map = predict_mask(model, image_rgb, device, transform)
        binary_mask = clean_mask(binary_mask)

    # Draw
    result, contours = draw_burn_edges(orig_bgr, binary_mask, prob_map)

    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
        # Also save mask
        mask_path = str(output_path).replace(".jpg", "_mask.png").replace(".png", "_mask.png")
        cv2.imwrite(mask_path, binary_mask)
        print(f"Saved: {output_path}")

    if show:
        cv2.imshow("Burn Edge Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result, binary_mask


# ── Folder batch processing ───────────────────────────────────────────────────
def process_folder(model, folder_path, output_dir, device, transform):
    folder     = Path(folder_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
    print(f"Processing {len(images)} images from {folder_path}...")

    for img_path in images:
        out_path = output_dir / img_path.name
        process_image(model, img_path, device, transform, output_path=out_path)

    print(f"\n✅ Done. Results saved to: {output_dir}")


# ── Video processing ──────────────────────────────────────────────────────────
def process_video(model, video_path, output_path, device, transform):
    cap = cv2.VideoCapture(video_path if video_path != "webcam" else 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print("Processing video... Press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        binary_mask, prob_map = predict_mask(model, image_rgb, device, transform)
        binary_mask = clean_mask(binary_mask, kernel_size=5)  # smaller kernel for speed
        result, _   = draw_burn_edges(frame, binary_mask)

        if writer:
            writer.write(result)

        cv2.imshow("Burn Detection", result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Burn Edge Detection")
    parser.add_argument("--checkpoint",   default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image",        type=str, help="Single image path")
    parser.add_argument("--folder",       type=str, help="Folder of images")
    parser.add_argument("--video",        type=str, help="Video path or 'webcam'")
    parser.add_argument("--output",       type=str, default="results/")
    parser.add_argument("--threshold",    type=float, default=THRESHOLD)
    parser.add_argument("--interactive",  action="store_true")
    parser.add_argument("--show",         action="store_true")
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = load_model(args.checkpoint, device)
    transform = get_transform()

    if args.image:
        out = Path(args.output) / Path(args.image).name
        process_image(model, args.image, device, transform,
                      output_path=out, show=args.show, interactive=args.interactive)

    elif args.folder:
        process_folder(model, args.folder, args.output, device, transform)

    elif args.video:
        out = str(Path(args.output) / "video_result.mp4")
        process_video(model, args.video, out, device, transform)

    else:
        print("Specify --image, --folder, or --video")
        parser.print_help()
