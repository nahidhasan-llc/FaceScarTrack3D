#!/usr/bin/env python3
"""
step1_render_front.py
────────────────────────────────────────────────────────────────
Render the frontal face view from a PLY point cloud.

Camera: from -X direction, Z-up, with yaw adjustment to center face.

Usage
-----
    python step1_render_front.py <input_ply> [output_image]

Example
-------
    python step1_render_front.py pat1day0C.ply front_view.png

Install
-------
    python -m pip install numpy open3d opencv-python
"""

import sys
import os
import argparse
import numpy as np
import cv2
import open3d as o3d


def render_view(pcd, centroid, front, up, zoom, W, H):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=W, height=H)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size       = 2.0
    opt.background_color = np.array([0.0, 0.0, 0.0])
    ctr = vis.get_view_control()
    ctr.set_lookat(centroid.tolist())
    ctr.set_up(up)
    ctr.set_front(front)
    ctr.set_zoom(zoom)
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    return (img * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ply")
    parser.add_argument("output_image", nargs="?", default="front_view_normal.png")
    parser.add_argument("--yaw", type=float, default=15.0,
                        help="Yaw angle in degrees to rotate camera around Z axis (default=15)")
    args = parser.parse_args()

    base = args.output_image.replace(".png","").replace(".jpg","")

    print(f"\n{'─'*50}")
    print(f"  Step 1 — Render Frontal View from PLY")
    print(f"{'─'*50}\n")

    print(f"  Loading: {args.input_ply}")
    pcd = o3d.io.read_point_cloud(args.input_ply)
    pts = np.asarray(pcd.points)
    print(f"  Points : {len(pts):,}")
    centroid = pts.mean(axis=0)
    print(f"  Centroid: {centroid.round(2)}")

    W, H  = 800, 1000
    zoom  = 0.55

    # Base front direction: -X
    # Rotate around Z axis by yaw degrees to correct the tilt
    yaw_rad = np.radians(-args.yaw)
    # Rotate [-1, 0, 0] around Z by yaw
    front = np.array([
        -np.cos(yaw_rad),
        -np.sin(yaw_rad),
        0.0
    ])
    up = [0, 0, 1]

    print(f"\n  Yaw correction : {args.yaw}°")
    print(f"  Camera front   : {front.round(4)}")
    print(f"  Rendering {W}×{H}...")

    img_rgb  = render_view(pcd, centroid, front.tolist(), up, zoom, W, H)
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Save raw
    path_raw = f"{base}.png"
    cv2.imwrite(path_raw, img_bgr)
    print(f"  Saved: {path_raw}")

    # Save enhanced
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = np.zeros_like(img_bgr)
    for i in range(3):
        enhanced[:,:,i] = clahe.apply(img_bgr[:,:,i])
    kernel    = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    path_enh  = f"{base}_enhanced.png"
    cv2.imwrite(path_enh, sharpened)
    print(f"  Saved: {path_enh}")

    # Also save a few yaw variations around the chosen angle for fine-tuning
    print(f"\n  Saving ±5° variations for fine-tuning:")
    for delta in [-10, -5, 5, 10]:
        yr   = np.radians(args.yaw + delta)
        fr   = [-np.cos(yr), -np.sin(yr), 0.0]
        img  = cv2.cvtColor(render_view(pcd, centroid, fr, up, zoom, W, H), cv2.COLOR_RGB2BGR)
        path = f"{base}_yaw{int(args.yaw+delta):+d}.png"
        cv2.imwrite(path, img)
        print(f"    {path}  (yaw={args.yaw+delta:.0f}°)")

    print(f"\n{'─'*50}")
    print(f"  Done. Check {path_raw} — if still tilted,")
    print(f"  run again with --yaw <angle> to adjust.")
    print(f"  Then: python step2_align.py {path_raw} <input.ply> <output.ply>")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

        


# # Option A — re-run step1 to get the inpainted version automatically
# python step1_render_front.py pat1day0C.ply front_view.png
