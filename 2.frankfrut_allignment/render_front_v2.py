#!/usr/bin/env python3
"""
step1_render_front.py
────────────────────────────────────────────────────────────────
Automatically detect the frontal face direction from a PLY point
cloud using skin color, then render and save the frontal view.

Works automatically for every scan — no manual yaw adjustment needed.

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


def detect_front_yaw(pts, cols):
    """
    Automatically find the yaw angle for the frontal camera view.

    Method: skin-colored points (R > G > B, R > 0.3) are concentrated
    on the face. Their XY centroid points toward the face direction.
    Camera yaw = opposite of face direction.

    Returns yaw in degrees.
    """
    r, g, b   = cols[:,0], cols[:,1], cols[:,2]
    skin_mask = (r > 0.3) & (r > g) & (g > b) & (r - b > 0.1)
    n_skin    = skin_mask.sum()

    if n_skin < 100:
        print(f"  WARNING: Only {n_skin} skin points found — using default yaw=0")
        return 0.0

    skin_pts   = pts[skin_mask]
    mean_xy    = skin_pts[:, :2].mean(axis=0)
    face_dir   = mean_xy / np.linalg.norm(mean_xy)
    face_angle = np.degrees(np.arctan2(face_dir[1], face_dir[0]))

    # Camera points from opposite side of face
    cam_yaw = face_angle + 180
    if cam_yaw >  180: cam_yaw -= 360
    if cam_yaw < -180: cam_yaw += 360

    print(f"  Skin points      : {n_skin:,}")
    print(f"  Face direction   : {face_angle:.1f}°")
    print(f"  Auto yaw         : {cam_yaw:.1f}°")

    return cam_yaw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ply")
    parser.add_argument("output_image", nargs="?", default="front_view.png")
    parser.add_argument("--yaw", type=float, default=None,
                        help="Override auto-detected yaw (degrees). "
                             "Only use if auto-detection fails.")
    args = parser.parse_args()

    print(f"\n{'─'*50}")
    print(f"  Step 1 — Render Frontal View from PLY")
    print(f"{'─'*50}\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"  Loading: {args.input_ply}")
    pcd  = o3d.io.read_point_cloud(args.input_ply)
    pts  = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    print(f"  Points  : {len(pts):,}")

    centroid = pts.mean(axis=0)
    print(f"  Centroid: {centroid.round(2)}")

    # ── Auto-detect front direction ───────────────────────────────────────────
    print(f"\n  Detecting front direction...")
    if args.yaw is not None:
        yaw = args.yaw
        print(f"  Using manual yaw: {yaw}°")
    else:
        yaw = detect_front_yaw(pts, cols)

    # ── Build camera front vector ─────────────────────────────────────────────
    yaw_rad = np.radians(-yaw)
    front   = [-np.cos(yaw_rad), -np.sin(yaw_rad), 0.0]
    up      = [0, 0, 1]
    zoom    = 0.55
    W, H    = 800, 1000

    print(f"\n  Rendering {W}×{H} (yaw={yaw:.1f}°)...")

    # ── Render ────────────────────────────────────────────────────────────────
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

    img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # ── Remove black background → white ──────────────────────────────────────
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    img_bgr[mask == 0] = [255, 255, 255]

    # ── Save ─────────────────────────────────────────────────────────────────
    cv2.imwrite(args.output_image, img_bgr)
    print(f"\n  ✓ Saved: {args.output_image}")
    print(f"\n  Next step:")
    print(f"    python step2_align.py {args.output_image} {args.input_ply} output_aligned.ply")
    print(f"\n{'─'*50}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)




        # python step1_v2.py pat1day0C.ply front_view.png