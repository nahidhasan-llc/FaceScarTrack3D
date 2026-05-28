import os
import cv2
import numpy as np
import open3d as o3d
import mediapipe as mp

# -------------------------------------------------
# Paths
# -------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

ply_path = os.path.join(script_dir, "pat1day0C.ply")
render_path = os.path.join(script_dir, "render.png")
output_path = os.path.join(script_dir, "output_landmarks.png")

# -------------------------------------------------
# 1. Load PLY
# -------------------------------------------------
pcd = o3d.io.read_point_cloud(ply_path)

if len(pcd.points) == 0:
    print("Error: Empty PLY.")
    exit()

print(f"Loaded PLY with {len(pcd.points)} points")

# -------------------------------------------------
# 2. Render PLY to image
# -------------------------------------------------
vis = o3d.visualization.Visualizer()
vis.create_window(
    visible=False,
    width=1000,
    height=1000
)

vis.add_geometry(pcd)

# Improve appearance
render_option = vis.get_render_option()
render_option.point_size = 2.0
render_option.background_color = np.asarray([0, 0, 0])

# Set camera view
ctr = vis.get_view_control()
ctr.set_front([0, 0, -1])
ctr.set_up([0, -1, 0])
ctr.set_zoom(0.7)

vis.poll_events()
vis.update_renderer()

vis.capture_screen_image(render_path)

vis.destroy_window()

print(f"Rendered image saved: {render_path}")

# -------------------------------------------------
# 3. Load rendered image
# -------------------------------------------------
image = cv2.imread(render_path)

if image is None:
    print("Could not load rendered image.")
    exit()

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -------------------------------------------------
# 4. MediaPipe FaceMesh
# -------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

results = face_mesh.process(rgb_image)

# -------------------------------------------------
# 5. Draw landmarks
# -------------------------------------------------
if results.multi_face_landmarks:

    height, width, _ = image.shape

    print("Face detected!")

    for face_landmarks in results.multi_face_landmarks:

        for landmark in face_landmarks.landmark:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                image,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

    cv2.imwrite(output_path, image)

    print(f"Saved result: {output_path}")

else:
    print("No face detected.")