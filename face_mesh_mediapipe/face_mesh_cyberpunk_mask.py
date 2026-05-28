import cv2
import numpy as np
import mediapipe as mp

# 1. Initialize FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False, # Standard 468 points is ideal for full-face tessellation mapping
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Grab the official connection blueprint
# FACEMESH_TESSELATION holds the pairs of connections that form all 1,326 triangles
tessellation_connections = mp_face_mesh.FACEMESH_TESSELATION

cap = cv2.VideoCapture(0)
print("Webcam starting... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # Extract all 468 landmarks as concrete (x, y) pixel coordinates first
            landmarked_pixels = []
            for lm in face_landmarks.landmark:
                x = int(lm.x * width)
                y = int(lm.y * height)
                landmarked_pixels.append((x, y))

            # --- RENDER THE FULL TRIANGLE MESH ---
            # MediaPipe's connection sets are structured as unique FrozeSets containing 2 index points.
            # We stitch matching connections together to dynamically rebuild the face facets.
            triangle_dict = {}
            for connection in tessellation_connections:
                p1, p2 = connection
                
                # We log shared point structures to reconstruct complete triangles
                triangle_dict.setdefault(p1, []).append(p2)

            # Draw the triangles
            for p1, neighbors in triangle_dict.items():
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        p2 = neighbors[i]
                        p3 = neighbors[j]
                        
                        # Double check if p2 and p3 form a closed triangle line
                        if (p2, p3) in tessellation_connections or (p3, p2) in tessellation_connections:
                            # Grab pixel coordinates for our 3 vertices
                            pt1 = landmarked_pixels[p1]
                            pt2 = landmarked_pixels[p2]
                            pt3 = landmarked_pixels[p3]
                            
                            triangle_pts = np.array([pt1, pt2, pt3], dtype=np.int32)
                            
                            # --- CALCULATE SHIFTING COLOR ---
                            # This uses the triangle's location on your face to vary the colors automatically.
                            # You can replace this logic with static colors or transparent blending!
                            blue_channel = int((pt1[0] / width) * 255)
                            green_channel = int((pt1[1] / height) * 255)
                            red_channel = 255 - blue_channel
                            
                            color = (blue_channel, green_channel, red_channel)
                            
                            # Fill the solid face mesh facets
                            cv2.fillPoly(frame, [triangle_pts], color)
                            # cv2.addWeighted(frame, [triangle_pts], color)

                            # (Optional) Un-comment the line below to draw black outlines on the triangles:
                            cv2.polylines(frame, [triangle_pts], True, (0, 0, 0), 1)

    cv2.imshow("Full Face Mesh Tessellation", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()