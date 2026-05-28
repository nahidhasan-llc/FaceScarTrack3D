import cv2
import numpy as np
import mediapipe as mp

# 1. Initialize FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

tessellation_connections = mp_face_mesh.FACEMESH_TESSELATION

# Pre-build connection dictionary
triangle_dict = {}
for connection in tessellation_connections:
    p1, p2 = connection
    triangle_dict.setdefault(p1, []).append(p2)

# --- DEFINING JOKER LANDMARK ZONES ---
# MediaPipe index ranges that correspond to specific facial anatomy features
MOUTH_LANDMARKS = set(range(0, 20)) | set(range(57, 92)) | set(range(267, 322)) | set(range(375, 410))
LEFT_EYE_LANDMARKS = set(range(22, 56)) | set(range(105, 115)) | set(range(221, 245))
RIGHT_EYE_LANDMARKS = set(range(252, 285)) | set(range(334, 345)) | set(range(441, 465))

# --- JOKER THEME COLOR PALETTE (BGR Format) ---
JOKER_RED = (30, 15, 200)     # Grimy Red for the smile
JOKER_BLUE = (180, 80, 10)    # Dark Clown Blue for the eye masks
JOKER_WHITE = (235, 240, 245) # Chalky white base coat

cap = cv2.VideoCapture(0)
print("Webcam starting... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    overlay = frame.copy()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # Map landmarks to pixel positions
            landmarked_pixels = []
            for lm in face_landmarks.landmark:
                x = int(lm.x * width)
                y = int(lm.y * height)
                landmarked_pixels.append((x, y))

            # Loop through connection blueprints to reconstruct facets
            for p1, neighbors in triangle_dict.items():
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        p2 = neighbors[i]
                        p3 = neighbors[j]
                        
                        if (p2, p3) in tessellation_connections or (p3, p2) in tessellation_connections:
                            pt1 = landmarked_pixels[p1]
                            pt2 = landmarked_pixels[p2]
                            pt3 = landmarked_pixels[p3]
                            triangle_pts = np.array([pt1, pt2, pt3], dtype=np.int32)
                            
                            # --- DETECT JOKER ZONE AND ASSIGN COLOR ---
                            # If all vertices fall into the mouth group -> paint it red
                            if p1 in MOUTH_LANDMARKS and p2 in MOUTH_LANDMARKS and p3 in MOUTH_LANDMARKS:
                                mesh_color = JOKER_RED
                            # If vertices touch eye groups -> paint them blue
                            elif (p1 in LEFT_EYE_LANDMARKS or p1 in RIGHT_EYE_LANDMARKS):
                                mesh_color = JOKER_BLUE
                            # Otherwise, it's face skin -> paint it white
                            else:
                                mesh_color = JOKER_WHITE
                            
                            # Fill the triangle facet on our mask overlay
                            cv2.fillPoly(overlay, [triangle_pts], mesh_color)
                            
                            # Draw subtle grey segment lines so the grid design stays visible
                            cv2.polylines(overlay, [triangle_pts], True, (140, 140, 140), 1)

        # --- ALPHA BLEND MASK WITH WEBCAM FEED ---
        alpha = 0.550  # 50% real background visibility
        beta = 0.450   # 50% digital paint opacity
        frame = cv2.addWeighted(overlay, beta, frame, alpha, 0)

    cv2.imshow("Joker Face Mesh Mask", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()