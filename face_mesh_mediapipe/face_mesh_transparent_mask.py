import cv2
import numpy as np
import mediapipe as mp

# 1. Initialize FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,  # Standard 468 points is ideal for clean face tessellation mapping
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Grab the official connection blueprint
tessellation_connections = mp_face_mesh.FACEMESH_TESSELATION

# Build the connection dictionary ONCE outside the loop to save computer processing speed
triangle_dict = {}
for connection in tessellation_connections:
    p1, p2 = connection
    triangle_dict.setdefault(p1, []).append(p2)

# 3. Open your local webcam
cap = cv2.VideoCapture(0)
print("Webcam starting... Press 'q' inside the video window to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    height, width, _ = frame.shape
    
    # Create an isolated layer copy of the live frame to paint our shapes onto safely
    overlay = frame.copy()

    # OpenCV reads frames as BGR, MediaPipe needs RGB
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

            # Loop through our pre-built connection map to find interlocking triangles
            for p1, neighbors in triangle_dict.items():
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        p2 = neighbors[i]
                        p3 = neighbors[j]
                        
                        # Check if p2 and p3 form a closed triangle line
                        if (p2, p3) in tessellation_connections or (p3, p2) in tessellation_connections:
                            # Grab pixel coordinates for our 3 vertices
                            pt1 = landmarked_pixels[p1]
                            pt2 = landmarked_pixels[p2]
                            pt3 = landmarked_pixels[p3]
                            
                            triangle_pts = np.array([pt1, pt2, pt3], dtype=np.int32)
                            
                            # --- COLOR SELECTION ---
                            # Classic glowing cyber cyan/teal color in BGR format
                            mesh_color = (255, 0, 180) 
                            
                            # Paint the solid triangle facets directly onto the overlay copy
                            cv2.fillPoly(overlay, [triangle_pts], mesh_color)
                            
                            # Draw clean, bright white wireframe lines over the triangle boundaries
                            cv2.polylines(overlay, [triangle_pts], True, (255, 255, 255), 1)

        # --- ALPHA BLENDING TRANSPARENCY ---
        # Adjust these decimals to change mask visibility! (They must add up to 1.0)
        alpha = 0.85  # 65% weight given to your actual webcam camera feed
        beta = 0.15   # 35% weight given to the glowing neon triangle overlay layer
        
        # Merge the overlay back onto our main video frame
        frame = cv2.addWeighted(overlay, beta, frame, alpha, 0)

    # Show the combined transparent hologram output window
    cv2.imshow("Cyberpunk Hologram Face Mesh", frame)
    
    # Hit 'q' on your keyboard to close the video preview gracefully
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Finished safely!")