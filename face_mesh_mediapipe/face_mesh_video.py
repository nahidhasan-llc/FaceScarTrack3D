import os
import cv2
import mediapipe as mp

# --- Setup Absolute Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, 'vid.mp4') 

print(f"Looking for the video file at: {video_path}")

# 1. Initialize FaceMesh using the legacy pipeline
# This completely bypasses the .task file and doesn't use the broken vision API
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Open your video file
cap = cv2.VideoCapture(video_path)

print("Processing video... Press 'q' inside the video window to exit early.")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Video reached the end or the file could not be opened.")
        break

    # OpenCV reads frames as BGR, MediaPipe needs RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find landmarks
    results = face_mesh.process(rgb_frame)

    # 3. If a face is found, draw the landmarks
    if results.multi_face_landmarks:
        height, width, _ = frame.shape
        
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 2, (100, 100, 0), -1)

    # Show the live frame window
    cv2.imshow("Face Mesh - Video Processing", frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Finished!")



# py -3.12 source\face_mesh_video.py