import os
import cv2
import mediapipe as mp

# 1. Initialize FaceMesh using the classic, reliable pipeline
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Open your local webcam (0 is usually the default laptop/USB camera)
cap = cv2.VideoCapture(0)

print("Webcam opening... Press 'q' inside the video window to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame from webcam.")
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

    # Show the live webcam frame window
    cv2.imshow("Face Mesh - Webcam Processing", frame)
    
    # 1ms delay is perfect for live webcams to keep the feed smooth and real-time
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Finished!")



# py -3.12 source\face_mesh_webcam.py