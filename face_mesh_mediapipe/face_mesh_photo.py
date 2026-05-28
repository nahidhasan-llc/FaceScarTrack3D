import os
import cv2
from mediapipe.python.solutions import face_mesh



# --- Setup Absolute Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# Put your picture inside the 'source' folder and change the name here:
image_name = 'burn_face.jpg' 
image_path = os.path.join(script_dir, image_name)
output_path = os.path.join(script_dir, 'burn_face_mesh_photo.jpg')

# 1. Load the Image
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not find or open the image at: {image_path}")
    print("Please make sure your photo is sitting inside the 'source' folder.")
    exit()

print(f"Successfully loaded: {image_path}")

# 2. Initialize FaceMesh using the reliable classic pipeline
mp_face_mesh = face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,  # Set to True tells MediaPipe to optimize for static pictures!
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# OpenCV reads images as BGR, MediaPipe needs RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and find landmarks
results = face_mesh.process(rgb_image)

# 3. If a face is found, draw the landmarks
if results.multi_face_landmarks:
    height, width, _ = image.shape
    print(f"Face detected! Drawing landmarks onto the photo...")
    
    for face_landmarks in results.multi_face_landmarks:
        for landmark in face_landmarks.landmark:
            # Convert normalized coordinates (0.0 to 1.0) into exact pixel positions
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            
            # Draw small dots on the face
            cv2.circle(image, (x, y), 2, (100, 100, 0), -1)
            
    # Save a copy of the new face-meshed image to your folder
    cv2.imwrite(output_path, image)
    print(f"Saved a copy of the result to: {output_path}")

else:
    print("No face detected in the photo. Try an image with clearer lighting or a direct headshot.")

# 4. Display the photo on your screen
cv2.imshow("Face Mesh - Photo Processing", image)

print("Displaying photo window. Press ANY key on your keyboard to close it.")
cv2.waitKey(0) # 0 means pause indefinitely until a key is pressed
cv2.destroyAllWindows()


# py -3.12 face_mesh_photo.py