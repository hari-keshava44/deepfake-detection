import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Function to calculate head pose (pitch, yaw, roll)
def get_head_pose(face_landmarks, image_shape):
    # Extract key facial landmarks: nose tip, chin, left eye, right eye, mouth corners
    h, w, _ = image_shape
    points = []
    for idx in [1, 9, 33, 263, 61, 291]:  # Indices for specific landmarks
        x = int(face_landmarks.landmark[idx].x * w)
        y = int(face_landmarks.landmark[idx].y * h)
        points.append((x, y))

    # Convert points to numpy array
    points = np.array(points, dtype="double")

    # Define camera matrix and distortion coefficients
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # Solve PnP problem to estimate head pose
    success, rotation_vector, translation_vector = cv2.solvePnP(
        np.array([
            points[0],  # Nose tip
            points[1],  # Chin
            points[2],  # Left eye
            points[3],  # Right eye
            points[4],  # Mouth left corner
            points[5]   # Mouth right corner
        ]),
        np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye
            (225.0, 170.0, -135.0),   # Right eye
            (-150.0, -150.0, -125.0), # Mouth left corner
            (150.0, -150.0, -125.0)   # Mouth right corner
        ]), camera_matrix, dist_coeffs)

    # Convert rotation vector to Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pitch, yaw, roll = cv2.decomposeProjectionMatrix(np.hstack([rotation_matrix, translation_vector]))[6]
    return pitch[0], yaw[0], roll[0]

# Function to detect deepfakes based on head pose inconsistencies
def detect_deepfake(video_path, threshold=10.0):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    pose_inconsistencies = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face landmarks
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get head pose
                pitch, yaw, roll = get_head_pose(face_landmarks, frame.shape)
                print(f"Frame {frame_count}: Pitch={pitch:.2f}, Yaw={yaw:.2f}, Roll={roll:.2f}")

                # Check for pose inconsistencies
                if abs(pitch) > threshold or abs(yaw) > threshold or abs(roll) > threshold:
                    pose_inconsistencies.append(frame_count)

        frame_count += 1

    cap.release()

    # Report results
    if pose_inconsistencies:
        print(f"Suspicious frames detected at: {pose_inconsistencies}")
        print("Deepfake detected!")
    else:
        print("No deepfake detected.")

# Example usage
if __name__ == "__main__":
    video_path = "input_video.mp4"  # Replace with your video file
    detect_deepfake(video_path)