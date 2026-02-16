import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# --- INIT FACE LANDMARKER ---
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# --- START CAMERA ---
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cv2.namedWindow("Face Landmarks", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Face Landmarks", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    # Create a blank background for face landmarks
    h, w, _ = frame.shape
    face_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]  # first face

        # Draw all points
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(face_canvas, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Face Landmarks", face_canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()