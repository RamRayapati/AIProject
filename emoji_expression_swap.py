import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

# Load Emojis
def load_emoji(path):
    emoji = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if emoji is None:
        raise ValueError(f"Could not load emoji at path: {path}")
    return emoji

happy_emoji = load_emoji(r"C:\Users\ramra\Desktop\ai pro\happy_emoji.png")
sad_emoji = load_emoji(r"C:\Users\ramra\Desktop\ai pro\sad.png")
surprised_emoji = load_emoji(r"C:\Users\ramra\Desktop\ai pro\suprised.png")
neutral_emoji = load_emoji(r"C:\Users\ramra\Desktop\ai pro\4805c697-8150-4014-a529-0c52aab1583b.png")

# Overlay PNG with transparency
def overlay_transparent(background, overlay, x, y, overlay_size=None):
    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size)

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background.shape[1] or y + h > background.shape[0]:
        return background

    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (
            alpha_overlay * overlay[:, :, c] +
            alpha_background * background[y:y+h, x:x+w, c]
        )
    return background

# Detect facial expression based on landmarks
def detect_expression(landmarks):
    left_eye = landmarks[159].y - landmarks[145].y
    right_eye = landmarks[386].y - landmarks[374].y
    mouth_open = landmarks[13].y - landmarks[14].y

    if mouth_open > 0.08 and left_eye > 0.04 and right_eye > 0.04:
        return "surprised"
    elif mouth_open > 0.04:
        return "happy"
    elif left_eye < 0.02 and right_eye < 0.02:
        return "sad"
    else:
        return "neutral"

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            expression = detect_expression(landmarks.landmark)

            # Get coordinates for forehead (e.g., landmark 10 - top of head)
            forehead_x = int(landmarks.landmark[10].x * w) - 50
            forehead_y = int(landmarks.landmark[10].y * h) - 50

            # Choose emoji
            if expression == "happy":
                emoji = happy_emoji
            elif expression == "sad":
                emoji = sad_emoji
            elif expression == "surprised":
                emoji = surprised_emoji
            else:
                emoji = neutral_emoji

            # Overlay emoji
            frame = overlay_transparent(frame, emoji, forehead_x, forehead_y, (100, 100))

    cv2.imshow('Face Detection with Emoji', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
