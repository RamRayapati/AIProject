import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam")
else:
    print("✅ Webcam opened successfully")
cap.release()
