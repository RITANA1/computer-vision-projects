import cv2

# ==========================================
# Load YuNet model
# ==========================================

model_path = "models/face_detection_yunet_2023mar_int8.onnx"

face_detector = cv2.FaceDetectorYN.create(
    model_path,
    "",
    (320, 320),
    0.6,      # confidence threshold
    0.3,      # NMS threshold
    5000      # top K
)

print("Face detector loaded!")


# ==========================================
# Open webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam opened!")


# ==========================================
# Face detection
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Frame size
    height, width = frame.shape[:2]

    # Tell YuNet the frame size
    face_detector.setInputSize((width, height))

    # Detect faces
    _, faces = face_detector.detect(frame)

    if faces is not None:

        print("Faces detected:", len(faces))

        for face in faces:

            x, y, w, h = face[:4].astype(int)

            # Bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Label
            cv2.putText(
                frame,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    else:
        print("No faces detected")

    # Show webcam
    cv2.imshow("YuNet Face Detection", frame)

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# Cleanup
# ==========================================

cap.release()
cv2.destroyAllWindows()
