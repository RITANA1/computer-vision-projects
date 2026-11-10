
import cv2
from pathlib import Path

# ==========================================
# 1. FIND YUNET MODEL
# ==========================================

# Get the emotion_detection project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Model location
model_path = PROJECT_DIR / "models" / "face_detection_yunet_2023mar_int8.onnx"

# Check if model exists
if not model_path.exists():
    raise FileNotFoundError(f"YuNet model not found: {model_path}")

# ==========================================
# 2. CREATE FACE DETECTOR
# ==========================================

detector = cv2.FaceDetectorYN.create(
    str(model_path),
    "",
    (320, 320),
    0.8,    # Confidence threshold
    0.3,    # NMS threshold
    5000    # Maximum number of faces
)

print("YuNet model loaded successfully!")

# ==========================================
# 3. OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam!")

print("Webcam started! Press Q to quit.")

# ==========================================
# 4. REAL-TIME FACE DETECTION
# ==========================================

while True:

    # Capture frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Update detector input size
    detector.setInputSize((width, height))

    # Detect faces
    _, faces = detector.detect(frame)

    # ======================================
    # 5. DRAW FACE BOUNDING BOX
    # ======================================

    if faces is not None:

        for face in faces:

            # Extract bounding box
            x, y, w, h = face[:4].astype(int)

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Display label
            cv2.putText(
                frame,
                "Face Detected",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # ======================================
    # 6. DISPLAY RESULT
    # ======================================

    cv2.imshow("VisionLab - YuNet Face Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# 7. RELEASE RESOURCES
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Face detection stopped.")
