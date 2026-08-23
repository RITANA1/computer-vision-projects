import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# Load Hand Landmarker
# =========================

model_path = "models/hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

print("Hand Landmarker loaded successfully! 🖐️")


# =========================
# Open Webcam
# =========================

cap = cv2.VideoCapture(0)


while True:

    success, frame = cap.read()

    if not success:
        print("Could not read webcam.")
        break

    # Mirror the webcam
    frame = cv2.flip(frame, 1)

    # OpenCV uses BGR
    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert OpenCV image to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hands
    result = detector.detect(mp_image)

    # Draw landmarks
    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            # Draw 21 landmarks
            for landmark in hand_landmarks:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (255, 0, 255),
                    -1
                )

            # Draw connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]

            for start, end in connections:

                x1 = int(hand_landmarks[start].x * frame.shape[1])
                y1 = int(hand_landmarks[start].y * frame.shape[0])

                x2 = int(hand_landmarks[end].x * frame.shape[1])
                y2 = int(hand_landmarks[end].y * frame.shape[0])

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 255),
                    2
                )

    else:
        cv2.putText(
            frame,
            "No hand detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Show result
    cv2.imshow("Hand Detection - MediaPipe", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()