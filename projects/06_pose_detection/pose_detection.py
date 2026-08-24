import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# Load Pose Landmarker Model
# ==========================================

model_path = "models/pose_landmarker_lite.task"

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.PoseLandmarker.create_from_options(options)

print("Pose Landmarker loaded successfully! 🧍")


# ==========================================
# Open Webcam
# ==========================================

cap = cv2.VideoCapture(0)

timestamp_ms = 0


while True:

    success, frame = cap.read()

    if not success:
        print("Could not read webcam.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # OpenCV = BGR
    # MediaPipe = RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert frame to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Increase timestamp
    timestamp_ms += 33

    # Detect pose
    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ==========================================
    # Draw Pose Landmarks
    # ==========================================

    if result.pose_landmarks:

        for pose_landmarks in result.pose_landmarks:

            # Draw landmarks
            for landmark in pose_landmarks:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (255, 0, 255),
                    -1
                )


            # ==================================
            # Pose Connections
            # ==================================

            connections = [
                # Face
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 7),
                (0, 4),
                (4, 5),
                (5, 6),
                (6, 8),

                # Shoulders
                (11, 12),

                # Left arm
                (11, 13),
                (13, 15),

                # Right arm
                (12, 14),
                (14, 16),

                # Torso
                (11, 23),
                (12, 24),
                (23, 24),

                # Left leg
                (23, 25),
                (25, 27),

                # Right leg
                (24, 26),
                (26, 28),

                # Feet
                (27, 29),
                (29, 31),
                (28, 30),
                (30, 32)
            ]


            # Draw connections
            for start, end in connections:

                x1 = int(
                    pose_landmarks[start].x
                    * frame.shape[1]
                )

                y1 = int(
                    pose_landmarks[start].y
                    * frame.shape[0]
                )

                x2 = int(
                    pose_landmarks[end].x
                    * frame.shape[1]
                )

                y2 = int(
                    pose_landmarks[end].y
                    * frame.shape[0]
                )

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
            "No pose detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ==========================================
    # Display
    # ==========================================

    cv2.imshow(
        "Pose Detection - MediaPipe",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# Cleanup
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Pose detection stopped.")