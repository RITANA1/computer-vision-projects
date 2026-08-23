import cv2

# ==========================================
# 1. Load YuNet face detection model
# ==========================================

model_path = "../02_face_detection/models/face_detection_yunet_2023mar_int8.onnx"

face_detector = cv2.FaceDetectorYN.create(
    model_path,
    "",
    (320, 320),
    0.6,      # confidence threshold
    0.3,      # NMS threshold
    5000      # maximum detections
)

print("YuNet loaded!")


# ==========================================
# 2. Open webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam opened!")


# ==========================================
# 3. Main loop
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Tell YuNet the input size
    face_detector.setInputSize((width, height))

    # Detect faces
    _, faces = face_detector.detect(frame)


    # ==========================================
    # 4. Process detected faces
    # ==========================================

    if faces is not None:

        for face in faces:

            # ----------------------------------
            # Face bounding box
            # ----------------------------------

            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            # ----------------------------------
            # Right Eye
            # ----------------------------------

            right_eye_x = int(face[4])
            right_eye_y = int(face[5])

            cv2.circle(
                frame,
                (right_eye_x, right_eye_y),
                8,
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                "Right Eye",
                (right_eye_x - 40, right_eye_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )


            # ----------------------------------
            # Left Eye
            # ----------------------------------

            left_eye_x = int(face[6])
            left_eye_y = int(face[7])

            cv2.circle(
                frame,
                (left_eye_x, left_eye_y),
                8,
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                "Left Eye",
                (left_eye_x - 40, left_eye_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )


            # ----------------------------------
            # Nose
            # ----------------------------------

            nose_x = int(face[8])
            nose_y = int(face[9])

            cv2.circle(
                frame,
                (nose_x, nose_y),
                6,
                (0, 255, 255),
                -1
            )

            cv2.putText(
                frame,
                "Nose",
                (nose_x + 10, nose_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )


            # ----------------------------------
            # Right Mouth Corner
            # ----------------------------------

            right_mouth_x = int(face[10])
            right_mouth_y = int(face[11])

            cv2.circle(
                frame,
                (right_mouth_x, right_mouth_y),
                6,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                "Right Mouth",
                (right_mouth_x + 10, right_mouth_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )


            # ----------------------------------
            # Left Mouth Corner
            # ----------------------------------

            left_mouth_x = int(face[12])
            left_mouth_y = int(face[13])

            cv2.circle(
                frame,
                (left_mouth_x, left_mouth_y),
                6,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                "Left Mouth",
                (left_mouth_x + 10, left_mouth_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )


            # ----------------------------------
            # Confidence
            # ----------------------------------

            confidence = face[14]

            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}",
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


    # ==========================================
    # 5. Display
    # ==========================================

    cv2.imshow("Eye & Facial Landmark Detection", frame)


    # ==========================================
    # 6. Press Q to quit
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# 7. Cleanup
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Program finished.")