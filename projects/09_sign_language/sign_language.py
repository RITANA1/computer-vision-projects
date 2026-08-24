import cv2
import mediapipe as mp
import math
import time


# ==========================================
# MediaPipe
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# Webcam
# ==========================================

cap = cv2.VideoCapture(0)


# ==========================================
# FPS
# ==========================================

previous_time = 0


# ==========================================
# Distance
# ==========================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ==========================================
# Finger detection
# ==========================================

def get_fingers(hand):

    lm = hand.landmark

    fingers = []

    # Thumb
    if lm[4].x < lm[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index
    if lm[8].y < lm[6].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Middle
    if lm[12].y < lm[10].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Ring
    if lm[16].y < lm[14].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Pinky
    if lm[20].y < lm[18].y:
        fingers.append(1)
    else:
        fingers.append(0)

    return fingers


# ==========================================
# Gesture recognition
# ==========================================

def recognize_gesture(hand):

    fingers = get_fingers(hand)

    lm = hand.landmark

    # Fist
    if fingers == [0, 0, 0, 0, 0]:
        return "FIST"

    # Thumbs Up
    if fingers == [1, 0, 0, 0, 0]:

        if lm[4].y < lm[3].y:
            return "THUMBS UP"

    # One
    if fingers == [0, 1, 0, 0, 0]:
        return "ONE"

    # Peace
    if fingers == [0, 1, 1, 0, 0]:
        return "PEACE"

    # Three
    if fingers == [0, 1, 1, 1, 0]:
        return "THREE"

    # Open hand
    if fingers == [1, 1, 1, 1, 1]:
        return "OPEN HAND"

    # Rock
    if fingers == [1, 0, 0, 0, 1]:
        return "ROCK"

    # Cross fingers
    if fingers == [0, 1, 0, 0, 1]:
        return "CROSS FINGERS"

    return "UNKNOWN"


# ==========================================
# Main loop
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        print("Camera error")
        break


    # Mirror
    frame = cv2.flip(frame, 1)


    # ======================================
    # RGB
    # ======================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ======================================
    # Detect hands
    # ======================================

    results = hands.process(rgb)


    # ======================================
    # Process hands
    # ======================================

    if results.multi_hand_landmarks:

        for index, hand_landmarks in enumerate(
            results.multi_hand_landmarks
        ):

            lm = hand_landmarks.landmark


            # ==================================
            # Draw landmarks
            # ==================================

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


            # ==================================
            # Finger information
            # ==================================

            fingers = get_fingers(
                hand_landmarks
            )

            finger_count = sum(fingers)


            # ==================================
            # Gesture
            # ==================================

            gesture = recognize_gesture(
                hand_landmarks
            )


            # ==================================
            # Bounding box
            # ==================================

            h, w, _ = frame.shape

            x_coordinates = [
                int(point.x * w)
                for point in lm
            ]

            y_coordinates = [
                int(point.y * h)
                for point in lm
            ]

            x_min = min(x_coordinates)
            x_max = max(x_coordinates)

            y_min = min(y_coordinates)
            y_max = max(y_coordinates)


            # Draw bounding box

            cv2.rectangle(
                frame,
                (x_min - 10, y_min - 10),
                (x_max + 10, y_max + 10),
                (255, 0, 0),
                2
            )


            # ==================================
            # Hand center
            # ==================================

            center_x = int(
                sum(x_coordinates) / len(x_coordinates)
            )

            center_y = int(
                sum(y_coordinates) / len(y_coordinates)
            )


            cv2.circle(
                frame,
                (center_x, center_y),
                6,
                (0, 0, 255),
                -1
            )


            # ==================================
            # Hand size
            # ==================================

            hand_width = x_max - x_min

            hand_height = y_max - y_min


            # ==================================
            # Display information
            # ==================================

            y_position = 40 + index * 150


            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"Fingers: {finger_count}",
                (20, y_position + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Center: ({center_x}, {center_y})",
                (20, y_position + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Size: {hand_width} x {hand_height}",
                (20, y_position + 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


    # ==========================================
    # FPS
    # ==========================================

    current_time = time.time()

    fps = 1 / (current_time - previous_time)

    previous_time = current_time


    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    # ==========================================
    # Show
    # ==========================================

    cv2.imshow(
        "Sign Language",
        frame
    )


    # Q to quit

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# Cleanup
# ==========================================

cap.release()

cv2.destroyAllWindows()

hands.close()