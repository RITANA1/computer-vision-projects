
import cv2
import time

# ==========================================
# 1. LOAD VIDEO
# ==========================================

video_path = r"C:\Users\iafri\Downloads\VisionLab\projects\object_tracking\videos\cars.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Cannot open video!")
    exit()

print("Video loaded successfully!")

# ==========================================
# 2. VIDEO DISPLAY SETTINGS
# ==========================================

DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 450

window_name = "VisionLab - Object Tracking"

# ==========================================
# 3. CREATE CSRT TRACKER
# ==========================================

def create_tracker():

    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    elif hasattr(cv2, "legacy") and hasattr(
        cv2.legacy, "TrackerCSRT_create"
    ):
        return cv2.legacy.TrackerCSRT_create()

    else:
        raise RuntimeError("CSRT tracker is unavailable!")


tracker = None
tracking = False
object_lost = False
paused = False

prev_time = time.time()

# ==========================================
# 4. CREATE WINDOW
# ==========================================

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(
    window_name,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT
)

print("\nOBJECT TRACKING")
print("--------------------------------")
print("S = Select object")
print("R = Reset tracking")
print("P = Pause / Resume")
print("Q = Quit")
print("--------------------------------")

# ==========================================
# 5. MAIN VIDEO LOOP
# ==========================================

while True:

    # Read next frame when not paused
    if not paused:

        ret, frame = cap.read()

        if not ret:
            print("Video finished!")
            break

        # Resize frame
        frame = cv2.resize(
            frame,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )

        current_frame = frame.copy()

    else:

        frame = current_frame.copy()

    display = frame.copy()

    # ======================================
    # 6. UPDATE OBJECT TRACKING
    # ======================================

    if tracking:

        success, box = tracker.update(frame)

        if success:

            x, y, w, h = map(int, box)

            # Draw bounding box
            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            # Tracking label
            cv2.putText(
                display,
                "TRACKING",
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            object_lost = False

        else:

            tracking = False
            object_lost = True

    # ======================================
    # 7. DISPLAY STATUS
    # ======================================

    if tracking:
        status = "STATUS: TRACKING"
        status_color = (0, 255, 0)

    elif object_lost:
        status = "STATUS: OBJECT LOST"
        status_color = (0, 0, 255)

    else:
        status = "STATUS: WAITING FOR SELECTION"
        status_color = (0, 255, 255)

    cv2.putText(
        display,
        status,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2
    )

    # ======================================
    # 8. CALCULATE FPS
    # ======================================

    current_time = time.time()

    fps = 1 / max(current_time - prev_time, 1e-6)

    prev_time = current_time

    cv2.putText(
        display,
        f"FPS: {int(fps)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2
    )

    # ======================================
    # 9. DISPLAY INSTRUCTIONS
    # ======================================

    cv2.putText(
        display,
        "S: Select | R: Reset | P: Pause | Q: Quit",
        (15, DISPLAY_HEIGHT - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.imshow(window_name, display)

    # ======================================
    # 10. KEYBOARD CONTROLS
    # ======================================

    key = cv2.waitKey(30) & 0xFF

    # SELECT OBJECT
    if key == ord('s'):

        # Pause while selecting
        paused = True

        bbox = cv2.selectROI(
            window_name,
            frame,
            False,
            False
        )

        if bbox[2] > 0 and bbox[3] > 0:

            tracker = create_tracker()

            tracker.init(frame, bbox)

            tracking = True
            object_lost = False

            print("Object selected! Tracking started.")

        paused = False

    # RESET TRACKING
    elif key == ord('r'):

        tracker = None
        tracking = False
        object_lost = False

        print("Tracking reset. Press S to select again.")

    # PAUSE / RESUME
    elif key == ord('p'):

        paused = not paused

        if paused:
            print("Video paused.")
        else:
            print("Video resumed.")

    # QUIT
    elif key == ord('q'):

        print("Exiting Object Tracking...")
        break

# ==========================================
# 11. RELEASE RESOURCES
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("Object Tracking completed!")
