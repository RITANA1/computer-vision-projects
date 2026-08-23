import cv2

# Open webcam
cap = cv2.VideoCapture(0)

# YELLOW color range
lower_yellow = (20, 100, 100)
upper_yellow = (35, 255, 255)

# Create kernel
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

while True:

    # Read frame from camera
    ret, frame = cap.read()

    if not ret:
        print("Could not read camera.")
        break

    # BGR → HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create YELLOW mask
    mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow
    )

    # Remove small noise
    clean_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small holes
    clean_mask = cv2.morphologyEx(
        clean_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find contours
    contours, hierarchy = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw boxes
    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "YELLOW OBJECT",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Show camera
    cv2.imshow("Real-Time Yellow Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
cap.release()
cv2.destroyAllWindows()