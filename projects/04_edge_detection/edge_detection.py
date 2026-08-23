import cv2

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam opened!")

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert BGR → Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny
    edges = cv2.Canny(
        gray,
        100,
        200
    )

    # Show original
    cv2.imshow(
        "Original",
        frame
    )

    # Show edges
    cv2.imshow(
        "Canny Edge Detection",
        edges
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release webcam
cap.release()
cv2.destroyAllWindows()

print("Program finished!")