
import cv2
import numpy as np
import time
import os
from collections import deque
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_NAME = "yolo26n-seg.pt"
CONFIDENCE = 0.35
IMAGE_SIZE = 416
CAMERA_INDEX = 0

OUTPUT_DIR = "projects/image_segmentation/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

video_path = os.path.join(
    OUTPUT_DIR, f"segmentation_{timestamp}.mp4"
)

# ==========================================
# 2. LOAD MODEL
# ==========================================

print("Loading YOLO segmentation model...")
model = YOLO(MODEL_NAME)
print("Model loaded successfully!")

# ==========================================
# 3. INITIALIZE WEBCAM
# ==========================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# Lower resolution helps CPU performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

camera_fps = cap.get(cv2.CAP_PROP_FPS)
if not camera_fps or camera_fps <= 0:
    camera_fps = 20.0

# ==========================================
# 4. INITIALIZE VIDEO WRITER
# ==========================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    video_path,
    fourcc,
    camera_fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not initialize video writer.")

# FPS smoothing
fps_history = deque(maxlen=20)

print("\nSegmentation started!")
print("Q = Quit | S = Screenshot")

# ==========================================
# 5. MAIN LOOP
# ==========================================

try:
    while True:

        start_time = time.time()

        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        # AI prediction
        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            max_det=10,
            verbose=False
        )

        result = results[0]

        # Prepare overlay only once per frame
        overlay = frame.copy()
        detected_count = 0

        if result.masks is not None and result.boxes is not None:

            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes

            for i, mask in enumerate(masks):

                # Resize mask to webcam dimensions
                mask = cv2.resize(
                    mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

                mask = mask > 0.5

                # Generate a different color for each instance
                hue = (i * 45) % 180

                color = cv2.cvtColor(
                    np.uint8([[[hue, 255, 255]]]),
                    cv2.COLOR_HSV2BGR
                )[0][0]

                color = tuple(int(c) for c in color)

                # Apply mask to overlay
                overlay[mask] = color

                # Extract object information
                x1, y1, x2, y2 = map(
                    int, boxes.xyxy[i].cpu().numpy()
                )

                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                label = result.names[class_id]

                detected_count += 1

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Draw label
                text = f"{label} {confidence:.2f}"

                cv2.putText(
                    frame,
                    text,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                # Draw segmentation boundary
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                cv2.drawContours(
                    frame, contours, -1, color, 2
                )

        # Blend all masks in one operation
        frame = cv2.addWeighted(
            overlay, 0.35,
            frame, 0.65,
            0
        )

        # ==================================
        # 6. FPS AND INFORMATION
        # ==================================

        elapsed = time.time() - start_time
        current_fps = 1.0 / max(elapsed, 1e-6)

        fps_history.append(current_fps)
        average_fps = sum(fps_history) / len(fps_history)

        cv2.putText(
            frame,
            f"FPS: {average_fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Objects: {detected_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ==================================
        # 7. RECORD VIDEO
        # ==================================

        writer.write(frame)

        # ==================================
        # 8. DISPLAY
        # ==================================

        cv2.imshow(
            "YOLO Instance Segmentation",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):

            screenshot_name = (
                f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            )

            screenshot_path = os.path.join(
                OUTPUT_DIR, screenshot_name
            )

            cv2.imwrite(screenshot_path, frame)

            print("Screenshot saved:", screenshot_path)

        elif key == ord("q"):
            print("Stopping segmentation...")
            break

finally:
    # ======================================
    # 9. CLEANUP
    # ======================================

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\nSegmentation stopped.")
    print("Video saved:", video_path)
