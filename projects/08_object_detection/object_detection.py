from ultralytics import YOLO
import cv2
import os


# ==========================================
# 1. Exact image path
# ==========================================

image_path = r"C:\Users\iafri\Downloads\VisionLab\projects\08_object_detection\images\twopeople_karting.png"

output_path = r"C:\Users\iafri\Downloads\VisionLab\projects\08_object_detection\outputs\twopeople_karting_yolo.png"


# ==========================================
# 2. Verify the image exists
# ==========================================

print("\n========================================")
print("YOLO PROJECT 08")
print("========================================")

print(f"📷 Image path:")
print(image_path)

if not os.path.exists(image_path):

    print("❌ ERROR: Image does NOT exist!")

    exit()

else:

    print("✅ Image exists!")


# ==========================================
# 3. Load the image ourselves
# ==========================================

image = cv2.imread(image_path)

if image is None:

    print("❌ OpenCV could not load the image!")

    exit()

else:

    print("✅ Image loaded successfully!")


# ==========================================
# 4. Show image dimensions
# ==========================================

height, width = image.shape[:2]

print(f"📐 Image size: {width} x {height}")


# ==========================================
# 5. Load YOLO
# ==========================================

model = YOLO("yolo26n.pt")

print("✅ YOLO model loaded successfully!")


# ==========================================
# 6. Run detection
# ==========================================

results = model(
    image_path,
    conf=0.25
)

print("✅ Detection completed!")


# ==========================================
# 7. Analyze detections
# ==========================================

object_count = 0

print("\n🎯 Detected objects:")

for result in results:

    for box in result.boxes:

        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        object_count += 1

        print(
            f"Object {object_count}: "
            f"{class_name} | "
            f"Confidence: {confidence:.2%} | "
            f"Box: ({x1}, {y1}, {x2}, {y2})"
        )


# ==========================================
# 8. Create annotated image
# ==========================================

result_image = results[0].plot()


# ==========================================
# 9. Save result
# ==========================================

cv2.imwrite(
    output_path,
    result_image
)

print("\n========================================")
print(f"🎯 Total objects detected: {object_count}")
print("========================================")

print("💾 Result saved to:")

print(output_path)


# ==========================================
# 10. Display result
# ==========================================

display_width = 900

scale = display_width / width

display_height = int(height * scale)

display_image = cv2.resize(
    result_image,
    (display_width, display_height)
)

cv2.namedWindow(
    "YOLO - Project 08",
    cv2.WINDOW_NORMAL
)

cv2.imshow(
    "YOLO - Project 08",
    display_image
)

cv2.waitKey(0)

cv2.destroyAllWindows()