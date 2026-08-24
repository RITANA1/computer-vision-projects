import cv2
import os


# ==========================================
# 1. Load Image
# ==========================================

image_path = "images/fruits.jpg"

image = cv2.imread(image_path)

if image is None:
    print("❌ Could not load image.")
    exit()

print("✅ Image loaded successfully!")


# ==========================================
# 2. Convert to Grayscale
# ==========================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# ==========================================
# 3. Blur the Image
# ==========================================

blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# ==========================================
# 4. Detect Edges with Canny
# ==========================================

edges = cv2.Canny(
    blur,
    50,
    150
)


# ==========================================
# 5. Find External Contours
# ==========================================

contours, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print(f"🔎 Total contours found: {len(contours)}")


# ==========================================
# 6. Create Result Image
# ==========================================

result = image.copy()

object_count = 0


# ==========================================
# 7. Analyze Each Contour
# ==========================================

for contour in contours:

    # Calculate contour area
    area = cv2.contourArea(contour)

    # Ignore very small contours
    if area < 1000:
        continue

    object_count += 1


    # ------------------------------------------
    # Bounding Rectangle
    # ------------------------------------------

    x, y, w, h = cv2.boundingRect(contour)


    # ------------------------------------------
    # Contour Center Using Moments
    # ------------------------------------------

    moments = cv2.moments(contour)

    if moments["m00"] != 0:

        center_x = int(
            moments["m10"] / moments["m00"]
        )

        center_y = int(
            moments["m01"] / moments["m00"]
        )

    else:

        center_x = x + w // 2
        center_y = y + h // 2


    # ------------------------------------------
    # Draw Contour
    # ------------------------------------------

    cv2.drawContours(
        result,
        [contour],
        -1,
        (0, 255, 0),
        2
    )


    # ------------------------------------------
    # Draw Bounding Box
    # ------------------------------------------

    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )


    # ------------------------------------------
    # Draw Center Point
    # ------------------------------------------

    cv2.circle(
        result,
        (center_x, center_y),
        5,
        (0, 0, 255),
        -1
    )


    # ------------------------------------------
    # Add Label
    # ------------------------------------------

    cv2.putText(
        result,
        f"Object {object_count}",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


    # ------------------------------------------
    # Print Information
    # ------------------------------------------

    print(
        f"Object {object_count}: "
        f"Area = {area:.0f}, "
        f"Center = ({center_x}, {center_y}), "
        f"Bounding Box = ({x}, {y}, {w}, {h})"
    )


# ==========================================
# 8. Print Final Result
# ==========================================

print()
print("=" * 40)
print(f"🍎 Objects detected: {object_count}")
print("=" * 40)


# ==========================================
# 9. Create Output Folder
# ==========================================

os.makedirs(
    "outputs",
    exist_ok=True
)


# ==========================================
# 10. Save Result
# ==========================================

output_path = "outputs/fruit_contours.png"

cv2.imwrite(
    output_path,
    result
)

print(f"💾 Result saved to: {output_path}")


# ==========================================
# 11. Display Result
# ==========================================

window_name = "Project 07 - Contour Detection"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    900,
    600
)

cv2.imshow(
    window_name,
    result
)

cv2.waitKey(0)

cv2.destroyAllWindows()