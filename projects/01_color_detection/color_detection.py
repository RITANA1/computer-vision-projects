import cv2

# 1. Read image
image = cv2.imread(
    "projects/01_color_detection/images/colorful_pencils.jpg"
)
# 2. Convert BGR → HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 3. Define RED color range
lower_red = (0, 100, 100)
upper_red = (10, 255, 255)

# 4. Create RED mask
mask = cv2.inRange(
    hsv,
    lower_red,
    upper_red
)

# 5. Create kernel
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

# 6. Clean mask - remove small white noise
clean_mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)

# 7. Clean mask - fill small black holes
clean_mask = cv2.morphologyEx(
    clean_mask,
    cv2.MORPH_CLOSE,
    kernel
)

# 8. Find contours
contours, hierarchy = cv2.findContours(
    clean_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 9. Draw bounding boxes
for contour in contours:

    x, y, w, h = cv2.boundingRect(contour)

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

# 10. Create adjustable windows
cv2.namedWindow("RED Objects", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RED Objects", 800, 600)

cv2.namedWindow("Clean RED Mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Clean RED Mask", 800, 600)

# 11. Display results
cv2.imshow("RED Objects", image)
cv2.imshow("Clean RED Mask", clean_mask)

# 12. Wait for a key
cv2.waitKey(0)

# 13. Close windows
cv2.destroyAllWindows()