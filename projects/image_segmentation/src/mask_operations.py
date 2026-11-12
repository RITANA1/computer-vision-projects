
import cv2
import numpy as np

# 1. Create an image
height = 400
width = 600

image = np.zeros((height, width, 3), dtype=np.uint8)

# Draw some objects
cv2.rectangle(image, (50, 100), (200, 300), (0, 0, 255), -1)
cv2.circle(image, (350, 200), 90, (255, 0, 0), -1)
cv2.ellipse(image, (500, 200), (50, 100), 0, 0, 360, (0, 255, 255), -1)


# 2. Create a binary mask using NumPy
mask = np.zeros((height, width), dtype=np.uint8)

# Select a rectangular region
mask[100:300, 50:200] = 255


# 3. Extract selected pixels
segmented = cv2.bitwise_and(
    image,
    image,
    mask=mask
)


# 4. Count selected pixels
selected_pixels = np.count_nonzero(mask == 255)

total_pixels = height * width

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)

print("Selected pixels:", selected_pixels)
print("Total pixels:", total_pixels)

print("Selected percentage:",
      round(selected_pixels / total_pixels * 100, 2), "%")


# 5. Display results
mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

display = np.hstack((image, mask_bgr, segmented))

cv2.imshow("Original | Mask | Extracted Region", display)

cv2.waitKey(0)
cv2.destroyAllWindows()
