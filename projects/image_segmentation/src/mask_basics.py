
import cv2
import numpy as np

# 1. Create an empty image
height = 400
width = 600

image = np.zeros((height, width, 3), dtype=np.uint8)

# 2. Draw three objects
cv2.rectangle(image, (50, 100), (200, 300), (0, 0, 255), -1)

cv2.circle(image, (350, 200), 90, (255, 0, 0), -1)

cv2.ellipse(image, (500, 200), (50, 100), 0, 0, 360, (0, 255, 255), -1)

# 3. Create a black mask
mask = np.zeros((height, width), dtype=np.uint8)

# 4. Select the blue circle
cv2.circle(mask, (350, 200), 90, 255, -1)

# 5. Extract the object
segmented_object = cv2.bitwise_and(
    image,
    image,
    mask=mask
)

# 6. Create a green overlay
overlay = image.copy()
overlay[mask == 255] = (0, 255, 0)

result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

# 7. Display results
mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

final_display = np.hstack((image, mask_bgr, result))

print("Original image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Mask values:", np.unique(mask))

cv2.imshow("Original | Mask | Segmentation", final_display)

cv2.waitKey(0)
cv2.destroyAllWindows()
