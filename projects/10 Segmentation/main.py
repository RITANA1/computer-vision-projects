
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the image
image = cv2.imread("images/test_image.jpg")

if image is None:
    raise FileNotFoundError("Check the image path!")

# 2. Convert BGR to RGB for correct display in Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 4. Choose a color range: green, for example
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])

# 5. Create a mask: white = selected color, black = everything else
mask = cv2.inRange(hsv, lower_green, upper_green)

# 6. Keep only the selected region
segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# 7. Display the results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Segmentation Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(segmented)
plt.title("Segmented Object")
plt.axis("off")

plt.tight_layout()
plt.show()

# 8. Save the results
cv2.imwrite("outputs/segmentation_mask.png", mask)
cv2.imwrite(
    "outputs/segmented_object.png",
    cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
)