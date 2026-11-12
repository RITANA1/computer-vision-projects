import cv2
import numpy as np

# Create a sample image
image = np.zeros((400, 600, 3), dtype=np.uint8)

# Draw a blue circle and red rectangle
cv2.circle(image, (200, 200), 100, (255, 0, 0), -1)
cv2.rectangle(image, (350, 100), (500, 300), (0, 0, 255), -1)

# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define blue color range
lower_blue = np.array([90, 100, 50])
upper_blue = np.array([130, 255, 255])

# Create mask
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Extract blue object
result = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow("Original", image)
cv2.imshow("Blue Mask", mask)
cv2.imshow("Blue Object", result)

cv2.waitKey(0)
cv2.destroyAllWindows()