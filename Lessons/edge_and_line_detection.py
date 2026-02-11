import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read image and convert to grayscale
img = cv2.imread('lane.png')           # Change to your image path
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 2: Edge detection using Canny
edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

# Step 3: Hough Transform for line detection
# rho=1, theta=1 degree in radian, threshold=100 votes
lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

# Prepare to draw lines on a color image
img_lines = img.copy()

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Calculate the endpoints of the line for display purposes
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the results
plt.figure(figsize=(16,6))
plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
plt.title('Detected Lines (Hough Transform)')
plt.axis('off')

plt.tight_layout()
plt.show()