import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1. Load and preprocess image
img = cv2.imread('circle.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

# 2. Find edge contours
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# 3. Fit ellipse to each valid contour
for cnt in contours:
    if len(cnt) >= 5:  # Minimum points to fit ellipse
        ellipse = cv2.fitEllipse(cnt)
        (xc, yc), (a, b), theta = ellipse
        cv2.ellipse(img, (int(xc), int(yc)), (int(a/2), int(b/2)), theta, 0, 360, (0,255,0), 2)

# 4. Show result
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Detected Ellipses')
plt.axis('off')
plt.show()