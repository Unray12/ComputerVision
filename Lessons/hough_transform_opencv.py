import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('frame_20260212_200057_045.jpg', 0)
edges = cv2.Canny(img, 100, 150)

lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img_color, (x1, y1), (x2, y2), (0,0,255), 2)

plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title('Detected Lines')
plt.show()