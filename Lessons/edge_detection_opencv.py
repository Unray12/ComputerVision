import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image as grayscale
img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel X
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# Sobel Y
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
# Combine the two gradients
sobel = cv2.magnitude(sobelx, sobely)
sobel = np.uint8(np.clip(sobel, 0, 255))



plt.imshow(sobel, cmap='gray')
plt.title('Sobel Edge')
plt.axis('off')
plt.show()

