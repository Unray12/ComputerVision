import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
img = imread('test.jpg', mode = 'F')
img = img.astype(float)
# Sobel kernels
#Kx = np.array([[1,0,1],[2,0,2],[1,0,1]])
#Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

Kx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])
Ky = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]])

def convolve(img, kernel):
    m, n = kernel.shape
    y, x = img.shape
    out = np.zeros((y, x))
    for i in range(1, y-1):
        for j in range(1, x-1):
            window = img[i-1:i+2, j-1:j+2]
            out[i,j] = np.sum(window * kernel)
    return out
Ix = convolve(img, Kx)
Iy = convolve(img, Ky)
G = np.sqrt(Ix**2 + Iy**2)
edges = (G > 100).astype(np.uint8) * 255  # threshold at 100
plt.imshow(edges, cmap='gray')
plt.title('Edge map (manual)')
plt.show()