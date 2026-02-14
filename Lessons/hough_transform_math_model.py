import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('frame_20260212_200109_142.jpg')  # Change to your image path
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, threshold1=100, threshold2=200)


rows, cols = edges.shape    #Size of the edge image, be equal to original image

thetas = np.deg2rad(np.arange(-90, 90, 1))  #Step = 1 degree in default, converted to radian
diag_len = int(np.ceil(np.sqrt(rows**2 + cols**2))) #Diagonal of the original image
rhos = np.linspace(-diag_len, diag_len, 2 * diag_len)

accumulator = np.zeros((2 * diag_len, len(thetas)), dtype=np.uint64)

y_idxs, x_idxs = np.nonzero(edges)  # edge (nonzero) pixel indices

for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]
    for t_idx in range(len(thetas)):
        theta = thetas[t_idx]
        rho = int(round(x * np.cos(theta) + y * np.sin(theta))) + diag_len
        accumulator[rho, t_idx] += 1


# Find indices of peaks in accumulator for detected lines:
lines_idxs = np.argwhere(accumulator > 100)   # choose threshold value experimentally

# Save (rho, theta)
lines = []
for line_idx in lines_idxs:
    rho_idx, theta_idx = line_idx
    rho = rho_idx - diag_len
    theta = thetas[theta_idx]
    lines.append((rho, theta))


for rho, theta in lines:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Detected Lines')
plt.show()

#plt.imshow(accumulator, cmap='hot')
#plt.title('Hough Accumulator')
#plt.xlabel('Theta')
#plt.ylabel('Rho')
#plt.show()



