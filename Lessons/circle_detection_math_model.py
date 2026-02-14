import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 1. Load the image and get edges ===
img = cv2.imread('circle.png')  # <-- your image path here
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use Canny for edges detection
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

#plt.figure(figsize=(10,6))
#plt.imshow(edges, cmap='gray')
#plt.title('Canny Edges')
#plt.axis('off')
#plt.show()

# === 2. Gather edge pixel coordinates ===
edge_points = np.column_stack(np.nonzero(edges))  # shape (N, 2), (y, x)

# === 3. Parameter search space for Hough Circle ===
# Tune these parameters depending on expected ellipse/circle size.
r_min, r_max = 80, 130  # min and max radii, experiment for your image
radii = np.arange(r_min, r_max, 2)
accumulator = np.zeros((gray.shape[0], gray.shape[1], len(radii)), dtype=np.uint16)

theta = np.linspace(0, 2*np.pi, 90, endpoint=False)
cos_t = np.cos(theta)
sin_t = np.sin(theta)

# === 4. Hough voting (vectorized for each radius) ===
for y, x in edge_points:
    for ir, r in enumerate(radii):
        xc = (x - r * cos_t).astype(int)
        yc = (y - r * sin_t).astype(int)
        mask = (xc >= 0) & (xc < gray.shape[1]) & (yc >= 0) & (yc < gray.shape[0])
        accumulator[yc[mask], xc[mask], ir] += 1

# === 5. Locate the best circle in the accumulator ===
yc_peak, xc_peak, r_peak = np.unravel_index(np.argmax(accumulator), accumulator.shape)
best_radius = radii[r_peak]

print(f"Best circle center: ({xc_peak}, {yc_peak}), radius: {best_radius}")

# === 6. Visualization ===
output_img = img.copy()
cv2.circle(output_img, (xc_peak, yc_peak), best_radius, (0, 0, 255), 3)  # Red circle

plt.figure(figsize=(10,6))
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.title(f'Detected Circle: Center=({xc_peak},{yc_peak}), Radius={best_radius}')
plt.axis('off')
plt.show()