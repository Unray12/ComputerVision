import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load & Edge Detection
img = cv2.imread('circle.png')  # your image path
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

plt.figure(figsize=(10,6))
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')
plt.show()

# 2. Edge Points
edge_pts = np.column_stack(np.nonzero(edges))  # (y, x)

# 3. Parameter space
y_range = np.arange(70, 190, 4)   # Change depending on likely center, step=4
x_range = np.arange(320, 490, 4)  # ... tune these for your image
a_vals = np.arange(10, 200, 2)     # semi-major axis
b_vals = np.arange(10, 200, 2)     # semi-minor axis
tol = 0.25                        # How close to 1 in the equation counts as inside

acc = np.zeros((len(y_range), len(x_range), len(a_vals), len(b_vals)), dtype=np.uint16)

# 4. Voting
# Vectorized edge-points handling
yy, xx = edge_pts[:,0], edge_pts[:,1]
for iy, yc in enumerate(y_range):
    for ix, xc in enumerate(x_range):
        for ia, a in enumerate(a_vals):
            for ib, b in enumerate(b_vals):
                # Ellipse eq: ((x-xc)/a)**2 + ((y-yc)/b)**2 == 1
                val = ((xx-xc)/a)**2 + ((yy-yc)/b)**2
                acc[iy, ix, ia, ib] += np.sum(np.abs(val - 1) < tol)

# 5. Find best ellipse(s)
ind = np.unravel_index(np.argmax(acc), acc.shape)
best_yc, best_xc = y_range[ind[0]], x_range[ind[1]]
best_a, best_b = a_vals[ind[2]], b_vals[ind[3]]
votes = acc[ind]
print(f'Best ellipse: xc={best_xc}, yc={best_yc}, a={best_a}, b={best_b}, votes={votes}')

# 6. Visualization
output_img = img.copy()
cv2.ellipse(output_img, (best_xc, best_yc), (best_a, best_b), 0, 0, 360, (0, 0, 255), 3)

plt.figure(figsize=(10,6))
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.title(
    f'Detected Ellipse: center=({best_xc},{best_yc}), axes=({best_a},{best_b})'
)
plt.axis('off')
plt.show()