import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('lane.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 100, 200)

rows, cols = edges.shape


min_radius = 20
max_radius = 100
radii = np.arange(min_radius, max_radius, 1)  # 1st parameter

# Tạo bộ tích lũy Hough 3 chiều: (a, b, r)
accumulator = np.zeros((rows, cols, len(radii)), dtype=np.uint64)

# Tìm các pixel biên
y_idxs, x_idxs = np.nonzero(edges)  # (y, x) là pixel biên

for idx in range(len(x_idxs)):
    x = x_idxs[idx]
    y = y_idxs[idx]
    for r_idx, radius in enumerate(radii):
        # Quét theo góc từ 0 đến 360 độ, vì trên đường tròn, mỗi điểm có nhiều khả năng thuộc mọi phương vị
        for t in range(0, 360, 5):  # bước 5 độ cho nhanh, muốn mịn hơn thì giảm bước
            theta = np.deg2rad(t)
            # (x, y) nằm trên đường tròn tâm (a, b), bán kính radius
            # => a = x - r cos(theta), b = y - r sin(theta)
            a = int(round(x - radius * np.cos(theta)))
            b = int(round(y - radius * np.sin(theta)))
            if (0 <= a < cols) and (0 <= b < rows):
                accumulator[b, a, r_idx] += 1

# Tìm các giá trị lớn nhất trong bộ tích lũy (có thể threshold hoặc tìm local maximum)
threshold = np.amax(accumulator) * 0.5   # ví dụ: lấy 50% giá trị lớn nhất
circles = np.argwhere(accumulator > threshold)  # (b, a, r_idx)

# Vẽ các đường tròn phát hiện được lên ảnh gốc
output_img = img.copy()
for b, a, r_idx in circles:
    cv2.circle(output_img, (a, b), radii[r_idx], (0, 255, 0), 2)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.title('Detected Circles (Hough Transform)')
plt.axis('off')
plt.show()