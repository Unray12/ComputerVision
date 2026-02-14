import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def draw_lines(img, lines, color=(0,255,0), thickness=5):
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def average_slope_intercept(lines, img_shape):
    left_lines = []
    right_lines = []
    left_weights = []
    right_weights = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Fit line to points: y = m*x + b
            if x2 - x1 == 0:
                continue  # avoid division by zero
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt((y2-y1)**2 + (x2-x1)**2)
            # Filter out non-lane lines by slope
            if slope < -0.5:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            elif slope > 0.5:
                right_lines.append((slope, intercept))
                right_weights.append(length)
    # Average using weighted mean (by length)
    lane_lines = []
    y1 = img_shape[0]  # bottom of the image
    y2 = int(y1 * 0.6)  # a bit higher (top of lanes)
    if left_lines:
        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights)
        left_slope, left_intercept = left_lane
        x1 = int((y1 - left_intercept) / left_slope)
        x2 = int((y2 - left_intercept) / left_slope)
        lane_lines.append([x1, y1, x2, y2])
    if right_lines:
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights)
        right_slope, right_intercept = right_lane
        x1 = int((y1 - right_intercept) / right_slope)
        x2 = int((y2 - right_intercept) / right_slope)
        lane_lines.append([x1, y1, x2, y2])
    return lane_lines

# 1. Read Image
img = cv2.imread('frame_20260212_200107_097.jpg')  # <-- replace with your own image path
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# 2. Edge Detection
edges = cv2.Canny(blur, 50, 150)

# 3. Mask the region of interest (focus on the bottom half)
height, width = img.shape[:2]
roi_vertices = np.array([
    [(int(0.1*width), height),
     (int(0.45*width), int(0.6*height)),
     (int(0.55*width), int(0.6*height)),
     (int(0.9*width), height)]
], dtype=np.int32)
masked_edges = region_of_interest(edges, roi_vertices[0])

# 4. Hough Line detection
lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=120)

# 5. Post-process lines to get robust lane lines
lane_lines = []
if lines is not None:
    lane_lines = average_slope_intercept(lines, img.shape)


line_img = np.zeros_like(img)
draw_lines(line_img, lane_lines, color=(0,255,0), thickness=10)

# Overlay detected lanes on the original image
result = cv2.addWeighted(img, 0.8, line_img, 1.0, 0)
center_point = None
if len(lane_lines) == 2:
    x1_l, y1_l, _, _ = lane_lines[0]
    x1_r, y1_r, _, _ = lane_lines[1]
    center_x = int((x1_l + x1_r) / 2)
    center_y = int((y1_l + y1_r) / 2)  # usually bottom of image (~height)
    center_point = (center_x, center_y)
    cv2.circle(result, center_point, 12, (255, 0, 0), -1)


# 6. Overlay detected lanes on the original image
result = cv2.addWeighted(img, 0.8, line_img, 1.0, 0)

# 7. Show results
plt.figure(figsize=(14,8))
plt.subplot(1,2,1)
plt.title("Edges (ROI)")
plt.imshow(masked_edges, cmap='gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.title("Detected Lanes")
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

if center_point is not None:
    plt.plot(center_point[0], center_point[1], 'bo', markersize=16, markeredgecolor="white")
plt.axis('off')
plt.tight_layout()
plt.show()