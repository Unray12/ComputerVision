import cv2
import numpy as np

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
    if lines is None:
        return []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt((y2-y1)**2 + (x2-x1)**2)
            if slope < -0.5:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            elif slope > 0.5:
                right_lines.append((slope, intercept))
                right_weights.append(length)
    lane_lines = []
    y1 = img_shape[0]
    y2 = int(y1 * 0.6)
    if left_lines:
        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights)
        if left_lane is not None:
            left_slope, left_intercept = left_lane
            x1 = int((y1 - left_intercept) / left_slope)
            x2 = int((y2 - left_intercept) / left_slope)
            lane_lines.append([x1, y1, x2, y2])
    if right_lines:
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights)
        if right_lane is not None:
            right_slope, right_intercept = right_lane
            x1 = int((y1 - right_intercept) / right_slope)
            x2 = int((y2 - right_intercept) / right_slope)
            lane_lines.append([x1, y1, x2, y2])
    return lane_lines

def get_direction(center_pt, img_shape, threshold=20):
    img_center_x = img_shape[1] // 2
    if center_pt is None:
        return 'NO LINE'
    cx = center_pt[0]
    if cx < img_center_x - threshold:
        return 'LEFT'
    elif cx > img_center_x + threshold:
        return 'RIGHT'
    else:
        return 'STRAIGHT'

def process_frame(frame):
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 60, 150)
    height, width = img.shape[:2]

    roi_vertices = np.array([
        [(int(0.1*width), height),
         (int(0.45*width), int(0.6*height)),
         (int(0.55*width), int(0.6*height)),
         (int(0.9*width), height)]
    ], dtype=np.int32)
    masked_edges = region_of_interest(edges, roi_vertices[0])
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=120)

    lane_lines = average_slope_intercept(lines, img.shape)
    line_img = np.zeros_like(img)
    draw_lines(line_img, lane_lines, (0,255,0), 10)
    result = cv2.addWeighted(img, 0.8, line_img, 1.0, 0)
    center_point = None
    if len(lane_lines) == 2:
        x1_l, y1_l, _, _ = lane_lines[0]
        x1_r, y1_r, _, _ = lane_lines[1]
        center_x = int((x1_l + x1_r) / 2)
        center_y = int((y1_l + y1_r) / 2)
        center_point = (center_x, center_y)
        cv2.circle(result, center_point, 12, (255, 0, 0), -1)
        cv2.line(result, (width//2, height), center_point, (0,0,255), 3)  # Draw red line from bottom center to lane center
    # Draw image center
    cv2.circle(result, (width//2, height), 8, (0,255,255), -1)
    # Draw ROI
    cv2.polylines(result, [roi_vertices], isClosed=True, color=(200,200,0), thickness=3)

    # Display direction suggestion
    direction = get_direction(center_point, img.shape)
    cv2.putText(result, "Direction: " + direction, (40,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,50,50), 5)
    return result

cap = cv2.VideoCapture(0)  # 0=default webcam; or replace with your video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (960, 540))
    out_frame = process_frame(frame)
    cv2.imshow('Lane detection and robot direction', out_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()