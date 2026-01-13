# HƯỚNG DẪN SINH VIÊN - COMPUTER VISION CAMERA PROJECT

## Tổng quan dự án

Dự án này xây dựng hệ thống camera giám sát bãi đỗ xe với khả năng:
- Kết nối 2 camera (RTSP/HTTP/USB)
- Xử lý ảnh theo từng bước (preprocessing, segmentation, tracking, OCR biển số xe)
- Hiển thị kết quả real-time trên web interface

## Cấu trúc code

```
├── app.py              # Flask web server (đã hoàn thiện)
├── camera.py           # Video camera handler (đã hoàn thiện)
├── process.py          # Image processing (SINH VIÊN CẦN HOÀN THIỆN)
├── templates/
│   └── index.html      # Web interface (đã hoàn thiện)
└── static/
    ├── main.js         # Frontend JavaScript (đã hoàn thiện)
    └── style.css       # Styling (đã hoàn thiện)
```

## Nhiệm vụ của sinh viên

**Sinh viên cần hoàn thiện các phương thức trong file `process.py`** theo từng bước trong `ProjectProgress.txt`

---

## HƯỚNG DẪN TỪNG BƯỚC

### STEP 1: Basic Image Capture (Weeks 1-2)

#### Phương thức: `capture_and_save_image()`

**Yêu cầu:** Lưu ảnh vào thư mục `CapturedImage/`

**Gợi ý:**
```python
def capture_and_save_image(self, bgr_img, filename):
    if bgr_img is None:
        return False
    
    # Tạo thư mục nếu chưa có
    import os
    os.makedirs('CapturedImage', exist_ok=True)
    
    # Lưu ảnh
    filepath = os.path.join('CapturedImage', filename)
    return cv2.imwrite(filepath, bgr_img)
```

**Test:** Kiểm tra xem file ảnh có được tạo trong thư mục `CapturedImage/` không

---

### STEP 2: Image Preprocessing (Week 3)

#### Phương thức cần hoàn thiện:
1. `convert_to_grayscale()`
2. `apply_gaussian_filter()`
3. `detect_edges_canny()`
4. `preprocess_image()`

**Yêu cầu:** Preprocessing pipeline với grayscale → Gaussian blur → Canny edge

**Gợi ý cho `convert_to_grayscale()`:**
```python
def convert_to_grayscale(self, bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
```

**Gợi ý cho `apply_gaussian_filter()`:**
```python
def apply_gaussian_filter(self, img, kernel_size=(5, 5), sigma=1.0):
    return cv2.GaussianBlur(img, kernel_size, sigma)
```

**Gợi ý cho `detect_edges_canny()`:**
```python
def detect_edges_canny(self, img, threshold1=50, threshold2=150):
    return cv2.Canny(img, threshold1, threshold2)
```

**Gợi ý cho `preprocess_image()`:**
```python
def preprocess_image(self, bgr_img):
    gray = self.convert_to_grayscale(bgr_img)
    filtered = self.apply_gaussian_filter(gray)
    edges = self.detect_edges_canny(filtered)
    
    return {
        'grayscale': gray,
        'filtered': filtered,
        'edges': edges
    }
```

**Test:** Bấm Capture và kiểm tra kết quả edges có hiển thị đúng không

---

### STEP 3: Color Space Conversion & Segmentation (Week 4)

#### Phương thức cần hoàn thiện:
1. `convert_to_hsv()`
2. `segment_by_color()`
3. `apply_morphology()`

**Yêu cầu:** Chuyển đổi HSV và segmentation theo màu sắc

**Gợi ý cho `convert_to_hsv()`:**
```python
def convert_to_hsv(self, bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
```

**Gợi ý cho `segment_by_color()`:**
```python
def segment_by_color(self, bgr_img, lower_bound, upper_bound):
    hsv = self.convert_to_hsv(bgr_img)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    return mask
```

**Ví dụ sử dụng:** Segment xe màu đỏ
```python
# Trong process_frame(), nếu step == 'segment':
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask = self.segment_by_color(bgr_img, lower_red, upper_red)
```

**Gợi ý cho `apply_morphology()`:**
```python
def apply_morphology(self, binary_img, operation='close', kernel_size=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    if operation == 'erode':
        return cv2.erode(binary_img, kernel)
    elif operation == 'dilate':
        return cv2.dilate(binary_img, kernel)
    elif operation == 'open':
        return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    return binary_img
```

**Test:** Segment một màu cụ thể và xem kết quả mask

---

### STEP 4: Homography and Calibration (Week 5)

#### Phương thức cần hoàn thiện:
1. `calibrate_camera()`
2. `undistort_image()`
3. `compute_homography()`
4. `apply_perspective_transform()`

**Yêu cầu:** Calibrate camera và correct perspective

**Gợi ý cho `calibrate_camera()`:**
```python
def calibrate_camera(self, calibration_images, pattern_size=(9, 6)):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    
    objpoints = []  # 3D points
    imgpoints = []  # 2D points
    
    for img in calibration_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    self.camera_matrix = mtx
    self.dist_coeffs = dist
    
    return mtx, dist, rvecs, tvecs
```

**Gợi ý cho `undistort_image()`:**
```python
def undistort_image(self, bgr_img):
    if self.camera_matrix is None or self.dist_coeffs is None:
        raise ValueError("Camera not calibrated yet")
    
    return cv2.undistort(bgr_img, self.camera_matrix, self.dist_coeffs)
```

**Test:** Chụp ảnh checkerboard, calibrate và so sánh trước/sau undistort

---

### STEP 5: Region of Interest Detection (Weeks 6-7)

#### Phương thức cần hoàn thiện:
1. `detect_corners_harris()`
2. `detect_features_orb()`
3. `detect_features_sift()`
4. `crop_vehicle_roi()`

**Yêu cầu:** Feature detection để tìm vùng xe

**Gợi ý cho `detect_features_orb()`:**
```python
def detect_features_orb(self, img, n_features=500):
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors
```

**Gợi ý cho `crop_vehicle_roi()`:**
```python
def crop_vehicle_roi(self, bgr_img, roi_coords=None):
    if roi_coords is None:
        # Auto-detect ROI using edge detection
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            roi_coords = (x, y, w, h)
        else:
            # Default to center crop
            h, w = bgr_img.shape[:2]
            roi_coords = (w//4, h//4, w//2, h//2)
    
    x, y, w, h = roi_coords
    return bgr_img[y:y+h, x:x+w].copy()
```

**Test:** Capture ảnh có xe và kiểm tra ROI có crop đúng vùng xe không

---

### STEP 6: Motion Detection (Weeks 9-10)

#### Phương thức cần hoàn thiện:
1. `detect_motion_frame_diff()`
2. `compute_optical_flow_lk()`
3. `compute_optical_flow_farneback()`

**Yêu cầu:** Detect chuyển động qua frame differencing và optical flow

**Gợi ý cho `detect_motion_frame_diff()`:**
```python
def detect_motion_frame_diff(self, current_frame, threshold=25):
    if self.previous_frame is None:
        self.previous_frame = current_frame.copy()
        return None
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Threshold
    _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Update previous frame
    self.previous_frame = current_frame.copy()
    
    return motion_mask
```

**Gợi ý cho `compute_optical_flow_lk()`:**
```python
def compute_optical_flow_lk(self, prev_gray, curr_gray, prev_points):
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_points, None, **lk_params
    )
    
    return next_points, status, error
```

**Test:** Bấm Capture nhiều lần liên tiếp và quan sát motion mask

---

### STEP 7: Object Tracking (Weeks 11-12)

#### Phương thức cần hoàn thiện:
1. `initialize_kalman_filter()`
2. `update_kalman_filter()`
3. `track_objects()`

**Yêu cầu:** Track xe sử dụng Kalman filter

**Gợi ý cho `initialize_kalman_filter()`:**
```python
def initialize_kalman_filter(self):
    kalman = cv2.KalmanFilter(4, 2)  # 4 states (x, y, dx, dy), 2 measurements (x, y)
    
    # Transition matrix
    kalman.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], np.float32)
    
    # Measurement matrix
    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], np.float32)
    
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
    
    return kalman
```

**Gợi ý cho `update_kalman_filter()`:**
```python
def update_kalman_filter(self, kalman, measurement):
    # Correct with measurement
    kalman.correct(np.array([[np.float32(measurement[0])], 
                              [np.float32(measurement[1])]]))
    
    # Predict next state
    prediction = kalman.predict()
    
    return (int(prediction[0]), int(prediction[1]))
```

**Test:** Track một object đơn giản qua nhiều frames

---

### STEP 8: License Plate Localization (Weeks 6-8, 13-14)

#### Phương thức cần hoàn thiện:
1. `locate_license_plate()`
2. `enhance_plate_image()`

**Yêu cầu:** Tìm vị trí biển số xe

**Gợi ý cho `locate_license_plate()`:**
```python
def locate_license_plate(self, vehicle_img):
    # Preprocessing
    gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blur, 50, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by aspect ratio (license plates are wider than tall)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # Typical license plate aspect ratio: 2-5
        if 2.0 <= aspect_ratio <= 5.0 and w > 50 and h > 15:
            # Found potential plate
            plate = vehicle_img[y:y+h, x:x+w].copy()
            return plate
    
    return None
```

**Gợi ý cho `enhance_plate_image()`:**
```python
def enhance_plate_image(self, plate_img):
    # Resize to standard size
    plate = cv2.resize(plate_img, (300, 100))
    
    # Convert to grayscale
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned
```

**Test:** Chụp ảnh xe có biển số rõ ràng và kiểm tra có detect được không

---

### STEP 9: License Plate Character Recognition (Weeks 13-14)

#### Phương thức cần hoàn thiện:
1. `recognize_plate_text()`
2. `segment_characters()`
3. `train_character_classifier()`
4. `classify_character()`

**Yêu cầu:** OCR biển số xe

**Cài đặt pytesseract:**
```bash
pip install pytesseract
# Download Tesseract từ: https://github.com/UB-Mannheim/tesseract/wiki
```

**Gợi ý cho `recognize_plate_text()`:**
```python
def recognize_plate_text(self, plate_img):
    import pytesseract
    
    # Config for license plate (only alphanumeric)
    config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    text = pytesseract.image_to_string(plate_img, config=config)
    
    # Post-process: remove special characters and spaces
    text = ''.join(filter(str.isalnum, text))
    
    return text
```

**Gợi ý cho `segment_characters()`:**
```python
def segment_characters(self, plate_img):
    # Find contours of characters
    contours, _ = cv2.findContours(plate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    characters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter noise (too small)
        if w > 5 and h > 15:
            char_img = plate_img[y:y+h, x:x+w]
            characters.append(char_img)
    
    return characters
```

**Test:** Nhận dạng biển số và hiển thị text lên màn hình

---

### STEP 10: System Integration (Week 14)

#### Phương thức cần hoàn thiện:
1. `process_frame()` - Pipeline hoàn chỉnh
2. `visualize_results()` - Vẽ kết quả lên ảnh

**Yêu cầu:** Tích hợp tất cả các bước

**Gợi ý cho `process_frame()` - hoàn chỉnh:**
```python
def process_frame(self, bgr_img, step='all'):
    if bgr_img is None:
        raise ValueError("Input frame is None")
    
    start_time = time.perf_counter()
    results = {}
    processed_img = bgr_img.copy()
    
    if step == 'preprocess' or step == 'all':
        # Step 2: Preprocessing
        preprocess_results = self.preprocess_image(bgr_img)
        results['preprocess'] = preprocess_results
        
        # Visualize edges
        edges_color = cv2.cvtColor(preprocess_results['edges'], cv2.COLOR_GRAY2BGR)
        processed_img = edges_color
    
    elif step == 'segment':
        # Step 3: Segmentation
        lower = np.array([0, 100, 100])
        upper = np.array([10, 255, 255])
        mask = self.segment_by_color(bgr_img, lower, upper)
        
        # Apply to original image
        segmented = cv2.bitwise_and(bgr_img, bgr_img, mask=mask)
        results['segmentation'] = {'mask': mask}
        processed_img = segmented
    
    elif step == 'roi':
        # Step 5: ROI detection
        cropped = self.crop_vehicle_roi(bgr_img)
        results['roi'] = {'cropped': cropped}
        processed_img = cropped
    
    elif step == 'motion':
        # Step 6: Motion detection
        motion_mask = self.detect_motion_frame_diff(bgr_img)
        if motion_mask is not None:
            motion_color = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            processed_img = motion_color
            results['motion'] = {'has_motion': np.sum(motion_mask) > 1000}
    
    elif step == 'license_plate':
        # Steps 8-9: License plate
        vehicle_roi = self.crop_vehicle_roi(bgr_img)
        plate = self.locate_license_plate(vehicle_roi)
        
        if plate is not None:
            enhanced = self.enhance_plate_image(plate)
            text = self.recognize_plate_text(enhanced)
            
            results['license_plate'] = {
                'plate_image': plate,
                'text': text
            }
            processed_img = enhanced
        else:
            results['license_plate'] = {'text': 'NOT FOUND'}
    
    process_time_ms = (time.perf_counter() - start_time) * 1000
    
    return processed_img, results, process_time_ms
```

**Gợi ý cho `visualize_results()`:**
```python
def visualize_results(self, bgr_img, results):
    annotated = bgr_img.copy()
    
    # Draw license plate text if available
    if 'license_plate' in results and 'text' in results['license_plate']:
        text = results['license_plate']['text']
        cv2.putText(annotated, f"Plate: {text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw tracking boxes if available
    if 'tracked_objects' in results:
        for obj in results['tracked_objects']:
            x, y, w, h = obj['bbox']
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(annotated, f"ID: {obj['id']}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated
```

---

## CÁCH TEST HỆ THỐNG

### 1. Test cơ bản:
```bash
# Chạy Flask app
python app.py

# Mở browser: http://localhost:5000
# Connect camera và bấm Capture
```

### 2. Test từng bước:
- **Step 2 (Preprocess):** Bấm Capture → Xem edges trong "Fragment (processed)"
- **Step 3 (Segment):** Thay đổi màu trong code → Bấm Capture → Xem mask
- **Step 5 (ROI):** Bấm Capture → Xem vùng xe được crop
- **Step 8-9 (License Plate):** Chụp xe có biển số → Xem OCR result

### 3. Modify frontend để chọn step:
Có thể thêm dropdown trong `index.html` để chọn step:
```html
<select id="step-1">
  <option value="all">All</option>
  <option value="preprocess">Preprocess</option>
  <option value="segment">Segment</option>
  <option value="roi">ROI</option>
  <option value="motion">Motion</option>
  <option value="license_plate">License Plate</option>
</select>
```

Và update `main.js`:
```javascript
async function capture(cam_id){
  const step = document.getElementById(`step-${cam_id}`).value;
  const res = await fetch('/capture', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({cam_id: cam_id, step: step})
  });
  // ... rest of code
}
```

---

## TIPS & BEST PRACTICES

1. **Debug từng bước:** Không nên implement hết tất cả một lúc. Test từng method một.

2. **Visualize intermediate results:** Luôn hiển thị kết quả trung gian để debug dễ hơn.

3. **Handle errors:** Thêm try-except để catch lỗi và log ra.

4. **Tune parameters:** Các threshold, kernel size cần tune theo từng camera/lighting.

5. **Save results:** Lưu ảnh kết quả vào thư mục để so sánh.

6. **Document code:** Thêm comments giải thích logic.

---

## TÀI LIỆU THAM KHẢO

- OpenCV Documentation: https://docs.opencv.org/
- OpenCV Python Tutorials: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- Flask Documentation: https://flask.palletsprojects.com/

---

## ĐÁNH GIÁ CUỐI KỲ

Sinh viên cần:
1. **Demo hệ thống hoàn chỉnh** (70%):
   - Kết nối 2 camera
   - Xử lý ảnh real-time
   - Nhận dạng biển số xe

2. **Báo cáo kỹ thuật** (20%):
   - Mô tả kiến trúc hệ thống
   - Giải thích thuật toán
   - Kết quả thử nghiệm

3. **Code quality** (10%):
   - Clean code
   - Comments đầy đủ
   - OOP principles

**Chúc các bạn thành công!** 🚀
