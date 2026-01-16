import time
import cv2
import numpy as np
import os


class ImageProcessor:
    """
    Class for processing images from camera feed
    Implements computer vision techniques from ProjectProgress.txt
    """
    
    def __init__(self):
        """Initialize image processor with calibration parameters"""
        self.camera_matrix = None  # Camera calibration matrix
        self.dist_coeffs = None    # Distortion coefficients
        self.homography_matrix = None  # Homography transformation matrix
        self.previous_frame = None  # For motion detection
        self.tracked_objects = []   # For object tracking
        pass
    
    # =============================================================================
    # STEP 1: BASIC IMAGE CAPTURE (Weeks 1-2)
    # Topic: Introduction to Computer Vision, Images as Functions & Filtering
    # =============================================================================
    
    def capture_and_save_image(self, bgr_img, filename):
        """
        Capture and save static image from camera
        
        Args:
            bgr_img: Input image in BGR format (numpy array)
            filename: Path to save the image
            
        Returns:
            bool: True if successful, False otherwise
        """
        # TODO: Implement image capture and saving
        # Sinh viên cần:
        # 1. Kiểm tra bgr_img có hợp lệ không
        # 2. Lưu ảnh vào thư mục CapturedImage/
        # 3. Trả về True nếu thành công, False nếu thất bại
        pass
    
        try:
            # 1. Kiểm tra ảnh đầu vào có hợp lệ không
            if bgr_img is None or not isinstance(bgr_img, np.ndarray):
                return False

            # 2. Tạo thư mục CapturedImage nếu chưa tồn tại
            save_dir = "CapturedImage"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Ghép đường dẫn lưu ảnh
            save_path = os.path.join(save_dir, filename)

            # 3. Lưu ảnh
            success = cv2.imwrite(save_path, bgr_img)

            return success

        except Exception as e:
            print("Error saving image:", e)
            return False
    
    def convert_to_grayscale(self, bgr_img):
        """
        Convert BGR image to grayscale
        
        Args:
            bgr_img: Input image in BGR format
            
        Returns:
            Grayscale image
        """
        # TODO: Implement grayscale conversion
        # Sinh viên cần: Sử dụng cv2.cvtColor để chuyển sang grayscale
        pass
    
        if bgr_img is None:
            return None

        # Chuyển ảnh từ BMP sang Grayscale
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        return gray_img
    
    
    def apply_gaussian_filter(self, img, kernel_size=(5, 5), sigma=1.0):
        """
        Apply Gaussian filtering to reduce noise
        
        Args:
            img: Input image
            kernel_size: Size of Gaussian kernel (must be odd)
            sigma: Standard deviation
            
        Returns:
            Filtered image
        """
        # TODO: Implement Gaussian filtering
        # Sinh viên cần: Sử dụng cv2.GaussianBlur
        pass
    
        if img is None:
            return None

        # Áp dụng Gaussian Blur
        filtered_img = cv2.GaussianBlur(img, kernel_size, sigma)
        return filtered_img
    # =============================================================================
    # STEP 2: IMAGE PREPROCESSING (Week 3)
    # Topic: Image Operations (Edge Detection, Convolution)
    # =============================================================================

    def detect_edges_canny(self, img, threshold1=50, threshold2=150):
        """
        Detect edges using Canny edge detector
        
        Args:
            img: Input grayscale image
            threshold1: Lower threshold for hysteresis
            threshold2: Upper threshold for hysteresis
            
        Returns:
            Edge map (binary image)
        """
        # TODO: Implement Canny edge detection
        # Sinh viên cần: Sử dụng cv2.Canny
        pass
    
    def preprocess_image(self, bgr_img):
        """
        Complete preprocessing pipeline: grayscale + Gaussian + edge detection
        
        Args:
            bgr_img: Input image in BGR format
            
        Returns:
            dict: Dictionary containing intermediate results
                  {'grayscale': ..., 'filtered': ..., 'edges': ...}
        """
        # TODO: Implement complete preprocessing pipeline
        # Sinh viên cần: 
        # 1. Gọi convert_to_grayscale()
        # 2. Gọi apply_gaussian_filter()
        # 3. Gọi detect_edges_canny()
        # 4. Trả về dictionary chứa tất cả kết quả trung gian
        pass
    
    # =============================================================================
    # STEP 3: COLOR SPACE CONVERSION & SEGMENTATION (Week 4)
    # Topic: Color Spaces, Segmentation, Morphology
    # =============================================================================
    
    def convert_to_hsv(self, bgr_img):
        """
        Convert BGR image to HSV color space
        
        Args:
            bgr_img: Input image in BGR format
            
        Returns:
            Image in HSV color space
        """
        # TODO: Implement HSV conversion
        # Sinh viên cần: Sử dụng cv2.cvtColor với cv2.COLOR_BGR2HSV
        pass
    
    def segment_by_color(self, bgr_img, lower_bound, upper_bound):
        """
        Segment image by color thresholding in HSV space
        
        Args:
            bgr_img: Input image in BGR format
            lower_bound: Lower HSV bound (e.g., np.array([0, 100, 100]))
            upper_bound: Upper HSV bound (e.g., np.array([10, 255, 255]))
            
        Returns:
            Binary mask of segmented regions
        """
        # TODO: Implement color segmentation
        # Sinh viên cần:
        # 1. Chuyển sang HSV
        # 2. Sử dụng cv2.inRange để tạo mask
        pass
    
    def apply_morphology(self, binary_img, operation='close', kernel_size=(5, 5)):
        """
        Apply morphological operations (erosion/dilation)
        
        Args:
            binary_img: Input binary image
            operation: 'erode', 'dilate', 'open', or 'close'
            kernel_size: Size of structuring element
            
        Returns:
            Image after morphological operation
        """
        # TODO: Implement morphological operations
        # Sinh viên cần:
        # 1. Tạo kernel với cv2.getStructuringElement
        # 2. Áp dụng phép toán tương ứng: cv2.erode, cv2.dilate, cv2.morphologyEx
        pass
    
    # =============================================================================
    # STEP 4: HOMOGRAPHY AND CALIBRATION (Week 5)
    # Topic: Camera & Calibration
    # =============================================================================
    
    def calibrate_camera(self, calibration_images, pattern_size=(9, 6)):
        """
        Calibrate camera using checkerboard pattern
        
        Args:
            calibration_images: List of calibration images
            pattern_size: Checkerboard pattern size (columns, rows)
            
        Returns:
            tuple: (camera_matrix, dist_coeffs, rvecs, tvecs)
        """
        # TODO: Implement camera calibration
        # Sinh viên cần:
        # 1. Chuẩn bị object points và image points
        # 2. Tìm góc checkerboard với cv2.findChessboardCorners
        # 3. Sử dụng cv2.calibrateCamera
        # 4. Lưu camera_matrix và dist_coeffs vào self
        pass
    
    def undistort_image(self, bgr_img):
        """
        Correct lens distortion using calibration parameters
        
        Args:
            bgr_img: Input distorted image
            
        Returns:
            Undistorted image
        """
        # TODO: Implement image undistortion
        # Sinh viên cần:
        # 1. Kiểm tra camera_matrix và dist_coeffs đã được tính chưa
        # 2. Sử dụng cv2.undistort
        pass
    
    def compute_homography(self, src_points, dst_points):
        """
        Compute homography matrix from source to destination points
        
        Args:
            src_points: Source points (Nx2 array)
            dst_points: Destination points (Nx2 array)
            
        Returns:
            3x3 homography matrix
        """
        # TODO: Implement homography computation
        # Sinh viên cần:
        # 1. Sử dụng cv2.findHomography
        # 2. Lưu homography_matrix vào self
        pass
    
    def apply_perspective_transform(self, bgr_img, homography_matrix=None):
        """
        Apply perspective transformation to image
        
        Args:
            bgr_img: Input image
            homography_matrix: 3x3 homography matrix (uses self.homography_matrix if None)
            
        Returns:
            Transformed image
        """
        # TODO: Implement perspective transformation
        # Sinh viên cần: Sử dụng cv2.warpPerspective
        pass
    
    # =============================================================================
    # STEP 5: REGION OF INTEREST DETECTION (Weeks 6-7)
    # Topic: Visual Features, Feature Matching
    # =============================================================================
    
    def detect_corners_harris(self, img, block_size=2, ksize=3, k=0.04):
        """
        Detect corners using Harris corner detector
        
        Args:
            img: Input grayscale image
            block_size: Size of neighborhood
            ksize: Aperture parameter for Sobel
            k: Harris detector free parameter
            
        Returns:
            Corner response map
        """
        # TODO: Implement Harris corner detection
        # Sinh viên cần: Sử dụng cv2.cornerHarris
        pass
    
    def detect_features_orb(self, img, n_features=500):
        """
        Detect ORB features (Oriented FAST and Rotated BRIEF)
        
        Args:
            img: Input grayscale image
            n_features: Maximum number of features to detect
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        # TODO: Implement ORB feature detection
        # Sinh viên cần:
        # 1. Tạo ORB detector với cv2.ORB_create
        # 2. Gọi detectAndCompute
        pass
    
    def detect_features_sift(self, img):
        """
        Detect SIFT features (Scale-Invariant Feature Transform)
        Note: SIFT may require opencv-contrib-python
        
        Args:
            img: Input grayscale image
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        # TODO: Implement SIFT feature detection
        # Sinh viên cần:
        # 1. Tạo SIFT detector với cv2.SIFT_create()
        # 2. Gọi detectAndCompute
        pass
    
    def crop_vehicle_roi(self, bgr_img, roi_coords=None):
        """
        Crop region of interest (vehicle area) from image
        
        Args:
            bgr_img: Input image
            roi_coords: Tuple (x, y, w, h) or None for auto-detection
            
        Returns:
            Cropped vehicle image
        """
        # TODO: Implement ROI cropping
        # Sinh viên cần:
        # 1. Nếu roi_coords is None, tự động detect ROI bằng feature detection
        # 2. Crop ảnh theo tọa độ
        pass
    
    # =============================================================================
    # STEP 6: MOTION DETECTION (Weeks 9-10)
    # Topic: Motion Estimation (Dense Flow, LK)
    # =============================================================================
    
    def detect_motion_frame_diff(self, current_frame, threshold=25):
        """
        Detect motion using frame differencing
        
        Args:
            current_frame: Current frame (BGR)
            threshold: Threshold for motion detection
            
        Returns:
            Motion mask (binary image)
        """
        # TODO: Implement frame differencing motion detection
        # Sinh viên cần:
        # 1. So sánh current_frame với self.previous_frame
        # 2. Tính absolute difference
        # 3. Threshold để tạo binary mask
        # 4. Cập nhật self.previous_frame
        pass
    
    def compute_optical_flow_lk(self, prev_gray, curr_gray, prev_points):
        """
        Compute optical flow using Lucas-Kanade method
        
        Args:
            prev_gray: Previous frame (grayscale)
            curr_gray: Current frame (grayscale)
            prev_points: Points to track from previous frame
            
        Returns:
            tuple: (new_points, status, error)
        """
        # TODO: Implement Lucas-Kanade optical flow
        # Sinh viên cần: Sử dụng cv2.calcOpticalFlowPyrLK
        pass
    
    def compute_optical_flow_farneback(self, prev_gray, curr_gray):
        """
        Compute dense optical flow using Farneback method
        
        Args:
            prev_gray: Previous frame (grayscale)
            curr_gray: Current frame (grayscale)
            
        Returns:
            Flow field (2-channel array)
        """
        # TODO: Implement Farneback dense optical flow
        # Sinh viên cần: Sử dụng cv2.calcOpticalFlowFarneback
        pass
    
    # =============================================================================
    # STEP 7: OBJECT TRACKING (Weeks 11-12)
    # Topic: Tracking (Kalman, Particle, Bayes Filters)
    # =============================================================================
    
    def initialize_kalman_filter(self):
        """
        Initialize Kalman filter for object tracking
        
        Returns:
            cv2.KalmanFilter object
        """
        # TODO: Implement Kalman filter initialization
        # Sinh viên cần:
        # 1. Tạo KalmanFilter với cv2.KalmanFilter(4, 2)
        # 2. Thiết lập transition matrix, measurement matrix, etc.
        pass
    
    def update_kalman_filter(self, kalman, measurement):
        """
        Update Kalman filter with new measurement
        
        Args:
            kalman: KalmanFilter object
            measurement: Measured position (x, y)
            
        Returns:
            Predicted position
        """
        # TODO: Implement Kalman filter update
        # Sinh viên cần:
        # 1. Gọi kalman.correct() với measurement
        # 2. Gọi kalman.predict()
        pass
    
    def track_objects(self, bgr_img, detections):
        """
        Track multiple objects across frames
        
        Args:
            bgr_img: Current frame
            detections: List of detected object bounding boxes [(x,y,w,h), ...]
            
        Returns:
            List of tracked objects with IDs
        """
        # TODO: Implement multi-object tracking
        # Sinh viên cần:
        # 1. Match detections với tracked_objects
        # 2. Update Kalman filters
        # 3. Handle occlusions
        # 4. Cập nhật self.tracked_objects
        pass
    
    # =============================================================================
    # STEP 8: LICENSE PLATE LOCALIZATION (Weeks 6-8, 13-14)
    # Topic: Visual Features, Recognition
    # =============================================================================
    
    def locate_license_plate(self, vehicle_img):
        """
        Locate license plate region in vehicle image
        
        Args:
            vehicle_img: Cropped vehicle image (BGR)
            
        Returns:
            Cropped license plate image or None if not found
        """
        # TODO: Implement license plate localization
        # Sinh viên cần:
        # 1. Preprocessing: grayscale, Gaussian blur
        # 2. Edge detection
        # 3. Tìm contours với cv2.findContours
        # 4. Lọc contours theo aspect ratio (width/height ~ 2-5)
        # 5. Crop vùng biển số
        pass
    
    def enhance_plate_image(self, plate_img):
        """
        Enhance license plate image for better OCR
        
        Args:
            plate_img: License plate image
            
        Returns:
            Enhanced plate image
        """
        # TODO: Implement plate enhancement
        # Sinh viên cần:
        # 1. Resize về kích thước chuẩn
        # 2. Adaptive thresholding
        # 3. Morphological operations để làm sạch
        pass
    
    # =============================================================================
    # STEP 9: LICENSE PLATE CHARACTER RECOGNITION (Weeks 13-14)
    # Topic: Recognition (Classification, SVM, PCA, Boosting)
    # =============================================================================
    
    def recognize_plate_text(self, plate_img):
        """
        Recognize text from license plate using OCR
        
        Args:
            plate_img: Enhanced license plate image
            
        Returns:
            Recognized text string
        """
        # TODO: Implement OCR
        # Sinh viên cần:
        # 1. Cài đặt pytesseract
        # 2. Gọi pytesseract.image_to_string() với config phù hợp
        # 3. Post-process text (remove special characters, etc.)
        pass
    
    def segment_characters(self, plate_img):
        """
        Segment individual characters from license plate
        
        Args:
            plate_img: Binary plate image
            
        Returns:
            List of character images
        """
        # TODO: Implement character segmentation
        # Sinh viên cần:
        # 1. Tìm contours của từng ký tự
        # 2. Sắp xếp từ trái sang phải
        # 3. Crop từng character
        pass
    
    def train_character_classifier(self, training_data, labels):
        """
        Train SVM classifier for character recognition
        
        Args:
            training_data: Array of training images
            labels: Corresponding labels
            
        Returns:
            Trained classifier
        """
        # TODO: Implement SVM training
        # Sinh viên cần:
        # 1. Extract features (HOG, PCA, etc.)
        # 2. Train SVM với cv2.ml.SVM_create()
        pass
    
    def classify_character(self, char_img, classifier):
        """
        Classify a single character using trained classifier
        
        Args:
            char_img: Character image
            classifier: Trained classifier
            
        Returns:
            Predicted character
        """
        # TODO: Implement character classification
        # Sinh viên cần:
        # 1. Extract features từ char_img
        # 2. Predict với classifier
        pass
    
    # =============================================================================
    # STEP 10: SYSTEM INTEGRATION (Week 14)
    # Topic: All Course Concepts
    # =============================================================================
    
    def process_frame(self, bgr_img):
        """
        Complete processing pipeline - integrates all steps
        
        Args:
            bgr_img: Input image in BGR format (numpy array)
            step: Which processing step to apply
                  Options: 'preprocess', 'segment', 'motion', 'track', 
                          'license_plate', 'all'
            
        Returns:
            tuple: (Processed image, results dict, process time in ms)
        """
        if bgr_img is None:
            raise ValueError("Input frame is None")
        
        start_time = time.perf_counter()
        results = {}
        processed_img = bgr_img.copy()
        
        # TODO: Implement complete pipeline
        # Sinh viên cần:
        # 1. Dựa vào tham số 'step', gọi các phương thức tương ứng
        # 2. Lưu kết quả vào results dict
        # 3. Visualize kết quả lên processed_img
        # 4. Trả về (processed_img, results, process_time_ms)
        
        ###################### WRITE YOUR PROCESS PIPELINE HERE #########################
        step1_image = self.capture_and_save_image(bgr_img, "test_capture.bmp") ## Step 1: Capture and Save Image
        if step1_image:
            img = cv2.imread("CapturedImage/test_capture.bmp")
        if img is None:
            print("Read image failed")
            return None

        gray = self.convert_to_grayscale(img)  ## Step 2: Convert to Grayscale
        processed_img = self.apply_gaussian_filter(gray) ## Step 3: Apply Gaussian Filter
        
        #################################################################################
        
        process_time_ms = (time.perf_counter() - start_time) * 1000
        return processed_img, results, process_time_ms
    
    def visualize_results(self, bgr_img, results):
        """
        Visualize all processing results on image
        
        Args:
            bgr_img: Original image
            results: Dictionary of results from process_frame
            
        Returns:
            Annotated image
        """
        # TODO: Implement result visualization
        # Sinh viên cần:
        # 1. Vẽ bounding boxes
        # 2. Hiển thị text (biển số, tracking ID, etc.)
        # 3. Vẽ optical flow vectors
        # 4. Highlight detected features
        pass
