import time
import cv2
import numpy as np
import os

from Filtering.Week1_grayscale import GrayscaleProcessor


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
        grayScaleProcessor = GrayscaleProcessor()
        if bgr_img is None:
            raise ValueError("Input frame is None")
        
        start_time = time.perf_counter()
        results = {}
        
        # TODO: Implement complete pipeline
        # Sinh viên cần:
        # 1. gọi các phương thức tương ứng
        # 2. Lưu kết quả vào results dict
        # 3. Visualize kết quả lên processed_img
        # 4. Trả về (processed_img, results, process_time_ms)
        
        ###################### WRITE YOUR PROCESS PIPELINE HERE #########################
        
        
        step1_image = self.capture_and_save_image(bgr_img, "test_capture.bmp") ## Step 1: Capture and Save Image
        
        processed_img = grayScaleProcessor.convert_to_grayscale(bgr_img)  ## Step 2: Convert to Grayscale
        
        step3_image = self.capture_and_save_image(processed_img, "processed_capture.bmp") ## Step 3: Capture and Save Processed Image
        
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
