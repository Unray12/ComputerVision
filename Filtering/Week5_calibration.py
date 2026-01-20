import cv2
class CalibrationProcessor:
    def __init__(self):
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
    