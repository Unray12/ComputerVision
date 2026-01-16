import cv2
class featureMatchingProcessor:
    def __init__(self):
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
    