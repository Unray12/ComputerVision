import cv2
class ColorSpaceProcessor:
    
    def __init__(self):
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
    