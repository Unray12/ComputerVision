import cv2
class GaussianProcessor:
    def __init__(self):
        pass
    
    # =============================================================================
    # STEP 1: BASIC IMAGE CAPTURE (Weeks 1-2)
    # Topic: Introduction to Computer Vision, Images as Functions & Filtering
    # =============================================================================
    
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
    