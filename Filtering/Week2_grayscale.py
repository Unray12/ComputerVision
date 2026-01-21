import cv2


class GrayscaleProcessor:
    def __init__(self):
        pass
    
    # =============================================================================
    # STEP 1: BASIC IMAGE CAPTURE (Weeks 1-2)
    # Topic: Introduction to Computer Vision, Images as Functions & Filtering
    # =============================================================================
    
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
    

