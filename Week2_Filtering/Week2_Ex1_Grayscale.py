import cv2


class GrayscaleProcessor:
    def __init__(self):
        pass
    
    # =============================================================================
    # STEP 1: BASIC IMAGE CAPTURE (Weeks 1-2)
    # Topic: Introduction to Computer Vision, Images as Functions & Filtering
    # =============================================================================
    
    def convert_to_grayscale(self, bgr_img):

        # TODO: Implement grayscale conversion
        if bgr_img is None:
            return None

        # Convert BMP to Grayscale
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        return gray_img
    

