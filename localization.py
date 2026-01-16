import cv2


class localizationProcessor:
    def __init__(self):
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
    

