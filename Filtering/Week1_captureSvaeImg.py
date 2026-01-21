import cv2
import os
import numpy as np
class CaptureSaveImgProcessor:
    def __init__(self):
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
    