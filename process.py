import cv2
import numpy as np


class ImageProcessor:
    """
    Class for processing images from camera feed
    """
    frame = None  # BGR numpy array
    def __init__(self, frame=None):
        """Initialize image processor"""
        self.frame = frame
        pass
    
    def process_frame(self, bgr_img):
        """
        Process a single frame
        
        Args:
            bgr_img: Input image in BGR format (numpy array)
            
        Returns:
            Processed image (numpy array)
        """
        # TODO: Implement image processing logic
        pass
    
    
        h, w = bgr_img.shape[:2]
        side = int(min(h, w) * 0.5)
        cx, cy = w // 2, h // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        crop = bgr_img[y0:y0+side, x0:x0+side].copy()
        # for demo: draw a red rectangle around crop in original (optional)
        # cv2.rectangle(bgr_img, (x0,y0), (x0+side-1, y0+side-1), (0,0,255), 2)
        # return cropped image resized to 256x256
        return cv2.resize(crop, (256, 256))
    def preprocess(self, bgr_img):
        """
        Preprocess image (e.g., resize, normalize)
        
        Args:
            bgr_img: Input image in BGR format
            
        Returns:
            Preprocessed image
        """
        # TODO: Implement preprocessing
        pass
    
    def postprocess(self, result):
        """
        Postprocess results
        
        Args:
            result: Processed result
            
        Returns:
            Postprocessed result
        """
        # TODO: Implement postprocessing
        pass
