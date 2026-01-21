import time
import cv2
import numpy as np
import os
from Week1_Capturering.Week1_captureSaveImg import CaptureSaveImgProcessor
from Week2_Filtering.Week2_Ex1_Grayscale import GrayscaleProcessor


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
    # STEP 10: SYSTEM INTEGRATION (Week 14)
    # Topic: All Course Concepts
    # =============================================================================
    
    def process_frame(self, bgr_img):


        if bgr_img is None:
            raise ValueError("Input frame is None")
        
        start_time = time.perf_counter()
        results = {}
        

        
        ###################### WRITE YOUR PROCESS PIPELINE HERE #########################
        saveImg = CaptureSaveImgProcessor()
        
        step1_image = saveImg.capture_and_save_image(bgr_img, "test_capture.bmp") ## Step 1: Capture and Save Image
        ######################## IMAGE FILTERING ########################################
        ## Step 2: Convert to Grayscale
        grayScaleProcessor = GrayscaleProcessor()
        processed_img = grayScaleProcessor.convert_to_grayscale(bgr_img)
        ## Save Processed Image
        step3_image = saveImg.capture_and_save_image(processed_img, "processed_capture.bmp")
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

        pass
