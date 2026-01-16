import cv2
class MotionDetectionProcessor:
    def __init__(self):
        pass
    
    # =============================================================================
    # STEP 6: MOTION DETECTION (Weeks 9-10)
    # Topic: Motion Estimation (Dense Flow, LK)
    # =============================================================================
    
    def detect_motion_frame_diff(self, current_frame, threshold=25):
        """
        Detect motion using frame differencing
        
        Args:
            current_frame: Current frame (BGR)
            threshold: Threshold for motion detection
            
        Returns:
            Motion mask (binary image)
        """
        # TODO: Implement frame differencing motion detection
        # Sinh viên cần:
        # 1. So sánh current_frame với self.previous_frame
        # 2. Tính absolute difference
        # 3. Threshold để tạo binary mask
        # 4. Cập nhật self.previous_frame
        pass
    
    def compute_optical_flow_lk(self, prev_gray, curr_gray, prev_points):
        """
        Compute optical flow using Lucas-Kanade method
        
        Args:
            prev_gray: Previous frame (grayscale)
            curr_gray: Current frame (grayscale)
            prev_points: Points to track from previous frame
            
        Returns:
            tuple: (new_points, status, error)
        """
        # TODO: Implement Lucas-Kanade optical flow
        # Sinh viên cần: Sử dụng cv2.calcOpticalFlowPyrLK
        pass
    
    def compute_optical_flow_farneback(self, prev_gray, curr_gray):
        """
        Compute dense optical flow using Farneback method
        
        Args:
            prev_gray: Previous frame (grayscale)
            curr_gray: Current frame (grayscale)
            
        Returns:
            Flow field (2-channel array)
        """
        # TODO: Implement Farneback dense optical flow
        # Sinh viên cần: Sử dụng cv2.calcOpticalFlowFarneback
        pass
    