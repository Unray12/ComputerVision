import cv2
class ObjectTrackingProcessor:
    def __init__(self):
        pass
    
    # =============================================================================
    # STEP 7: OBJECT TRACKING (Weeks 11-12)
    # Topic: Tracking (Kalman, Particle, Bayes Filters)
    # =============================================================================
    
    def initialize_kalman_filter(self):
        """
        Initialize Kalman filter for object tracking
        
        Returns:
            cv2.KalmanFilter object
        """
        # TODO: Implement Kalman filter initialization
        # Sinh viên cần:
        # 1. Tạo KalmanFilter với cv2.KalmanFilter(4, 2)
        # 2. Thiết lập transition matrix, measurement matrix, etc.
        pass
    
    def update_kalman_filter(self, kalman, measurement):
        """
        Update Kalman filter with new measurement
        
        Args:
            kalman: KalmanFilter object
            measurement: Measured position (x, y)
            
        Returns:
            Predicted position
        """
        # TODO: Implement Kalman filter update
        # Sinh viên cần:
        # 1. Gọi kalman.correct() với measurement
        # 2. Gọi kalman.predict()
        pass
    
    def track_objects(self, bgr_img, detections):
        """
        Track multiple objects across frames
        
        Args:
            bgr_img: Current frame
            detections: List of detected object bounding boxes [(x,y,w,h), ...]
            
        Returns:
            List of tracked objects with IDs
        """
        # TODO: Implement multi-object tracking
        # Sinh viên cần:
        # 1. Match detections với tracked_objects
        # 2. Update Kalman filters
        # 3. Handle occlusions
        # 4. Cập nhật self.tracked_objects
        pass