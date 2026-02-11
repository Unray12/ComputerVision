import cv2


class MedianBurProcessor:
    def __init__(self):
        pass


    def convert_to_blur(self, bgr_img):
        # TODO: Implement grayscale conversion
        if bgr_img is None:
            return None

        # Convert BMP to Median
        median = cv2.medianBlur(bgr_img, 5)
        return median


