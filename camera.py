import threading
import cv2
import time
# --- Video camera handler per camera ---
class VideoCamera:
    def __init__(self):
        self.source = None
        self.cap = None
        self.frame = None         # BGR numpy array
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self, source):
        # nếu cùng source thì giữ nguyên
        if self.running and self.source == source:
            return
        # stop existing
        self.stop()
        self.source = source
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
        self.cap = None
        self.thread = None
        self.frame = None

    def _reader(self):
        # try open source
        try:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        except:
            self.cap = cv2.VideoCapture(self.source)
        # optional tune: set buffer size or transport
        # read loop
        while self.running:
            if not self.cap or not self.cap.isOpened():
                # try reopen every 2s
                time.sleep(2)
                try:
                    self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                except:
                    self.cap = cv2.VideoCapture(self.source)
                continue
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame = frame.copy()
            # small sleep to relinquish CPU
            time.sleep(0.02)
        # cleanup
        try:
            if self.cap:
                self.cap.release()
        except:
            pass
        self.cap = None

    def get_frame_jpeg(self):
        # return JPEG bytes of current frame, or None
        with self.lock:
            f = None if self.frame is None else self.frame.copy()
        if f is None:
            return None
        # encode as JPEG
        ret, jpeg = cv2.imencode('.jpg', f, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            return None
        return jpeg.tobytes()

    def get_frame_bgr(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def get_frame_bmp(self):
        # return BMP bytes of current frame, or None
        with self.lock:
            f = None if self.frame is None else self.frame.copy()
        if f is None:
            return None
        # encode as BMP
        ret, bmp = cv2.imencode('.bmp', f)
        if not ret:
            return None
        return bmp.tobytes()