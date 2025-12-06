from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading
import time
import base64
import numpy as np

app = Flask(__name__)

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


# initialize two camera handlers (two columns)
cameras = {
    1: VideoCamera(),
    2: VideoCamera()
}

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

def mjpeg_generator(cam_id):
    cam = cameras.get(cam_id)
    if cam is None:
        return
    boundary = b'--frame'
    while True:
        frame_bytes = cam.get_frame_jpeg()
        if frame_bytes:
            yield b'%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n' % (boundary, len(frame_bytes), frame_bytes)
        else:
            # serve a small blank JPEG fallback so client doesn't break
            blank = create_blank_jpeg()
            yield b'%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n' % (boundary, len(blank), blank)
        time.sleep(0.04)

def create_blank_jpeg():
    # create gray placeholder
    img = 128 * np.ones((240, 320, 3), dtype=np.uint8)
    ret, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes() if ret else b''

@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    # returns multipart mjpeg stream
    return Response(mjpeg_generator(cam_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_source', methods=['POST'])
def set_source():
    # payload: { cam_id: int, source: str }
    data = request.get_json()
    cam_id = int(data.get('cam_id'))
    source = data.get('source', '').strip()
    if cam_id not in cameras:
        return jsonify({'ok': False, 'error': 'invalid cam_id'}), 400
    if source == '':
        # stop camera if empty
        cameras[cam_id].stop()
        return jsonify({'ok': True, 'msg': 'stopped'})
    try:
        cameras[cam_id].start(source)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/capture', methods=['POST'])
def capture():
    # payload: { cam_id: int }
    data = request.get_json()
    cam_id = int(data.get('cam_id'))
    if cam_id not in cameras:
        return jsonify({'ok': False, 'error': 'invalid cam_id'}), 400
    cam = cameras[cam_id]
    frame = cam.get_frame_bgr()
    if frame is None:
        return jsonify({'ok': False, 'error': 'no frame yet'}), 400

    # Convert BGR -> JPEG base64 for immediate display
    ret, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ret:
        return jsonify({'ok': False, 'error': 'encode_failed'}), 500
    raw = jpg.tobytes()
    b64 = base64.b64encode(raw).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + b64

    # Placeholder processing function: crop center square (you can replace)
    processed = process_image_placeholder(frame)

    ret2, jpg2 = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    raw2 = jpg2.tobytes()
    b642 = base64.b64encode(raw2).decode('utf-8')
    processed_uri = 'data:image/jpeg;base64,' + b642

    return jsonify({'ok': True, 'image': data_uri, 'processed': processed_uri})

def process_image_placeholder(bgr_img):
    """
    Placeholder image processing:
    - crop a center square at 50% of min(height,width)
    Replace this with your real processing.
    """
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

if __name__ == '__main__':
    # debug mode off in production
    app.run(host='0.0.0.0', port=5000, threaded=True)
