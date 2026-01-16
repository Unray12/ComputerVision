from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading
import time
import base64
import numpy as np
from process import ImageProcessor
from camera import VideoCamera
app = Flask(__name__)


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
    """
    Capture image from camera and process it
    
    Payload: { cam_id: int, step: str (optional) }
    
    Step options:
        - 'preprocess': Step 2 - Grayscale, Gaussian, Edge Detection
        - 'segment': Step 3 - Color segmentation, Morphology
        - 'calibrate': Step 4 - Calibration and perspective correction
        - 'roi': Step 5 - Feature detection and ROI extraction
        - 'motion': Step 6 - Motion detection
        - 'track': Step 7 - Object tracking
        - 'license_plate': Steps 8-9 - License plate detection & OCR
        - 'all': Complete pipeline (default)
    """
    data = request.get_json()
    cam_id = int(data.get('cam_id'))
    # step = data.get('step', 'all')  # Default to 'all' if not specified
    
    if cam_id not in cameras:
        return jsonify({'ok': False, 'error': 'invalid cam_id'}), 400
    
    cam = cameras[cam_id]
    frame = cam.get_frame_bgr()
    if frame is None:
        return jsonify({'ok': False, 'error': 'no frame yet'}), 400

    # Convert BGR -> JPEG base64 for immediate display (original image)
    ret, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ret:
        return jsonify({'ok': False, 'error': 'encode_failed'}), 500
    raw = jpg.tobytes()
    b64 = base64.b64encode(raw).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + b64

    # Process image using ImageProcessor
    try:
        processor = ImageProcessor()
        processed, results, process_time_ms = processor.process_frame(frame)
        
        # Convert processed image to base64
        ret2, jpg2 = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ret2:
            return jsonify({'ok': False, 'error': 'processed_encode_failed'}), 500
        raw2 = jpg2.tobytes()
        b642 = base64.b64encode(raw2).decode('utf-8')
        processed_uri = 'data:image/jpeg;base64,' + b642
        
        return jsonify({
            'ok': True, 
            'image': data_uri, 
            'processed': processed_uri, 
            'process_time_ms': round(process_time_ms, 2),
            'results': results,  # Additional processing results
            'step': "all"
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    # debug mode off in production
    app.run(host='0.0.0.0', port=5000, threaded=True)
