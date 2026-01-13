# Computer Vision Camera Project

Parking lot surveillance camera system with image processing and license plate recognition.

## 📋 System Requirements

- Python 3.8+
- OpenCV 4.x
- Flask
- Camera (RTSP/HTTP/USB) or video files for testing

## 🚀 Installation

### 1. Clone/Download project

```bash
cd ComputerVisionCamera
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install Tesseract OCR for license plate recognition

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install and add to PATH

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**Mac:**
```bash
brew install tesseract
```

## 🎯 Run Application

```bash
python app.py
```

Open browser: **http://localhost:5000**

## 📁 Project Structure

```
ComputerVisionCamera/
│
├── app.py                  # Flask web server
├── camera.py               # Video camera handler (threading)
├── process.py              # Image processing class ⭐ (STUDENTS IMPLEMENT)
│
├── templates/
│   └── index.html          # Web interface
│
├── static/
│   ├── main.js             # Frontend JavaScript
│   └── style.css           # Styling
│
├── CapturedImage/          # Folder for captured images
│
├── ProjectProgress.txt     # Weekly requirements
├── STUDENT_GUIDE.md        # Detailed student guide ⭐
└── README.md               # This file
```

## 🎓 Student Guide

**See [STUDENT_GUIDE.md](STUDENT_GUIDE.md) for detailed instructions!**

### Task Summary:

Students need to complete the methods in `process.py` according to each step:

1. **Week 1-2:** Basic Image Capture
2. **Week 3:** Image Preprocessing (Grayscale, Gaussian, Canny)
3. **Week 4:** Color Segmentation & Morphology
4. **Week 5:** Camera Calibration & Homography
5. **Week 6-7:** Feature Detection & ROI Extraction
6. **Week 9-10:** Motion Detection & Optical Flow
7. **Week 11-12:** Object Tracking (Kalman Filter)
8. **Week 13-14:** License Plate Detection & OCR
9. **Week 14:** System Integration

## 🖥️ Usage

### Connect Camera

1. Enter camera IP/URL in the input field:
   - RTSP: `rtsp://username:password@ip:port/Streaming/Channels/101`
   - HTTP: `http://ip:port/path{videofeed}`
   - USB: `0` (default camera) or `/dev/video0`

2. Click **Connect**

3. Video stream will display

### Capture & Process

1. Click the 📷 (camera icon) on the video stream

2. Original image displays in **Captured Image**

3. Processed image displays in **Fragment (processed)**

4. Processing time displays below

### Test Individual Steps

You can modify the code to test individual steps:

```python
# In app.py, /capture route
# Change step parameter:
processor.process_frame(frame, step='preprocess')  # Test Step 2
processor.process_frame(frame, step='segment')     # Test Step 3
processor.process_frame(frame, step='roi')         # Test Step 5
processor.process_frame(frame, step='license_plate')  # Test Step 8-9
```

## 📝 Example Camera Sources

### RTSP Cameras
```
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
rtsp://192.168.1.101/live.sdp
```

### HTTP/MJPEG Cameras
```
http://192.168.1.100:8080/video
http://username:password@192.168.1.101/mjpeg
```

### USB Cameras
```
0          # Default camera
1          # Second camera
/dev/video0  # Linux USB camera
```

### Video Files (for testing)
```
D:/Videos/parking_lot.mp4
/home/user/test_video.avi
```

## 🔧 Troubleshooting

### Camera cannot connect

- Check network connectivity
- Verify username/password
- Test RTSP URL with VLC player first
- Try USB camera (source = 0)

### "No frame yet" error when capturing

- Wait a few seconds after connecting for camera buffer to fill
- Check if camera stream is working

### Process time is too long

- Reduce image resolution before processing
- Optimize code (vectorize operations)
- Only run necessary steps (don't run 'all')

### OCR not recognizing text

- Must install Tesseract OCR
- Check if plate image is clear
- Tune preprocessing parameters (threshold, blur, etc.)

## 🎨 Customization

### Change UI

Edit files in `static/` and `templates/`:
- `style.css` - Styling
- `main.js` - Frontend logic
- `index.html` - HTML structure

### Add new processing step

1. Add method to `ImageProcessor` class in `process.py`
2. Call method in `process_frame()` with corresponding step
3. Update frontend to select step (optional)

### Save Results

```python
# In process.py
def process_frame(self, bgr_img, step='all'):
    # ... processing ...
    
    # Save processed image
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"processed_{timestamp}.jpg"
    self.capture_and_save_image(processed_img, filename)
    
    return processed_img, results, process_time_ms
```

## 📊 Performance Tips

1. **Reduce frame resolution:** Resize image before processing
2. **Use ROI:** Only process region of interest
3. **Optimize loops:** Vectorize with NumPy
4. **Parallel processing:** Process 2 cameras in parallel
5. **Cache results:** Save calibration matrix, trained models

## 📚 Reference Documentation

- [OpenCV Documentation](https://docs.opencv.org/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

## 📧 Support

If you have technical issues, refer to:
1. OpenCV documentation
2. Ask your friends, Stack Overflow, ChatGPT, Gemini, Copilot, etc.
3. Contact Dr. Le Trong Nhan

## 📄 License

Educational project - For learning purposes only.

---

**Good luck with your Computer Vision project! 🚀**
