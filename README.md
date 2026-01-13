# Computer Vision Camera Project

Hệ thống camera giám sát bãi đỗ xe với xử lý ảnh và nhận dạng biển số xe.

## 📋 Yêu cầu hệ thống

- Python 3.8+
- OpenCV 4.x
- Flask
- Camera (RTSP/HTTP/USB) hoặc video files để test

## 🚀 Cài đặt

### 1. Clone/Download project

```bash
cd ComputerVisionCamera
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. (Tùy chọn) Cài đặt Tesseract OCR cho nhận dạng biển số

**Windows:**
- Download từ: https://github.com/UB-Mannheim/tesseract/wiki
- Install và thêm vào PATH

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**Mac:**
```bash
brew install tesseract
```

## 🎯 Chạy ứng dụng

```bash
python app.py
```

Mở browser: **http://localhost:5000**

## 📁 Cấu trúc project

```
ComputerVisionCamera/
│
├── app.py                  # Flask web server
├── camera.py               # Video camera handler (threading)
├── process.py              # Image processing class ⭐ (SINH VIÊN HOÀN THIỆN)
│
├── templates/
│   └── index.html          # Web interface
│
├── static/
│   ├── main.js             # Frontend JavaScript
│   └── style.css           # Styling
│
├── CapturedImage/          # Thư mục lưu ảnh capture
│
├── ProjectProgress.txt     # Yêu cầu từng tuần
├── STUDENT_GUIDE.md        # Hướng dẫn chi tiết cho sinh viên ⭐
└── README.md               # File này
```

## 🎓 Hướng dẫn cho sinh viên

**Xem file [STUDENT_GUIDE.md](STUDENT_GUIDE.md) để biết chi tiết!**

### Tóm tắt nhiệm vụ:

Sinh viên cần hoàn thiện các phương thức trong `process.py` theo từng bước:

1. **Week 1-2:** Basic Image Capture
2. **Week 3:** Image Preprocessing (Grayscale, Gaussian, Canny)
3. **Week 4:** Color Segmentation & Morphology
4. **Week 5:** Camera Calibration & Homography
5. **Week 6-7:** Feature Detection & ROI Extraction
6. **Week 9-10:** Motion Detection & Optical Flow
7. **Week 11-12:** Object Tracking (Kalman Filter)
8. **Week 13-14:** License Plate Detection & OCR
9. **Week 14:** System Integration

## 🖥️ Sử dụng

### Connect Camera

1. Nhập IP/URL camera vào ô input:
   - RTSP: `rtsp://username:password@ip:port/stream`
   - HTTP: `http://ip:port/video`
   - USB: `0` (camera mặc định) hoặc `/dev/video0`

2. Bấm **Connect**

3. Stream video sẽ hiển thị

### Capture & Process

1. Bấm nút 📷 (camera icon) trên video stream

2. Ảnh gốc hiển thị trong **Captured Image**

3. Ảnh đã xử lý hiển thị trong **Fragment (processed)**

4. Thời gian xử lý hiển thị bên dưới

### Test từng Step

Có thể modify code để test từng bước:

```python
# Trong app.py, route /capture
# Thay đổi step parameter:
processor.process_frame(frame, step='preprocess')  # Test Step 2
processor.process_frame(frame, step='segment')     # Test Step 3
processor.process_frame(frame, step='roi')         # Test Step 5
processor.process_frame(frame, step='license_plate')  # Test Step 8-9
```

## 📝 Ví dụ Camera Sources

### RTSP Cameras
```
rtsp://admin:password@192.168.1.100:554/stream1
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

### Camera không connect được

- Kiểm tra network connectivity
- Verify username/password
- Test RTSP URL bằng VLC player trước
- Thử với camera USB (source = 0)

### Lỗi "No frame yet" khi Capture

- Đợi vài giây sau khi connect để camera buffer đủ frames
- Kiểm tra camera stream có hoạt động không

### Process time quá lâu

- Giảm resolution ảnh trước khi xử lý
- Optimize code (vectorize operations)
- Chỉ chạy các bước cần thiết (không chạy 'all')

### OCR không nhận dạng được

- Cần cài đặt Tesseract OCR
- Kiểm tra plate image có rõ ràng không
- Tune preprocessing parameters (threshold, blur, etc.)

## 🎨 Customization

### Thay đổi UI

Edit files trong `static/` và `templates/`:
- `style.css` - Styling
- `main.js` - Frontend logic
- `index.html` - HTML structure

### Thêm processing step mới

1. Thêm method vào `ImageProcessor` class trong `process.py`
2. Gọi method trong `process_frame()` với step tương ứng
3. Update frontend để chọn step (optional)

### Lưu kết quả

```python
# Trong process.py
def process_frame(self, bgr_img, step='all'):
    # ... processing ...
    
    # Save processed image
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"processed_{timestamp}.jpg"
    self.capture_and_save_image(processed_img, filename)
    
    return processed_img, results, process_time_ms
```

## 📊 Performance Tips

1. **Reduce frame resolution:** Resize ảnh trước khi xử lý
2. **Use ROI:** Chỉ xử lý vùng quan tâm
3. **Optimize loops:** Vectorize với NumPy
4. **Parallel processing:** Xử lý 2 cameras song song
5. **Cache results:** Lưu calibration matrix, trained models

## 📚 Tài liệu tham khảo

- [OpenCV Documentation](https://docs.opencv.org/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [STUDENT_GUIDE.md](STUDENT_GUIDE.md) - Hướng dẫn chi tiết

## 📧 Support

Nếu có vấn đề kỹ thuật, tham khảo:
1. File `STUDENT_GUIDE.md` để xem hướng dẫn chi tiết
2. OpenCV documentation
3. Stack Overflow với tag `opencv` và `python`

## 📄 License

Educational project - For learning purposes only.

---

**Good luck with your Computer Vision project! 🚀**
