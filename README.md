# Multi Object Tracking System

A real-time multi-object tracking system built with Python, OpenCV, YOLO, and Streamlit.  
The project supports object detection, tracking, class filtering, live webcam/video processing, and an interactive dashboard for monitoring tracking performance.

---

# Features

- Real-time object detection using YOLO
- Multi-object tracking with unique IDs
- Upload video files or use live webcam
- Track specific object IDs
- Filter tracking by object class
- Live FPS monitoring
- Progress bar for video processing
- Pause and resume tracking
- Streamlit interactive dashboard
- Save processed output video

---

# Supported Classes

- Person
- Car
- Truck
- Bus
- Motorcycle
- All classes mode

---

# Project Structure

```bash
Selective_Object_Tracking/
│
├── configs/
│   └── config.py
│
├── detection/
│   └── detector.py
│
├── tracker/
│   └── tracking.py
│
├── outputs/
│   └── output.mp4
│
├── ui/
│   └── streamlit_app.py
│
├── pipeline.py
├── requirements.txt
└── README.md
```

---

# Technologies Used

- Python
- OpenCV
- Streamlit
- Ultralytics YOLO
- NumPy

---

# Installation

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd Selective_Object_Tracking
```

---

## 2. Create Conda virtual environment

### Windows

```bash
Conda create -n myenv
conda activate myenv
```


---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Application

```bash
streamlit run ui/streamlit_app.py
```

---

# How It Works

## 1. Video Source Selection

The system supports:
- Uploaded videos
- Live webcam streams

---

## 2. Object Detection

YOLO detects objects frame-by-frame and returns:
- Bounding boxes
- Confidence scores
- Class IDs

---

## 3. Object Tracking

The tracker assigns:
- Unique tracking IDs
- Persistent tracking across frames

---

## 4. Visualization

Each tracked object is displayed with:
- Bounding box
- Track ID
- Class name

Example:

```text
ID: 12 | person
ID: 7  | car
```

---

# Streamlit Dashboard

The dashboard includes:

- FPS monitor
- Number of tracked objects
- Video progress bar
- Source selection
- Class filtering
- Start/Stop controls

---

# Example Workflow

1. Upload a video or choose webcam
2. Select object class
3. Click "Start Tracking"
4. Monitor tracking results live
5. Pause/resume anytime
6. Processed video is saved automatically

---

# Output

Processed videos are saved to:

```bash
outputs/output.mp4
```

---

# Future Improvements

- RTSP/IP camera support
- Heatmap visualization
- Track history visualization
- Multi-camera support
- GPU acceleration optimization
- Object counting analytics
- Re-identification support

---

# Common Issues

## Video not playing in Streamlit

Use compatible codecs such as:

```python
cv2.VideoWriter_fourcc(*"mp4v")
```

---

## Webcam warnings on Windows

If OpenCV MSMF warnings appear:

```python
cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
```

---

## Empty detection crash

Always validate detections before tracking:

```python
if len(xyxy) == 0:
    tracked_objects = []
```

---

# Author

Built by Omar Shaaban.

