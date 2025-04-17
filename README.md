# Stampede Risk Prediction Webapp

This project implements a web application for analyzing video and image content to detect potential stampede risks based on crowd density. It utilizes computer vision techniques, a Flask web server, and integration with the Fluvio streaming data platform.

---

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Core Functionality](#core-functionality)
  - [People Detection](#people-detection)
  - [Density Grid Analysis](#density-grid-analysis)
  - [Risk Status Determination](#risk-status-determination)
  - [Fluvio Integration](#fluvio-integration)
  - [Web Interface](#web-interface)
- [File Structure](#file-structure)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Setting up the Python Environment](#setting-up-the-python-environment)
  - [Setting up Fluvio](#setting-up-fluvio)
  - [Project File Setup](#project-file-setup)
- [How to Run](#how-to-run)
- [How to Use](#how-to-use)
  - [Upload Media](#upload-media)
  - [View Live Stream](#view-live-stream)
- [Future Enhancements](#future-enhancements)

---

## Project Description

The Stampede Risk Prediction Webapp provides tools to analyze crowd levels in visual media. It can process uploaded video files and images or connect to a live camera feed.

The core process involves identifying individuals, calculating their distribution within a defined grid, and assessing density levels to flag potential high-risk areas or overall critical situations. All processing results, including density data and risk status, are published to a Fluvio topic for external consumption and analysis.

---

## Features

- Analyze uploaded video files and images for crowd density and risk.
- Provide a live camera feed view (or fallback video) with real-time processing overlays.
- Utilize a pre-trained TensorFlow Object Detection model (SSD MobileNet V2) for people detection.
- Implement a configurable grid system for spatial density analysis.
- Determine and visualize risk statuses:
  - `"Normal"`
  - `"High Density Warning"`
  - `"CRITICAL RISK"`
- Overlay color-coded indicators on high-density grid cells.
- Display processing metrics: overall status, max person count, and processing time.
- For videos, save and display the frame with the highest risk status.
- Download link for full processed video after analysis.
- Audible beep on the results page for critical detections.
- Publish frame-by-frame results to a Fluvio streaming data topic.
- Separate consumer script (`predict_stampede.py`) for reading from Fluvio.

---

## Core Functionality

### People Detection

- Uses TensorFlow Hub model `ssd_mobilenet_v2/fpnlite_320x320`.
- Filters detections by `person` class (index 1) and confidence threshold (default 0.25).
- Extracts bounding boxes and detection scores.

### Density Grid Analysis

- Divides frame into an 8x8 grid (default).
- Maps detected people to cells based on the center of their bounding boxes.
- Populates a 2D `density_grid` with person counts per cell.

### Risk Status Determination

- Flags cells above `HIGH_DENSITY_THRESHOLD` or `CRITICAL_DENSITY_THRESHOLD`.
- Statuses:
  - `Normal`
  - `High Density Cell Detected`
  - `High Density Warning`
  - `Critical Density Cell Detected`
  - `CRITICAL RISK`
- `STATUS_HIERARCHY` defines precedence for video-level status.

### Fluvio Integration

- Connects to a Fluvio cluster and sends frame data to a topic (`crowd-data` by default).
- JSON payload includes:
  - `timestamp`, `frame index`, `density_grid`, `frame_status`, etc.
- `predict_stampede.py` demonstrates consuming these messages.

### Web Interface

- Built with Flask and Jinja2 templates.
- Pages:
  - `/` — Home page with upload options.
  - `/live` — Live stream with processing overlay.
  - `/video_feed` — MJPEG stream route.
  - `/upload_media` — Handles media upload and processing.
  - `/results` — Shows processed results and download options.

---

## File Structure

```
├── static/
│   ├── debug_frames/        # Optional debug frames
│   ├── processed_frames/    # Critical frames from videos
│   ├── processed_images/    # Processed uploaded images
│   ├── processed_videos/    # Full processed videos
│   └── beep.mp3             # Sound for critical alert
├── templates/
│   ├── index.html           # Upload page
│   ├── live.html            # Live stream viewer
│   └── results.html         # Results page
├── uploads/                 # Temporary upload storage
├── venv/                    # (Optional) Python Virtual Env
├── app.py                   # Main app logic
├── predict_stampede.py      # Fluvio consumer example
└── videoplayback.mp4        # Fallback video for webcam
```

---

## Setup

### Prerequisites

- Python 3.7+
- pip
- Access to a Fluvio cluster (use WSL on Windows if needed)

### Setting up the Python Environment

```bash
git clone <your-repo-url>
cd <your-repo-name>
python -m venv venv
# Activate the virtual environment:
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install Flask opencv-python tensorflow tensorflow-hub fluvio-python werkzeug mimetypes
```

### Setting up Fluvio

- Install Fluvio CLI: [Official Docs](https://www.fluvio.io/docs/install/)
- Start local Fluvio cluster:

```bash
fluvio cluster start
fluvio topic create crowd-data
```

Ensure the topic name matches `FLUVIO_CROWD_TOPIC` in `app.py`.

### Project File Setup

- **ML Model**: Auto-downloaded by `app.py` on first run.
- **Sound File**: Place `beep.mp3` in `static/`. Update `results.html` if you change the name.
- **Fallback Video**: Add `videoplayback.mp4` to the root directory.

---

## How to Run

1. Start your Fluvio cluster:
   ```bash
   fluvio cluster start
   fluvio topic create crowd-data
   ```
2. Run the Fluvio Consumer:
   ```bash
   python predict_stampede.py
   ```
4. Start the Flask app (preferably in a new terminal):
   ```bash
   python app.py
   ```
   Access at [http://localhost:5000](http://localhost:5000)

---

## How to Use

### Upload Media

1. Navigate to the main page.
2. Choose an image/video file to upload.
3. Click "Upload and Analyze".
4. Wait for processing to complete.
5. View:
   - Overall Risk Status
   - Max Persons Detected
   - Processing Time
   - Critical frame preview
6. Download the full processed video (if applicable).
7. A beep will play on `CRITICAL RISK` (browser sound must be on).

### View Live Stream

1. Go to [http://localhost:5000/live](http://localhost:5000/live)
2. The system attempts to access the default webcam.
3. If unavailable, `videoplayback.mp4` is used.
4. Real-time overlays and statuses will be shown.

---

## Future Enhancements

- **Temporal Analysis**: Use movement tracking and optical flow for dynamic behavior analysis.
- **Camera Calibration**: Adjust for perspective distortion to improve density measurement.
- **Scalability**: Add background job queues (e.g., Celery) and multi-feed architecture.
- **Advanced Alerting**: SMS, email, or push notifications for critical events.
- **Dashboard**: Web dashboard for real-time and historical monitoring.
- **Model Optimization**: Quantization or use of lighter ML models for better performance.
