# Stampede Risk Prediction Webapp

## Project Description

This project is a web application designed to analyze video feeds or uploaded media content (videos and images) to detect people, estimate crowd density using a grid-based approach, and predict potential stampede risks. The application provides a web interface for users to upload media or view a live camera feed, and it sends real-time density and status data to a Fluvio streaming data platform.

Developed for a hackathon, this project demonstrates the integration of computer vision (using OpenCV and TensorFlow Object Detection), Flask web framework, and Fluvio for data streaming.

## Features

* Upload video or image files for analysis.
* View a live camera feed (or fallback video) with real-time processing overlays.
* Detect people in frames/images using a pre-trained ML model (SSD MobileNet V2).
* Calculate crowd density based on a configurable grid overlaid on the analysis area.
* Assign risk statuses ("Normal", "High Density Warning", "CRITICAL RISK") based on density thresholds.
* Visualize risk levels with color-coded overlays on grid cells.
* Display overall analysis status, maximum person count detected, and processing time.
* For uploaded videos, display the "most critical" frame encountered during processing.
* Provide a download link for the full processed video file.
* Trigger an audible beep on the results page for critical risk detections in uploaded media.
* Send real-time processing data (grid density, status, counts) to a Fluvio topic.
* Includes a separate Python script (`predict_stampede.py`) to demonstrate consuming data from the Fluvio topic.

## File Structure

.
├── static
│   ├── debug_frames         (Not actively used in final version)
│   ├── processed_frames     (Stores images of critical/display frames)
│   ├── processed_images     (Stores processed uploaded images)
│   ├── processed_videos     (Stores full processed uploaded videos)
│   └── beep.mp3             (Required sound file for critical alert)
├── templates
│   ├── index.html           (Main upload/navigation page)
│   ├── live.html            (Live stream viewing page)
│   └── results.html         (Displays processing results for uploaded media)
├── uploads                  (Temporarily stores uploaded files)
├── venv                     (Python Virtual Environment - recommended)
├── app.py                   (Main Flask application file)
├── load_model.py            (Not used in final app.py - model loaded directly)
├── predict_stampede.py      (Separate script to consume Fluvio data)
├── send_crowd_data.py       (Functionality integrated into app.py)
├── test_camera.py           (Not used in final app.py)
├── test_fluvio.py           (Not used in final app.py)
├── test.py                  (Not used in final app.py)
└── videoplayback.mp4        (Optional: Fallback video file for live stream)


## Setup

### Prerequisites

* Python 3.7+
* `pip` (Python package installer)

### Setting up the Environment

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
3.  **Activate the Virtual Environment:**
    * On Windows:
        ```bash
        venv\Scripts\activate
        ```
    * On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
4.  **Install Dependencies:**
    ```bash
    pip install Flask opencv-python tensorflow tensorflow-hub fluvio-python werkzeug mimetypes
    ```

### Setting up Fluvio

Fluvio typically runs on Linux. If you are on Windows, using **WSL (Windows Subsystem for Linux)** is the recommended way to run a local Fluvio cluster.

1.  **Install Fluvio CLI:** Follow the official Fluvio installation guide for your operating system or WSL distribution: [https://www.fluvio.io/docs/install/](https://www.fluvio.io/docs/install/)
2.  **Start a Local Fluvio Cluster:**
    ```bash
    fluvio cluster start
    ```
3.  **Create the Crowd Data Topic:**
    ```bash
    fluvio topic create crowd-data
    ```
    Ensure the topic name matches `FLUVIO_CROWD_TOPIC` in `app.py` and `predict_stampede.py`.

### Project File Setup

1.  **ML Model:** The application automatically downloads the pre-trained TensorFlow Hub model on startup. You just need an internet connection when you run `app.py` for the first time.
2.  **Sound File:** Place a short MP3 or WAV sound file (e.g., for a beep alert) in the `static` directory. Update the `<audio>` tag's `src` in `templates/results.html` if you use a different filename (`beep.mp3` is the default expected name).
3.  **Fallback Video (Optional but Recommended):** For the live stream demo, place a video file named `videoplayback.mp4` in the root directory of your project (the same directory as `app.py`). This video will be used as a source if your webcam (`cv2.VideoCapture(0)`) fails to open.

## How to Run

1.  **Ensure your Fluvio cluster is running** (e.g., in your WSL terminal run `fluvio cluster start`).
2.  **Ensure the `crowd-data` topic exists** (`fluvio topic list`).
3.  **Open a terminal or command prompt** where your Python virtual environment is activated.
4.  **Navigate to the project directory.**
5.  **Run the Fluvio Consumer (Optional but good for demo):** Open a *separate* terminal where Fluvio client is installed (e.g., another WSL terminal) and run:
    ```bash
    python predict_stampede.py
    ```
    This script will connect to Fluvio and print the data it receives from the webapp.
6.  **Run the Flask Application:** In the terminal where your Python virtual environment is activated, run:
    ```bash
    python app.py
    ```
    The Flask development server will start, typically at `http://localhost:5000/`.

## How to Use

1.  Open your web browser and go to `http://localhost:5000/`.
2.  **Upload Media:**
    * Click the "Choose File" button to select a video or image file.
    * Click "Upload and Analyze".
    * The application will process the file and redirect you to the results page (`results.html`).
    * On the results page, you will see the overall risk status, max persons detected, processing time, and either the processed image (for images) or the most critical frame image (for videos). A button to download the full processed video will appear for video uploads.
    * If the status is critical, you should hear a beep (check browser sound permissions).
    * Data for each processed frame/image is sent to the `crowd-data` Fluvio topic, visible in the terminal running `predict_stampede.py`.
3.  **View Live Stream:**
    * Click the "View Live Stream" link on the index page or navigate directly to `http://localhost:5000/live`.
    * This page (`live.html`) will display an MJPEG stream (`/video_feed`) from your webcam or the fallback video file.
    * The video feed will show real-time processing overlays (density areas, status text).
    * Data for each processed frame is sent to the `crowd-data` Fluvio topic, visible in the terminal running `predict_stampede.py`.

## Project Structure Explanation

* `app.py`: The main Flask application. Handles routing (`/`, `/upload_media`, `/live`, `/video_feed`), loads the ML model, contains the core `process_media_content` function, manages file uploads, video/image processing, saving results, and sending data to Fluvio.
* `templates/`: Contains the HTML files rendered by Flask.
    * `index.html`: The starting page with upload form and live stream link.
    * `results.html`: Displays the output and metrics after uploading a file. Includes JavaScript for the critical alert beep and a download link for processed videos.
    * `live.html`: Embeds the MJPEG stream from the `/video_feed` route to show live processing.
* `static/`: Serves static files like CSS, JavaScript, and output media (`.mp4`, `.jpg`, `.mp3`).
* `uploads/`: A temporary directory where uploaded files are saved before processing.
* `predict_stampede.py`: A standalone script demonstrating how to consume the processed crowd data from the `crowd-data` Fluvio topic.

## Potential Improvements (Beyond Hackathon Scope)

* **Movement Analysis:** Implement object tracking and optical flow to detect panicked crowd movement, which is crucial for true stampede prediction beyond just density.
* **Camera Calibration:** Implement perspective transformation for accurate density calculations from cameras not mounted directly overhead.
* **Scalability:** Use background job queues (e.g., Celery) for file processing and explore distributed processing for multiple live feeds.
* **Advanced Alerting:** Integrate email, SMS, or dashboard-based alerts triggered by critical status.
* **User Interface:** Develop a more sophisticated dashboard showing live status for multiple cameras, historical data, and configuration options.
* **Model Optimization:** Quantize the ML model or use a more lightweight architecture for better performance on edge devices.

This documentation covers the project setup, running, usage, and structure, along with acknowledging areas for future development, making it comprehensive for your GitHub repository. Remember to replace `<your-repo-url>` and `

<your-repo-name>` with your actual GitHub details.

