# **Stampede Risk Prediction Webapp**

This project implements a web application for analyzing video and image content to detect potential stampede risks based on crowd density. It utilizes computer vision techniques, a Flask web server, and integration with the Fluvio streaming data platform.

## **Table of Contents**

1. [Project Description](#bookmark=id.btwdj17m49ff)  
2. [Features](#bookmark=id.r6xd25lctfee)  
3. [Core Functionality](#bookmark=id.fbp7cd2d75ou)  
   * [People Detection](#bookmark=id.a4oql1aaej4z)  
   * [Density Grid Analysis](#bookmark=id.pafc0v9xx5dh)  
   * [Risk Status Determination](#bookmark=id.y74n6yvzo0si)  
   * [Fluvio Integration](#bookmark=id.r5d9auefi3b7)  
4. [Web Interface](#bookmark=id.ty7rzdllt9kp)  
5. [File Structure](#bookmark=id.joaoj967kjco)  
6. [Setup](#bookmark=id.qmj0g03wrqwo)  
   * [Prerequisites](#bookmark=id.9cbwls4mhrt0)  
   * [Setting up the Python Environment](#bookmark=id.u4be00s0vdr8)  
   * [Setting up Fluvio](#bookmark=id.lth3tsnzxy3i)  
   * [Project File Setup](#bookmark=id.za5df5qa6hzf)  
7. [How to Run](#bookmark=id.7f9qhqkg1zg0)  
8. [How to Use](#bookmark=id.d477e1mo0pwd)  
   * [Upload Media](#bookmark=id.ej1udiwp04dd)  
   * [View Live Stream](#bookmark=id.mgwzhpbpgqur)  
9. [Future Enhancements](#bookmark=id.xm1e1wotfo3o)

## **Project Description**

The Stampede Risk Prediction Webapp provides tools to analyze crowd levels in visual media. It can process uploaded video files and images, or connect to a live camera feed. The core process involves identifying individuals, calculating their distribution within a defined grid, and assessing density levels to flag potential high-risk areas or overall critical situations. All processing results, including density data and risk status, are published to a Fluvio topic for external consumption and analysis. The web interface allows users to interact with the system, upload content, view processed results, and monitor the live analysis feed.

## **Features**

* Analyze uploaded video files and images for crowd density and risk.  
* Provide a live camera feed view with real-time processing overlays.  
* Utilize a pre-trained TensorFlow Object Detection model (SSD MobileNet V2) for people detection.  
* Implement a configurable grid system for spatial density analysis.  
* Determine and visualize risk statuses ("Normal", "High Density Warning", "CRITICAL RISK") based on adjustable density thresholds.  
* Overlay color-coded indicators on high-density grid cells.  
* Display processing metrics such as overall status, maximum person count, and processing time.  
* For uploaded videos, save and display the single frame identified as having the highest risk status.  
* Offer a direct download link for the full processed video file after analysis.  
* Include an audible alert on the results page for critical risk detections in uploaded media.  
* Publish detailed processing data for each analyzed frame/image to a Fluvio streaming data topic.  
* Provide a separate script (predict\_stampede.py) as an example of how to consume the data from the Fluvio topic.

## **Core Functionality**

The application's core functionality is implemented within the app.py file, primarily centered around the process\_media\_content function and the logic within the Flask routes.

### **People Detection**

* The application loads a pre-trained object detection model (ssd\_mobilenet\_v2/fpnlite\_320x320) from TensorFlow Hub on startup.  
* For each input frame (from a video, image, or live stream), the model is run to detect objects.  
* Detections are filtered to include only objects classified as 'person' (PERSON\_CLASS\_INDEX \= 1\) with a confidence score above a defined DETECTION\_THRESHOLD (default 0.25).  
* The bounding boxes and scores of these confirmed person detections are obtained.

### **Density Grid Analysis**

* The input frame/image is conceptually divided into a grid of GRID\_ROWS x GRID\_COLS cells (default 8x8).  
* For each confirmed person detection, the center point of their bounding box is calculated.  
* This center point is then mapped to a specific cell within the grid.  
* A 2D array (density\_grid) is populated, where each element represents the count of people whose center point falls within that corresponding grid cell.

### **Risk Status Determination**

* Based on the density\_grid, the application analyzes the count of people in each cell.  
* Cells exceeding HIGH\_DENSITY\_THRESHOLD are flagged as high density.  
* Cells exceeding CRITICAL\_DENSITY\_THRESHOLD are flagged as critical density.  
* An overall status for the frame/image is determined based on the *number* of high and critical density cells:  
  * "Normal": No cells exceed thresholds.  
  * "High Density Cell Detected": At least one cell exceeds HIGH\_DENSITY\_THRESHOLD.  
  * "High Density Warning": The number of high density cells meets or exceeds HIGH\_DENSITY\_CELL\_COUNT\_THRESHOLD.  
  * "Critical Density Cell Detected": At least one cell exceeds CRITICAL\_DENSITY\_THRESHOLD.  
  * "CRITICAL RISK": The number of critical density cells meets or exceeds CRITICAL\_DENSITY\_CELL\_COUNT\_THRESHOLD.  
* A STATUS\_HIERARCHY dictionary defines the priority of these statuses, used to determine the overall status for an entire video (the highest priority status encountered across all frames).

### **Fluvio Integration**

* On application startup, the system attempts to connect to a Fluvio cluster and obtain a topic\_producer for the specified FLUVIO\_CROWD\_TOPIC (default "crowd-data").  
* After processing each frame or image in process\_media\_content, a JSON payload containing the timestamp, frame index, density\_grid, frame\_status, confirmed\_persons, and counts of high/critical cells is constructed.  
* This JSON payload is sent as a record to the configured Fluvio topic using the fluvio\_producer.  
* The separate predict\_stampede.py script demonstrates connecting as a partition\_consumer to the same topic and printing the received data, illustrating the data flow through Fluvio.

## **Web Interface**

The application provides a simple web interface built with Flask and Jinja2 templates:

* / (index.html): The landing page, offering options to upload media or navigate to the live stream page.  
* /live (live.html): Displays the real-time processed video feed from the camera or fallback video source via an \<img\> tag pointing to the /video\_feed route.  
* /video\_feed: A Flask route that streams processed frames as an MJPEG multi-part response, consumed by the \<img\> tag in live.html.  
* /upload\_media: A Flask route that handles POST requests with uploaded files, performs the analysis, saves results, and redirects to /results.  
* /results (results.html): Displays the outcome of the media upload analysis, including the overall status, metrics, the processed image or critical frame image, and a download link for processed videos. Includes client-side JavaScript to play a sound on critical status.

## **File Structure**

.  
├── static/                  \# Static files served by the web server  
│   ├── debug\_frames/        \# (Optional) For saving debug frames  
│   ├── processed\_frames/    \# Stores images of critical/display frames from videos  
│   ├── processed\_images/    \# Stores processed uploaded images  
│   ├── processed\_videos/    \# Stores full processed uploaded videos  
│   └── beep.mp3             \# Required sound file for critical alert on results page  
├── templates/               \# HTML templates rendered by Flask  
│   ├── index.html           \# Main upload/navigation page  
│   ├── live.html            \# Live stream viewing page  
│   └── results.html         \# Displays processing results for uploaded media  
├── uploads/                 \# Temporary directory for uploaded files  
├── venv/                    \# (Optional) Python Virtual Environment  
├── app.py                   \# Main Flask application and processing logic  
├── predict\_stampede.py      \# Separate script to demonstrate Fluvio data consumption  
└── videoplayback.mp4        \# (Optional) Fallback video file for live stream if webcam fails

## **Setup**

### **Prerequisites**

* Python 3.7+  
* pip (Python package installer)  
* Access to a Fluvio cluster (local or remote). If running locally on Windows, WSL is typically required for the Fluvio cluster itself.

### **Setting up the Python Environment**

1. **Clone the repository:**  
   git clone \<your-repo-url\>  
   cd \<your-repo-name\>

2. **Create a Virtual Environment (Highly Recommended):**  
   python \-m venv venv

3. **Activate the Virtual Environment:**  
   * On Windows:  
     venv\\Scripts\\activate

   * On macOS and Linux:  
     source venv/bin/activate

4. **Install Dependencies:**  
   pip install Flask opencv-python tensorflow tensorflow-hub fluvio-python werkzeug mimetypes

### **Setting up Fluvio**

If you don't have a running Fluvio cluster, you'll need to set one up. On Windows, this is most commonly done using WSL.

1. **Install Fluvio CLI:** Follow the official Fluvio installation guide: [https://www.fluvio.io/docs/install/](https://www.fluvio.io/docs/install/)  
2. **Start a Local Fluvio Cluster:**  
   fluvio cluster start

3. **Create the Crowd Data Topic:** The application publishes to a topic named crowd-data. Create this topic:  
   fluvio topic create crowd-data

   Ensure the topic name matches FLUVIO\_CROWD\_TOPIC in app.py and predict\_stampede.py.

### **Project File Setup**

1. **ML Model:** The necessary TensorFlow Hub model is downloaded automatically by app.py when it runs for the first time (requires internet access).  
2. **Sound File:** Obtain a short sound file (e.g., beep.mp3) and place it in the static directory. If you use a different filename, update the src attribute of the \<audio\> tag in templates/results.html.  
3. **Fallback Video (Optional but Recommended for Live Stream):** Place a video file named videoplayback.mp4 in the root directory of your project (the same directory as app.py). This file will be used as the source for the live stream if the application cannot access the default webcam (cv2.VideoCapture(0)).

## **How to Run**

1. **Ensure your Fluvio cluster is running** (e.g., in your WSL terminal, run fluvio cluster start).  
2. **Ensure the crowd-data topic exists.**  
3. **Open a terminal or command prompt** where your Python virtual environment is activated.  
4. **Navigate to the project directory.**  
5. **Run the Fluvio Consumer (Optional):** To see the data being sent to Fluvio in real-time, open a *separate* terminal where the Fluvio client is installed and can connect to your cluster (e.g., another WSL terminal) and run:  
   python predict\_stampede.py

6. **Run the Flask Application:** In the terminal where your Python virtual environment is activated, run:  
   python app.py

   The Flask development server will start, typically accessible at http://localhost:5000/.

## **How to Use**

1. Open your web browser and go to http://localhost:5000/.

### **Upload Media**

1. On the main page, click the "Choose File" button.  
2. Select a video (.mp4, .avi, etc.) or image (.jpg, .png, etc.) file from your computer.  
3. Click the "Upload and Analyze" button.  
4. The application will process the file. This may take some time for videos.  
5. You will be redirected to the results page (results.html).  
6. On the results page, review the "Overall Risk Status", "Max Persons Detected", and "Processing Time".  
7. An image will be displayed: the processed image for an image upload, or the "most critical" frame image for a video upload.  
8. For video uploads, a "Download Full Processed Video" button will be available.  
9. If the "Overall Risk Status" is "CRITICAL RISK" or "Critical Density Cell Detected", a beep sound should play (ensure browser sound is enabled).  
10. The terminal running predict\_stampede.py will show the data sent to Fluvio during the processing.

### **View Live Stream**

1. On the main page, click the "View Live Stream" link or navigate directly to http://localhost:5000/live.  
2. The page will attempt to connect to your default webcam (cv2.VideoCapture(0)). If it fails, it will try to load videoplayback.mp4 from the project root directory.  
3. The page displays a real-time feed with the processing overlays (density areas, status text, person count).  
4. The terminal running predict\_stampede.py will show the data sent to Fluvio for each frame of the live stream.

## **Future Enhancements**

This project provides a strong foundation. Potential areas for further development include:

* **Temporal Analysis:** Implement algorithms to analyze movement patterns across frames (e.g., optical flow, tracking) to better predict stampedes based on dynamic behavior, not just static density.  
* **Camera Calibration:** Add functionality to calibrate cameras to account for perspective distortion and improve density accuracy.  
* **Scalability:** Implement asynchronous processing for uploads (e.g., using Celery) and explore architectures for handling multiple live camera feeds concurrently.  
* **Advanced Alerting:** Integrate email, SMS, or push notification services to send alerts based on critical risk detection.  
* **Dashboard & Monitoring:** Create a dedicated dashboard page to monitor multiple feeds, view historical data, and configure system settings.  
* **Model Improvement:** Explore using different or fine-tuning existing object detection models for better performance in specific crowd scenarios.