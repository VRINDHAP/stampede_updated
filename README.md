![github-submission-banner](https://github.com/user-attachments/assets/a1493b84-e4e2-456e-a791-ce35ee2bcf2f)


# 🚀 Stampede Predictor


> A web application for real-time crowd density analysis and stampede risk prediction.


---


## 📌 Problem Statement


Problem Statement: Enhancing Public Safety through Proactive Crowd Monitoring


---


## 🎯 Objective


The primary objective of the Stampede Predictor project is to develop and deploy a robust system capable of analyzing crowd dynamics from video feeds (live streams or uploaded files) to identify potential stampede risks. It aims to provide actionable insights and early warnings to stakeholders such as event organizers, security personnel, and public safety agencies. By visually highlighting high-density areas and indicating risk levels, the application empowers users to take timely preventative measures, thereby enhancing safety and potentially saving lives in crowded environments like concerts, festivals, sporting events, and public gatherings. The real-world value lies in transforming reactive crowd management into a proactive safety strategy.


---


## 🧠 Team & Approach


### Team Name:

`Team Arete`


### Team Members:

- Nayana Shaji Mekkunnel (GitHub / LinkedIn / Role)

- Gabriel James

- Jany Sabarinath

- Vrindha P


### Your Approach:

- **Why you chose this problem:** We were motivated by the critical need for improved safety measures in crowded public spaces. Witnessing incidents in such environments underscored the importance of developing a tool that could proactively identify and alert to potential dangers before they escalate into stampedes. Leveraging AI for this purpose felt like a direct way to contribute to public safety.
- **Key challenges you addressed:** Significant challenges included achieving efficient real-time object detection (specifically, detecting people) in varying conditions, accurately calculating and interpreting crowd density across a defined grid, managing the streaming of video data and analysis results, ensuring the application could handle both live camera feeds and static file uploads, and integrating a messaging queue like Fluvio for decoupled data processing and monitoring. Building a responsive and intuitive user interface that clearly communicates risk levels was also a key focus.
- **Any pivots, brainstorms, or breakthroughs during hacking:** An initial consideration was using simpler image processing techniques, but brainstorming led us to adopt a deep learning approach with YOLO for more accurate and robust person detection. Integrating Fluvio was a crucial pivot that allowed us to separate the core analysis logic from the web presentation layer, making the system more scalable and enabling external consumption of the crowd data. Developing the Server-Sent Events (SSE) implementation for the live feed was a breakthrough that provided seamless, real-time status updates to the user interface without constant polling.


---


## 🛠️ Tech Stack


### Core Technologies Used:

- Frontend: HTML, CSS, JavaScript (for the web interface and live status updates via SSE)
- Backend: Flask (Python framework for the web server, handling requests and running the analysis)
- Database: None (The application focuses on real-time processing and data streaming rather than persistent storage of analysis results, though Fluvio could potentially feed into a database).
- APIs:
    - OpenCV (`cv2`): For video/image capture, processing, drawing overlays, and encoding frames.
    - Ultralytics (`YOLO`): Provides the implementation for the YOLOv11 Nano object detection model.
    - Fluvio: Used as a real-time messaging queue for streaming analysis data.
- Hosting: Local Development Environment (designed for potential deployment on cloud platforms or edge devices).


### Sponsor Technologies Used:

- ✅ **Fluvio:** Fluvio is a mandatory component of this project, used for real-time data streaming. The Flask application (`app.py`) acts as a Fluvio producer, publishing structured data (including frame number, overall status, person count, and the detailed density grid) to the `crowd-data` topic for each processed frame or image. A separate Python script (`predict_stampede.py`) acts as a Fluvio consumer, subscribing to the `crowd-data` topic to receive and process this stream of real-time crowd metrics. This architecture demonstrates a decoupled approach where the analysis results are available for immediate consumption by other services or monitoring tools, enhancing the system's flexibility and scalability.


---


## ✨ Key Features


Highlight the most important features of your project:


- ✅ **Real-time Live Analysis:** Provides a live video feed from a selected camera source, analyzing crowd density and predicting risk in real-time with visual overlays.
- ✅ **Media File Analysis:** Allows users to upload image or video files for detailed, frame-by-frame or image analysis of crowd density and stampede risk.
- ✅ **AI-Powered Person Detection:** Leverages the efficient YOLOv11 Nano deep learning model for accurate identification and localization of individuals in the crowd.
- ✅ **Granular Density Grid Analysis:** Divides the frame into a grid and calculates the person count within each cell, providing a detailed spatial understanding of crowd distribution.
- ✅ **Clear Risk Level Indication:** Displays an overall risk status (Normal, High Density Cell Detected, High Density Warning, Critical Density Cell Detected, CRITICAL RISK) based on the density grid analysis.
- ✅ **Real-time Person Count:** Shows the total number of detected persons in the current frame or image.
- ✅ **Visual Risk Overlays:** Highlights grid cells exceeding high or critical density thresholds with color-coded overlays on the processed media.
- ✅ **Critical Frame/Video Output:** For uploaded videos, the application identifies and displays the frame corresponding to the highest detected risk level and offers a download link for the full processed video with overlays.
- ✅ **Audio Alert Notification:** Plays an audible alert sound on the results page when the analysis of an uploaded file indicates a critical risk level.
- ✅ **Fluvio Data Streaming:** Publishes detailed crowd analysis data for each processed frame/image to a Fluvio topic, enabling real-time monitoring and integration with other data processing pipelines.


Add images, GIFs, or screenshots if helpful!

*(Consider adding screenshots of the `index.html`, `live.html`, and `results.html` pages here)*


---


## 📽️ Demo & Deliverables


- **Demo Video Link:** [Paste YouTube or Loom link here]

- **Pitch Deck / PPT Link:** [Paste Google Slides / PDF link here]


---


## ✅ Tasks & Bonus Checklist


- [ ] **All members of the team completed the mandatory task - Followed at least 2 of our social channels and filled the form** (Details in Participant Manual)

- [ ] **All members of the team completed Bonus Task 1 - Sharing of Badges and filled the form (2 points)** (Details in Participant Manual)

- [ ] **All members of the team completed Bonus Task 2 - Signing up for Sprint.dev and filled the form (3 points)** (Details in Participant Manual)


*(Mark with ✅ if completed)*


---


## 🧪 How to Run the Project


### Requirements:

- Operating System: Linux or macOS (Recommended)
- Python 3.7+
- pip (Python package installer)
- Fluvio Cluster (Running locally or accessible)
- OpenCV library and its dependencies (usually installed with `opencv-python`)
- YOLO model file (`yolo11n.pt`) - The `ultralytics` library will attempt to download this automatically on the first run if it's not found locally.


### Local Setup:


1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-team/project-name.git](https://github.com/your-team/project-name.git)
    cd project-name
    ```

2.  **Set up a Python Virtual Environment (Recommended):**
    This isolates project dependencies from your system Python environment.

    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Install all necessary Python packages listed in `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```
    This command will install `Flask`, `opencv-python`, `ultralytics`, `numpy`, `fluvio`, `werkzeug`, `mimetypes`, and `shutil`.

4.  **Ensure Fluvio Cluster is Running:**
    The application requires a running Fluvio cluster to stream data. If you don't have Fluvio installed, follow the official Fluvio installation guide. Once installed, start a local cluster:

    ```bash
    fluvio cluster start
    ```
    Verify the `crowd-data` topic exists. If not, create it:

    ```bash
    fluvio topic create crowd-data
    ```

5.  **Run the Fluvio Consumer:**
    Open a **new terminal window** separate from where you will run the Flask app. Activate the virtual environment (steps from step 2). Then run the consumer script:

    ```bash
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate

    python predict_stampede.py
    ```
    This script will connect to your Fluvio cluster and subscribe to the `crowd-data` topic. It is configured to start reading from the end of the topic (`Offset.from_end(0)`), meaning it will only display messages produced *after* the consumer starts running. You will see analysis data printed to this terminal as the Flask app processes media.

6.  **Run the Flask Web Application:**
    Open a **third terminal window** (or reuse one if you don't need the consumer output visible simultaneously). Activate the virtual environment (steps from step 2). Then run the Flask application:

    ```bash
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate

    python app.py
    ```
    The application will start the Flask development server, typically hosting on `http://127.0.0.1:5000/`. It will attempt to load the YOLO model and connect to Fluvio during its startup process. Check the console output for confirmation of successful loading and connection.

7.  **Access the Web Application:**
    Open your preferred web browser and navigate to the application's address:

    * **Main Page (Upload):** `http://127.0.0.1:5000/`
    * **Live Feed Page:** `http://127.0.0.1:5000/live` (You can specify a camera index as a query parameter, e.g., `http://127.0.0.1:5000/live?camera=1`, if you have multiple webcams connected).

    Use the web interface to upload media files or start the live webcam analysis. Observe the output in your browser and, if running, the data stream in the `predict_stampede.py` terminal.


---


## 🧬 Future Scope


List improvements, extensions, or follow-up features:


- 📈 **Advanced Fluvio Consumer Analytics:** Enhance the `predict_stampede.py` script or create a separate service to perform more complex analysis on the Fluvio data stream, such as long-term trend analysis, anomaly detection, or integrating with databases for historical data storage and querying.
- 🛡️ **Model Selection and Management:** Implement functionality within the web application to allow users to select different pre-trained YOLO models or even upload custom-trained models.
- 🌐 **Dockerization and Cloud Deployment:** Create Docker containers for the Flask application and the Fluvio consumer to simplify deployment and scaling on cloud platforms (e.g., AWS EC2, Google Cloud Run, Azure App Services).
- 📊 **Dedicated Monitoring Dashboard:** Develop a separate, more sophisticated dashboard application that consumes the Fluvio data stream and provides rich visualizations, alerts, and historical data analysis capabilities.
- 🔊 **Configurable Alerting System:** Build a system to trigger alerts via multiple channels (e.g., email, SMS, push notifications) based on configurable risk thresholds and rules defined by the user.
- 📱 **Enhanced Mobile Responsiveness and Native App:** Improve the web application's responsiveness for a better mobile experience or explore developing a native mobile application.
- ⚙️ **Performance Optimization and Edge Deployment:** Investigate techniques for optimizing the AI model inference speed and explore deployment strategies for resource-constrained edge devices.
- 🚧 **Crowd Behavior Analysis:** Extend the analysis beyond just density to include crowd flow, movement patterns, and unusual behaviors that might indicate potential issues.


---


## 📎 Resources / Credits


- Flask: The Python micro web framework used for the application backend.
- OpenCV: Essential library for computer vision tasks, including video processing and image manipulation.
- Ultralytics: Provides the easy-to-use implementation for the YOLO object detection models.
- NumPy: Fundamental package for scientific computing with Python, used for numerical operations on image data.
- Fluvio: The real-time data streaming platform used for decoupling analysis data from the web application.
- YOLOv11 Nano: The specific object detection model used for person detection due to its balance of speed and performance.


---


## 🏁 Final Words


Share your hackathon journey — challenges, learnings, fun moments, or shout-outs!

*(Add your team's personal touch here!)*


---
