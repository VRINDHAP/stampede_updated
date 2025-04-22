# Required Imports
from flask import Flask, render_template, request, url_for, Response, stream_with_context, redirect, send_from_directory, jsonify
import os
import cv2 # OpenCV
# import tensorflow as tf -> No longer needed for TF Hub model
# import tensorflow_hub as hub -> No longer needed
import numpy as np
import sys
import time
from fluvio import Fluvio # Assuming you still want Fluvio integration [cite: 1]
import json
from werkzeug.utils import secure_filename
import mimetypes
import shutil
from ultralytics import YOLO # Import YOLO from ultralytics [cite: 1]
from queue import Queue # For sharing status between threads/generators [cite: 1]
import threading # For thread safety with shared status [cite: 1]

# --- Flask Application Setup --- [cite: 2]
app = Flask(__name__)

# --- Configuration for Folders --- [cite: 2]
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
PROCESSED_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'processed_frames')
PROCESSED_VIDEO_FOLDER = os.path.join(STATIC_FOLDER, 'processed_videos')
PROCESSED_IMAGE_FOLDER = os.path.join(STATIC_FOLDER, 'processed_images')
DEBUG_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'debug_frames')

for folder in [UPLOAD_FOLDER, PROCESSED_FRAMES_FOLDER, PROCESSED_VIDEO_FOLDER, PROCESSED_IMAGE_FOLDER, DEBUG_FRAMES_FOLDER]: # [cite: 2]
    if not os.path.exists(folder):
        os.makedirs(folder)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_VIDEO_FOLDER'] = PROCESSED_VIDEO_FOLDER

# --- Load Machine Learning Model ---
# Using YOLOv11 Nano (yolo11n.pt) - fast and lightweight. [cite: 2]
# You can change to yolov8s.pt, yolov8m.pt, yolo11n.pt etc. for higher accuracy but slower speed. [cite: 3]
MODEL_PATH = "yolo11n.pt" # [cite: 4]
yolo_model = None
try:
    print(f"Loading YOLO model from: {MODEL_PATH}...") # [cite: 4]
    yolo_model = YOLO(MODEL_PATH)
    # Perform a dummy prediction to fully initialize the model (optional but good practice)
    _ = yolo_model.predict(np.zeros((640, 640, 3)), verbose=False) # [cite: 4]
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load the YOLO model ({MODEL_PATH}): {e}") # [cite: 4]
    print("Ensure 'ultralytics' is installed (`pip install ultralytics`) and the model file exists.")
    yolo_model = None

# --- Model Specific Settings ---
# For COCO dataset used by standard YOLO models, 'person' is usually class index 0 [cite: 5]
PERSON_CLASS_INDEX = 0
DETECTION_THRESHOLD = 0.02125 # Confidence threshold for YOLO detections (adjust as needed) [cite: 5]

# --- Density Analysis Settings (remain the same) --- [cite: 5]
HIGH_DENSITY_THRESHOLD = 5
CRITICAL_DENSITY_THRESHOLD = 8
HIGH_DENSITY_CELL_COUNT_THRESHOLD = 3
CRITICAL_DENSITY_CELL_COUNT_THRESHOLD = 2
GRID_ROWS = 8
GRID_COLS = 8
STATUS_HIERARCHY = {
    "Normal": 0, "High Density Cell Detected": 1, "High Density Warning": 2,
    "Critical Density Cell Detected": 3, "CRITICAL RISK": 4,
    "Processing Started": -2, "Analysis Incomplete": -1, "Analysis Incomplete (No Content)": -1,
    "Analysis Incomplete (Invalid Content)": -1, "Analysis Incomplete (Tiny Frame)": -1,
    "Error: Model Not Loaded": -10, "Error: Could not open input video": -5, # [cite: 6]
    "Error: Failed to initialize VideoWriter": -5, "Error: Video writing failed": -5,
    "Error: Output video generation failed": -5, "Error: Image processing failed": -5,
    "Error: Unsupported file type": -5, "Error: Unexpected failure during video processing": -6,
    "Error: Invalid video dimensions": -5, "Error: Processing Error": -7, # Added generic processing error
    "Error: TF Resource Exhausted": -8 # Keep just in case, though less likely now [cite: 6]
}


# --- Fluvio Settings (Optional - Keep if needed) --- [cite: 7]
FLUVIO_CROWD_TOPIC = "crowd-data"
fluvio_client = None
fluvio_producer = None
# --- Fluvio connect/send functions (keep as before if using Fluvio) --- [cite: 7]
def connect_fluvio():
    global fluvio_client, fluvio_producer
    if fluvio_producer: return True
    print("Attempting to connect to Fluvio...")
    sys.stdout.flush()
    try:
        fluvio_client = Fluvio.connect() # [cite: 8]
        print("Fluvio client connected.")
        fluvio_producer = fluvio_client.topic_producer(FLUVIO_CROWD_TOPIC)
        print(f"Fluvio producer ready for topic '{FLUVIO_CROWD_TOPIC}'.")
        sys.stdout.flush()
        return True # [cite: 8]
    except Exception as e:
        print(f"!!! FLUVIO ERROR: {e}") # [cite: 9]
        fluvio_client = None
        fluvio_producer = None
        return False

def send_to_fluvio(key, data_dict):
    global fluvio_producer
    if not fluvio_producer: return
    try:
        key_bytes = str(key).encode('utf-8')
        data_json_str = json.dumps(data_dict)
        data_bytes = data_json_str.encode('utf-8')
        fluvio_producer.send(key_bytes, data_bytes)
    except Exception as e:
        print(f"!!! FLUVIO WARNING sending data (Key: {key}): {e}") # [cite: 10]

# --- Shared State for Live Status Updates (SSE) ---
# Using a simple dictionary with a lock for basic thread safety [cite: 10]
live_status_lock = threading.Lock()
live_status_data = {"status": "Initializing", "persons": 0}
# Using a Queue to signal updates to the SSE generator
status_update_queue = Queue()

# --- Helper Functions ---
def analyze_density_grid(density_grid):
    """Analyzes the grid to determine status and risky cells."""
    # (Keep this function exactly as in the previous version) [cite: 10]
    high_density_cells = 0
    critical_density_cells = 0
    risky_cell_coords = [] # List of (row, col) for cells needing overlay [cite: 11]
    overall_status = "Normal"
    total_people_in_grid = 0

    if not isinstance(density_grid, list) or len(density_grid) != GRID_ROWS: # [cite: 11]
        return "Analysis Incomplete (Invalid Grid)", risky_cell_coords, 0, 0, 0

    for r_idx, row in enumerate(density_grid):
        if not isinstance(row, list) or len(row) != GRID_COLS: continue
        for c_idx, count in enumerate(row):
            try:
                 person_count = int(count) # [cite: 12]
                 total_people_in_grid += person_count
                 if person_count >= CRITICAL_DENSITY_THRESHOLD:
                     critical_density_cells += 1
                     risky_cell_coords.append((r_idx, c_idx))
                 elif person_count >= HIGH_DENSITY_THRESHOLD: # [cite: 13]
                     high_density_cells += 1
                     risky_cell_coords.append((r_idx, c_idx)) # Also mark high density cells
            except (ValueError, TypeError):
                continue # Skip non-integer cell counts

    # Determine overall status based on cell counts and thresholds [cite: 14]
    if critical_density_cells >= CRITICAL_DENSITY_CELL_COUNT_THRESHOLD:
        overall_status = "CRITICAL RISK"
    elif critical_density_cells > 0:
        overall_status = "Critical Density Cell Detected"
    elif high_density_cells >= HIGH_DENSITY_CELL_COUNT_THRESHOLD:
        overall_status = "High Density Warning"
    elif high_density_cells > 0:
        overall_status = "High Density Cell Detected"

    return overall_status, risky_cell_coords, total_people_in_grid, high_density_cells, critical_density_cells


def get_higher_priority_status(status1, status2):
    """Compares two status strings and returns the one with higher priority.""" # [cite: 15]
    # (Keep this function exactly as in the previous version)
    p1 = STATUS_HIERARCHY.get(status1, -99) # [cite: 15]
    p2 = STATUS_HIERARCHY.get(status2, -99)
    return status1 if p1 >= p2 else status2

# --- Text Drawing Helper ---
def draw_text_with_bg(img, text, origin, font, scale, fg_color, bg_color, thickness, padding, bg_alpha):
    """Draws text with a semi-transparent background."""
    # (Keep this function exactly as in the previous version) [cite: 15]
    try:
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness) # [cite: 16]
        text_w, text_h = text_size
        x, y = origin

        # Adjust y-origin upwards based on text height/baseline for better positioning
        # Calculate rectangle coordinates
        rect_y1 = y - text_h - padding - baseline // 2 # [cite: 16]
        rect_y2 = y + padding - baseline // 2
        rect_x1 = x - padding
        rect_x2 = x + text_w + padding # [cite: 17]

        # Ensure rectangle coordinates are within image bounds
        rect_y1 = max(0, rect_y1); rect_x1 = max(0, rect_x1) # [cite: 18]
        rect_y2 = min(img.shape[0], rect_y2); rect_x2 = min(img.shape[1], rect_x2) # [cite: 19]

        if rect_y2 > rect_y1 and rect_x2 > rect_x1: # Check if rectangle has valid dimensions
            sub_img = img[rect_y1:rect_y2, rect_x1:rect_x2]
            if sub_img.shape[0] > 0 and sub_img.shape[1] > 0: # Ensure sub-image is valid
                bg_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                bg_rect[:] = bg_color # [cite: 20]
                res = cv2.addWeighted(sub_img, 1.0 - bg_alpha, bg_rect, bg_alpha, 0)
                img[rect_y1:rect_y2, rect_x1:rect_x2] = res
            else:
                 print(f"Warning: Invalid sub-image for text background at {origin}.")


        # Draw the text itself (adjust y slightly for baseline)
        cv2.putText(img, text, (x, y - baseline // 2), font, scale, fg_color, thickness, cv2.LINE_AA) # [cite: 21]
    except Exception as e:
        print(f"Error drawing text '{text}' at {origin}: {e}")


# --- Frame/Image Processing Function (UPDATED FOR YOLO) ---
def process_media_content(content, content_width, content_height, frame_or_image_index, is_live_stream=False):
    """
    Processes a single image or video frame using YOLO: detects people, calculates density,
    determines status, draws overlays/text, sends data to Fluvio, and updates live status.
    Returns the processed content, frame status, and person count.
    """ # [cite: 22]
    global live_status_data # Access the shared status dictionary

    if content is None:
        return None, "Analysis Incomplete (No Content)", 0

    if yolo_model is None:
         # Draw error message directly on a blank frame if model is missing
         error_frame = np.zeros((content_height or 480, content_width or 640, 3), dtype=np.uint8)
         draw_text_with_bg(error_frame, "MODEL NOT LOADED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), (0,0,0), 2, 5, 0.7) # [cite: 23]
         print(f"!!! Skipping processing for index {frame_or_image_index}: YOLO model not loaded.")
         # Update live status if applicable
         if is_live_stream:
            with live_status_lock:
                live_status_data["status"] = "Error: Model Not Loaded"
                live_status_data["persons"] = 0 # [cite: 24]
            status_update_queue.put(True) # Signal update
         return error_frame, "Error: Model Not Loaded", 0

    start_process_time = time.time()
    processed_content = content.copy() # Work on a copy

    content_status = "Analysis Incomplete" # Default status for this frame/image
    confirmed_person_count_this_content = 0
    density_grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    risky_coords = []
    high_cells = 0
    crit_cells = 0 # [cite: 25]

    try:
        # --- 1. YOLO Detection ---
        # YOLOv8 typically expects BGR images directly from OpenCV
        # The model handles resizing and normalization internally. [cite: 25]
        results = yolo_model.predict(source=content, verbose=False, conf=DETECTION_THRESHOLD, classes=[PERSON_CLASS_INDEX]) # [cite: 26]

        # Results is a list, usually with one element for one image
        if results and results[0]:
            detected_boxes = results[0].boxes.xyxy.cpu().numpy() # Bounding boxes (xmin, ymin, xmax, ymax)
            # confidences = results[0].boxes.conf.cpu().numpy() # Confidence scores
            # classes = results[0].boxes.cls.cpu().numpy() # Class indices

            confirmed_person_count_this_content = len(detected_boxes) # [cite: 27]
        else:
            confirmed_person_count_this_content = 0
            detected_boxes = []


        # --- 2. Calculate Grid Density ---
        cell_height = content_height // GRID_ROWS
        cell_width = content_width // GRID_COLS

        if cell_height <= 0 or cell_width <= 0:
             print(f"Warning: Content dimensions ({content_width}x{content_height}) too small for grid. Skipping density.") # [cite: 28]
             content_status = "Analysis Incomplete (Tiny Frame)"
        else:
             # Populate density grid based on center point of detected boxes
             for box in detected_boxes:
                 xmin, ymin, xmax, ymax = box # [cite: 29]
                 # Calculate center point
                 center_x = int((xmin + xmax) / 2)
                 center_y = int((ymin + ymax) / 2)
                 # Clamp coordinates to be within image bounds before calculating grid cell
                 center_x = max(0, min(center_x, content_width - 1)) # [cite: 30]
                 center_y = max(0, min(center_y, content_height - 1))

                 row = min(max(0, center_y // cell_height), GRID_ROWS - 1)
                 col = min(max(0, center_x // cell_width), GRID_COLS - 1)
                 density_grid[row][col] += 1 # [cite: 31]

             # --- 3. Analyze Density Grid ---
             content_status, risky_coords, _, high_cells, crit_cells = analyze_density_grid(density_grid) # [cite: 31]


        # --- 4. Send Data to Fluvio (Optional) ---
        if fluvio_producer: # Check if Fluvio is active [cite: 7]
            fluvio_payload = {
                 "timestamp": int(time.time()), "frame": frame_or_image_index, "density_grid": density_grid, # [cite: 32]
                 "frame_status": content_status, "confirmed_persons": confirmed_person_count_this_content,
                 "high_density_cells": high_cells, "critical_density_cells": crit_cells
            } # [cite: 32]
            send_to_fluvio(f"media-{frame_or_image_index}", fluvio_payload) # [cite: 10]

        # --- 5. Update Live Status (if processing for live stream) ---
        if is_live_stream: # [cite: 33]
            with live_status_lock:
                live_status_data["status"] = content_status
                live_status_data["persons"] = confirmed_person_count_this_content
            status_update_queue.put(True) # Signal that new data is available [cite: 33]


        # --- 6. Draw Overlays and Text ---
        # Draw overlays for risky grid cells (same logic as before) [cite: 34]
        overlay_alpha = 0.4
        overlay_color_critical = (0, 0, 255) # Red (BGR)
        overlay_color_high = (0, 165, 255) # Orange (BGR) [cite: 34]

        if cell_height > 0 and cell_width > 0:
             overlay = processed_content.copy()
             for r, c in risky_coords:
                 cell_y_start = r * cell_height # [cite: 35]
                 cell_y_end = (r + 1) * cell_height
                 cell_x_start = c * cell_width
                 cell_x_end = (c + 1) * cell_width
                 color = overlay_color_critical if density_grid[r][c] >= CRITICAL_DENSITY_THRESHOLD else overlay_color_high # [cite: 36]
                 cv2.rectangle(overlay, (cell_x_start, cell_y_start), (cell_x_end, cell_y_end), color, -1) # [cite: 36]
             cv2.addWeighted(overlay, overlay_alpha, processed_content, 1 - overlay_alpha, 0, processed_content)

        # Determine colors and text
        frame_display_status = content_status
        status_color = (0, 128, 0); # Default Green [cite: 37]
        if "CRITICAL" in frame_display_status: status_color = (0, 0, 255) # [cite: 37]
        elif "Warning" in frame_display_status or "High" in frame_display_status or "Detected" in frame_display_status: status_color = (0, 165, 255) # [cite: 37]
        elif "Error" in frame_display_status or "Incomplete" in frame_display_status: status_color = (0, 0, 255) # Red

        if "CRITICAL" in frame_display_status: chance_text, chance_color = "Stampede Chance: Critical", (0, 0, 255) # [cite: 37]
        elif "Warning" in frame_display_status or "High" in frame_display_status or "Detected" in frame_display_status: chance_text, chance_color = "Stampede Chance: High", (0, 165, 255) # [cite: 38]
        else: chance_text, chance_color = "Stampede Chance: Low", (0, 128, 0) # [cite: 38]

        status_text = f"Risk: {frame_display_status}"
        person_count_text = f"Persons: {confirmed_person_count_this_content}"

        # Draw text using helper function
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_padding = 5 # [cite: 39]
        bg_alpha = 0.6
        bg_color = (0, 0, 0) # Black background

        draw_text_with_bg(processed_content, status_text, (10, 20 + text_padding), font, font_scale, status_color, bg_color, font_thickness, text_padding, bg_alpha) # [cite: 15]
        (text_w_p, _), _ = cv2.getTextSize(person_count_text, font, font_scale, font_thickness) # [cite: 39]
        draw_text_with_bg(processed_content, person_count_text, (content_width - text_w_p - 10 - text_padding * 2, 20 + text_padding), font, font_scale, (255, 255, 255), bg_color, font_thickness, text_padding, bg_alpha)
        (text_w_c, text_h_c), baseline_c = cv2.getTextSize(chance_text, font, font_scale, font_thickness) # [cite: 40]
        draw_text_with_bg(processed_content, chance_text, (10, content_height - 10 - baseline_c), font, font_scale, chance_color, bg_color, font_thickness, text_padding, bg_alpha)


    except Exception as e:
        print(f"!!! UNEXPECTED ERROR during process_media_content (YOLO) for index {frame_or_image_index}: {e}") # [cite: 41]
        # Draw generic error on frame
        error_frame = content.copy() if content is not None else np.zeros((content_height or 480, content_width or 640, 3), dtype=np.uint8)
        err_msg = f"Processing Error: {type(e).__name__}" # [cite: 41]
        draw_text_with_bg(error_frame, err_msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), (0,0,0), 2, 5, 0.7)
        content_status = "Error: Processing Error"
        confirmed_person_count_this_content = 0
        # Update live status if applicable [cite: 42]
        if is_live_stream:
            with live_status_lock:
                live_status_data["status"] = content_status
                live_status_data["persons"] = 0
            status_update_queue.put(True) # Signal update
        return error_frame, content_status, confirmed_person_count_this_content


    # Optional: Log processing time [cite: 43]
    # end_process_time = time.time()
    # print(f" -> Index {frame_or_image_index} processed in {end_process_time - start_process_time:.3f}s. Status: {content_status}") [cite: 44]

    return processed_content, content_status, confirmed_person_count_this_content


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index_route():
    """Serves the main upload page (index.html)."""
    return render_template('index.html')

@app.route('/upload_media', methods=['POST'])
def upload_media_route():
    """Handles media uploads, processes with YOLO, and shows results."""
    # (This route remains largely the same as the previous version,
    #  but calls the updated process_media_content function.
    #  Key changes are checking yolo_model instead of detector,
    #  and the process_media_content call itself.)
    print("\n--- Request received for /upload_media ---") # [cite: 44]
    # Optional: Reconnect Fluvio if needed [cite: 45]
    connect_fluvio()

    if yolo_model is None: # Check if YOLO model loaded [cite: 45]
        print("!!! ERROR: YOLO model not loaded. Cannot process media.")
        # Render results page with error
        return render_template('results.html', prediction_status="Error: Model Not Loaded") # [cite: 45]

    start_time = time.time()

    # --- File Handling (same as before) ---
    if 'media' not in request.files: return 'No media file part', 400 # [cite: 45]
    media_file = request.files['media'] # [cite: 46]
    if media_file.filename == '': return 'No selected file', 400 # [cite: 46]
    original_filename = secure_filename(media_file.filename) # [cite: 46]
    unique_filename_prefix = f"{int(time.time())}_{os.urandom(4).hex()}" # [cite: 46]
    unique_original_filename = f"{unique_filename_prefix}_{original_filename}" # [cite: 46]
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_original_filename) # [cite: 46]
    try:
        media_file.save(upload_path)
        print(f"Media saved to: {upload_path}")
    except Exception as e:
        print(f"Error saving media file: {e}")
        return f"Error saving file: {e}", 500

    mimetype = mimetypes.guess_type(upload_path)[0] # [cite: 47]
    file_type = 'unknown'
    if mimetype:
        if mimetype.startswith('video/'): file_type = 'video' # [cite: 47]
        elif mimetype.startswith('image/'): file_type = 'image' # [cite: 47]
    print(f"Detected file type: {file_type}")

    # --- Initialize vars ---
    processed_media_url = None
    download_video_url = None
    overall_processing_status = "Processing Started"
    max_persons = 0
    output_media_type = file_type

    # --- Process Video ---
    if file_type == 'video':
        base_name = os.path.splitext(f"processed_{unique_original_filename}")[0] # [cite: 48]
        output_video_filename = f"{base_name}.mp4"
        output_video_path = os.path.join(PROCESSED_VIDEO_FOLDER, output_video_filename) # [cite: 48]
        cap = None
        out_video = None

        try:
            cap = cv2.VideoCapture(upload_path)
            if not cap.isOpened():
                 raise IOError(f"Failed to open input video file: {upload_path}") # [cite: 49]

            fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps and fps > 0 else 30.0 # [cite: 50]
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # [cite: 50]
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # [cite: 50]

            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid video dimensions ({width}x{height})") # [cite: 50]

            print(f"Video Input: {width}x{height} @ {fps:.2f} FPS")

            # --- Codec Selection (same as before) --- [cite: 51]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Default
            try:
                test_output_path = os.path.join(app.config['PROCESSED_VIDEO_FOLDER'], f"codec_test_{unique_filename_prefix}.mp4") # [cite: 51]
                test_writer = cv2.VideoWriter()
                if test_writer.open(test_output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height), True): # [cite: 52]
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    print("Using avc1 (H.264) codec.")
                else:
                    print("avc1 codec not available, using mp4v.")
                test_writer.release() # [cite: 53]
                if os.path.exists(test_output_path): os.remove(test_output_path)
            except Exception as codec_e:
                print(f"Codec test failed ('avc1'): {codec_e}. Using mp4v.") # [cite: 54]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # --- End Codec Selection ---

            out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height)) # [cite: 54]
            if not out_video.isOpened():
                raise IOError(f"Failed to initialize VideoWriter for {output_video_path}") # [cite: 54]

            print("VideoWriter opened. Processing frames...") # [cite: 55]
            frame_num = 0
            video_highest_status = "Normal"
            critical_frame_content_to_save = None
            first_processed_frame_content = None # [cite: 55]

            # --- Process first frame (same logic as before) ---
            ret_first, first_frame = cap.read() # [cite: 55]
            if ret_first and first_frame is not None: # [cite: 56]
                # Call the *updated* processing function
                first_processed_frame_content, first_frame_status, first_persons = process_media_content(
                    first_frame, width, height, -1, is_live_stream=False # Indicate not live stream [cite: 56]
                )
                if first_processed_frame_content is not None: # [cite: 57]
                    video_highest_status = first_frame_status
                    critical_frame_content_to_save = first_processed_frame_content.copy()
                    max_persons = first_persons
                print(f"Processed initial frame (Status: {first_frame_status})") # [cite: 58]
            else:
                print("Warning: Could not read the first frame.")
            # --- End Process first frame ---

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset for main loop [cite: 58]
            frame_num = 0

            while True: # [cite: 59]
                ret, frame = cap.read()
                if not ret or frame is None: break

                # Call the *updated* processing function
                processed_frame, current_frame_status, people_count = process_media_content(
                     frame, width, height, frame_num, is_live_stream=False # Indicate not live stream [cite: 60]
                )

                if processed_frame is None:
                    print(f"Warning: Skipping frame {frame_num} due to processing failure.")
                    frame_num += 1 # [cite: 61]
                    continue

                # Update highest status logic (same as before)
                new_highest_status = get_higher_priority_status(video_highest_status, current_frame_status) # [cite: 15]
                if STATUS_HIERARCHY.get(new_highest_status, -99) > STATUS_HIERARCHY.get(video_highest_status, -99): # [cite: 61]
                     video_highest_status = new_highest_status # [cite: 62]
                     critical_frame_content_to_save = processed_frame.copy()
                     print(f"New highest status '{video_highest_status}' at frame {frame_num}.")

                max_persons = max(max_persons, people_count) # Update max persons

                # Write frame (same as before) [cite: 63]
                try:
                    out_video.write(processed_frame) # [cite: 63]
                except Exception as write_e:
                    print(f"!!! ERROR writing frame {frame_num}: {write_e}") # [cite: 64]
                    overall_processing_status = get_higher_priority_status(overall_processing_status, f"Error: Video writing failed frame {frame_num}") # [cite: 15]
                    break

                frame_num += 1
                # if frame_num % 50 == 0: print(f"  Processed frame {frame_num}...")

            print(f"Finished processing {frame_num} frames.") # [cite: 65]
            overall_processing_status = video_highest_status # Final status

        except Exception as video_proc_e:
            print(f"!!! ERROR during video processing: {video_proc_e}") # [cite: 65]
            overall_processing_status = get_higher_priority_status(overall_processing_status, f"Error: Video processing failed - {type(video_proc_e).__name__}") # [cite: 15]
            processed_media_url = None
            download_video_url = None # [cite: 66]
            # Ensure video writer is released on error
            if out_video and out_video.isOpened(): out_video.release()
            # Clean up partial file
            if os.path.exists(output_video_path):
                try: os.remove(output_video_path)
                except OSError: pass # [cite: 67]

        finally:
            if cap and cap.isOpened(): cap.release(); print("VideoCapture released.") # [cite: 67, 68]
            if out_video and out_video.isOpened(): out_video.release(); print("VideoWriter released.") # [cite: 68, 69]

        # --- Save critical frame (same logic as before) ---
        if overall_processing_status.startswith("Error:") is False: # Only save if no major error occurred
            frame_to_save_display = critical_frame_content_to_save if critical_frame_content_to_save is not None else first_processed_frame_content
            if frame_to_save_display is not None:
                critical_frame_filename = f"display_{base_name}.jpg"
                critical_frame_path = os.path.join(PROCESSED_FRAMES_FOLDER, critical_frame_filename) # [cite: 70]
                try:
                    cv2.imwrite(critical_frame_path, frame_to_save_display)
                    processed_media_url = url_for('static', filename=f'processed_frames/{critical_frame_filename}') # [cite: 70]
                    print(f"Saved display frame to {critical_frame_path}")
                except Exception as img_save_e: # [cite: 71]
                     print(f"!!! ERROR saving display frame image: {img_save_e}")
            else:
                print("!!! WARNING: No frame content to save as display image.") # [cite: 72]

        # --- Check output video (same logic as before) ---
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 1024: # [cite: 72]
            download_video_url = url_for('static', filename=f'processed_videos/{output_video_filename}')
            print(f"Full processed video available: {output_video_path}")
        else:
             # Only flag as error if no other major error already occurred [cite: 73]
            if not overall_processing_status.startswith("Error:"):
                 print(f"!!! WARNING: Output video file missing or empty: {output_video_path}") # [cite: 73]
                 overall_processing_status = get_higher_priority_status(overall_processing_status, "Error: Output video generation failed") # [cite: 15]
            download_video_url = None
            if os.path.exists(output_video_path): # Clean up empty file
                 try: os.remove(output_video_path) # [cite: 74]
                 except OSError: pass
        # --- End Process Video ---

    # --- Process Image ---
    elif file_type == 'image':
        # (This section remains largely the same, just calls the updated process_media_content)
        original_ext = os.path.splitext(original_filename)[1].lower()
        if original_ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']: original_ext = '.jpg' # [cite: 75]
        base_name = os.path.splitext(f"processed_{unique_original_filename}")[0]
        output_image_filename = f"{base_name}{original_ext}" # [cite: 75]
        output_image_path = os.path.join(PROCESSED_IMAGE_FOLDER, output_image_filename) # [cite: 75]

        try:
            image = cv2.imread(upload_path)
            if image is None: raise ValueError("Could not read image file.") # [cite: 75]
            height, width = image.shape[:2]
            print(f"Image Input: {width}x{height}") # [cite: 76]

            # Call the *updated* processing function
            processed_image, image_status, people_count = process_media_content(
                image, width, height, 0, is_live_stream=False # Indicate not live stream [cite: 76]
            )

            if processed_image is None: raise ValueError("Processing returned None.") # [cite: 76]


            overall_processing_status = image_status # [cite: 77]
            max_persons = people_count

            save_success = cv2.imwrite(output_image_path, processed_image) # [cite: 77]
            if not save_success: raise ValueError("Failed to save processed image.")

            print(f"Processed image saved to: {output_image_path}")
            processed_media_url = url_for('static', filename=f'processed_images/{output_image_filename}')
            download_video_url = None # [cite: 78]

        except Exception as img_proc_e:
           print(f"!!! ERROR during image processing: {img_proc_e}") # [cite: 79]
           overall_processing_status = get_higher_priority_status(overall_processing_status, f"Error: Image processing failed - {type(img_proc_e).__name__}") # [cite: 15]
           processed_media_url = None
           download_video_url = None
        # --- End Process Image ---

    # --- Handle Unknown File Type (same as before) ---
    else:
        print(f"Unsupported file type: {mimetype or 'Unknown'}")
        overall_processing_status = "Error: Unsupported file type" # [cite: 80]
        processed_media_url = None
        download_video_url = None

    # --- Cleanup Upload (same as before) ---
    try:
        if os.path.exists(upload_path):
            os.remove(upload_path)
            print(f"Removed temporary upload: {upload_path}")
    except OSError as e:
        print(f"Warning: Could not remove temporary upload {upload_path}: {e}")

    processing_time_secs = time.time() - start_time # [cite: 81]
    print(f"Total upload processing time: {processing_time_secs:.2f} seconds")

    # --- Render Results (same as before) ---
    print(f"---> Rendering results page:") # [cite: 81]
    print(f"     Media Type: {output_media_type}")
    print(f"     Displayed Media URL: {processed_media_url}")
    print(f"     Download Video URL: {download_video_url}")
    print(f"     Overall Status: {overall_processing_status}")
    print(f"     Max Persons Detected: {max_persons}")

    return render_template('results.html',
                           output_media_type=output_media_type, # [cite: 82]
                           processed_media_url=processed_media_url,
                           download_video_url=download_video_url,
                           prediction_status=overall_processing_status, # [cite: 83]
                           max_persons=max_persons,
                           processing_time=f"{processing_time_secs:.2f}")


# --- Live Stream Video Feed Route (UPDATED FOR YOLO & Camera Index) ---
def generate_live_frames(camera_index=0): # Default to 0 [cite: 84]
    """Generator for processing and yielding live video frames using YOLO."""
    print(f"\n--- Request received for /video_feed (Live Stream) with camera index: {camera_index} ---") # Log the index
    connect_fluvio() # [cite: 83]

    live_cap = None

    # Helper to yield error frame (same as before) [cite: 84]
    def yield_error_frame(message="Error", width=640, height=480):
        blank_img = np.zeros((height, width, 3), dtype=np.uint8)
        font_scale = min(width, height) / 600.0
        (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        text_x = max(10, int(width/2 - tw/2)) # Ensure text starts within bounds
        text_y = int(height/2 + th/2)
        draw_text_with_bg(blank_img, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), (0,0,0), 2, 5, 0.7) # [cite: 15]
        ret_enc, buffer = cv2.imencode('.jpg', blank_img) # [cite: 85]
        if ret_enc:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n') # [cite: 85]


    # Check if YOLO model is loaded (remains the same) [cite: 85]
    if yolo_model is None: # [cite: 86]
        print("!!! ERROR: YOLO model not loaded. Cannot start live stream.") # [cite: 87]
        # Update shared status for SSE
        with live_status_lock: # [cite: 87]
            live_status_data["status"] = "Error: Model Not Loaded"
            live_status_data["persons"] = 0
        status_update_queue.put(True) # Signal update [cite: 87]
        yield from yield_error_frame("ML Model Load Failed")
        return

    # --- Open Video Source (Use camera_index) --- [cite: 88]
    try:
        print(f"Attempting to open camera index: {camera_index}") # Log attempt
        live_cap = cv2.VideoCapture(camera_index) # Use the passed index [cite: 88]
        source_desc = f"webcam ({camera_index})"
        if not live_cap.isOpened():
            # Fallback logic remains similar, but report specific index failure
            print(f"Warning: Cannot open {source_desc}. Trying fallback video...") # [cite: 88]
            fallback_video_path = "videoplayback.mp4" # [cite: 88]
            if os.path.exists(fallback_video_path):
                 live_cap = cv2.VideoCapture(fallback_video_path) # [cite: 89]
                 source_desc = f"fallback video ({fallback_video_path})"
                 if live_cap.isOpened(): print(f"Using {source_desc}")
                 else: raise IOError(f"Failed to open primary camera ({camera_index}) AND fallback video.") # [cite: 89]
            else:
                raise IOError(f"Failed to open camera index {camera_index} and fallback video not found.") # [cite: 89]

        frame_width = int(live_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # [cite: 90]
        frame_height = int(live_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # [cite: 90]
        if frame_width <= 0 or frame_height <= 0: # [cite: 90]
            # If dimensions are invalid, release and raise error
            if live_cap.isOpened(): live_cap.release()
            raise ValueError(f"Invalid dimensions from {source_desc} (Index: {camera_index})") # [cite: 90]

        print(f"Live stream started ({source_desc}): {frame_width}x{frame_height}")

        # --- Frame Processing Loop ---
        frame_num = 0
        while True:
            ret, frame = live_cap.read() # [cite: 91]
            if not ret or frame is None: # [cite: 91]
                # Handle end of video file (looping) or camera error
                is_file_source = live_cap.get(cv2.CAP_PROP_POS_AVI_RATIO)
                is_camera_source = not is_file_source or is_file_source == -1

                if is_file_source is not None and not is_camera_source and live_cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                    print("Looping video source...") # [cite: 92]
                    live_cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # [cite: 92]
                    continue # Go to next iteration to read the first frame again
                elif is_camera_source:
                     print(f"Error reading frame from camera index {camera_index}. Stream may have ended.")
                     yield from yield_error_frame(f"Camera {camera_index} Error")
                     break # Exit loop on camera read error
                else:
                    print("End of live stream source or read error.") # [cite: 93]
                    break # Exit loop


            # Call the *updated* processing function, indicating it's for live stream
            processed_frame, frame_status, people_count = process_media_content(
                frame, frame_width, frame_height, frame_num, is_live_stream=True # [cite: 93]
            ) # [cite: 94]

            if processed_frame is None:
                print(f"Warning: Skipping live frame {frame_num} processing failure.") # [cite: 94]
                frame_num += 1
                continue

            # Encode and yield frame (same as before)
            ret_enc, buffer = cv2.imencode('.jpg', processed_frame) # [cite: 95]
            if not ret_enc:
                print(f"Error encoding live frame {frame_num}.") # [cite: 95]
                frame_num += 1
                continue
            frame_bytes = buffer.tobytes() # [cite: 95]
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n') # [cite: 96]

            frame_num += 1
            # time.sleep(0.01) # Optional delay

    except Exception as live_err:
        print(f"!!! ERROR during live stream generation for index {camera_index}: {live_err}") # [cite: 97]
        err_msg = f"Stream Error: {type(live_err).__name__}"
        # Update shared status for SSE (remains the same)
        with live_status_lock: # [cite: 97]
            live_status_data["status"] = err_msg
            live_status_data["persons"] = 0
        status_update_queue.put(True) # Signal update [cite: 97]
        yield from yield_error_frame(err_msg) # Yield error frame to client
    finally:
        if live_cap and live_cap.isOpened(): # [cite: 98]
            live_cap.release()
        print(f"Live stream generator finished for index {camera_index}.")
         # Signal potential end of stream status update (remains the same)
        with live_status_lock: # [cite: 98]
            live_status_data["status"] = "Stream Ended"
            live_status_data["persons"] = 0
        status_update_queue.put(True) # [cite: 98]


@app.route('/live')
def live_route():
    """Serves the live stream page (live.html).""" # [cite: 99]
    connect_fluvio() # Optional [cite: 99]
    return render_template('live.html')

@app.route('/video_feed')
def video_feed_route():
    """Provides the MJPEG video stream, accepting a camera index."""
    try:
        # Get 'camera' query parameter, default to 0 if not present or invalid
        cam_index = int(request.args.get('camera', 0))
    except ValueError:
        cam_index = 0 # Default to 0 if conversion fails
    # Pass the index to the generator
    return Response(generate_live_frames(camera_index=cam_index),
                    mimetype='multipart/x-mixed-replace; boundary=frame') # [cite: 100]


# --- Server-Sent Events (SSE) Route for Live Status ---
def generate_status_updates():
    """Generator function to send live status updates via SSE."""
    print("SSE client connected for status updates.") # [cite: 100]
    last_sent_status = None # Keep track of last sent data to avoid redundant messages
    try:
        while True:
            # Block until a status update is signaled via the queue
            status_update_queue.get() # Wait for a signal [cite: 101]
            current_status = None
            with live_status_lock: # Get the latest status safely [cite: 101]
                current_status = live_status_data.copy()

            # Only send if the status or person count has changed
            if current_status != last_sent_status:
                 json_data = json.dumps(current_status) # [cite: 102]
                 yield f"data: {json_data}\n\n" # SSE format [cite: 102]
                 last_sent_status = current_status
                 # print(f"SSE Update Sent: {json_data}") # Debug log
            status_update_queue.task_done() # Mark queue item as processed [cite: 102]

            # Add a small sleep to prevent overly tight loops if many signals arrive quickly [cite: 103]
            time.sleep(0.1)

    except GeneratorExit:
         print("SSE client disconnected.") # [cite: 103]
    finally:
         print("SSE status update generator finished.")
         # Ensure queue is cleared if generator exits unexpectedly
         while not status_update_queue.empty(): # [cite: 103]
             try:
                 status_update_queue.get_nowait() # [cite: 104]
                 status_update_queue.task_done()
             except Exception:
                 break


@app.route('/stream_status')
def stream_status_route():
    """Provides the SSE stream for live status updates."""
    return Response(generate_status_updates(), mimetype='text/event-stream') # [cite: 104]


# --- Run Application ---
if __name__ == '__main__':
    print("--- Initializing Stampede Prediction Application (YOLOv11) ---") # [cite: 105]

    connect_fluvio() # Optional Fluvio connect on startup [cite: 105]

    if yolo_model is None:
        print("!!! WARNING: YOLO model failed to load. Processing will fail.") # [cite: 106]
    else:
        print("+++ YOLO model appears loaded.") # [cite: 106]

    print("--- Starting Flask Development Server ---")
    try:
        # Use debug=False for production
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True) # [cite: 106]
    except Exception as run_e:
        print(f"!!! FATAL ERROR: Flask application failed to start: {run_e}") # [cite: 106]
    finally:
        print("--- Application Shutting Down ---") # [cite: 107]