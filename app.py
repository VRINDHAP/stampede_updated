# app.py

# Required Imports
from flask import Flask, render_template, request, url_for, Response, stream_with_context, redirect, send_from_directory
import os
import cv2 # OpenCV
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sys
import time
from fluvio import Fluvio
import json
from werkzeug.utils import secure_filename
import mimetypes
import shutil

# --- Flask Application Setup ---
app = Flask(__name__)

# --- Configuration for Folders ---
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
PROCESSED_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'processed_frames') # Will store critical frames
PROCESSED_VIDEO_FOLDER = os.path.join(STATIC_FOLDER, 'processed_videos') # Will store full processed videos
PROCESSED_IMAGE_FOLDER = os.path.join(STATIC_FOLDER, 'processed_images') # Will store processed images
DEBUG_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'debug_frames') # Not actively used in final version, but kept

for folder in [UPLOAD_FOLDER, PROCESSED_FRAMES_FOLDER, PROCESSED_VIDEO_FOLDER, PROCESSED_IMAGE_FOLDER, DEBUG_FRAMES_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_VIDEO_FOLDER'] = PROCESSED_VIDEO_FOLDER # Add this for send_from_directory


# --- Load Machine Learning Model ---
DETECTOR_HANDLE = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
detector = None
try:
    print(f"Loading detection model from: {DETECTOR_HANDLE}...")
    # Suppress TF logging during model loading for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress info and warning messages
    detector = hub.load(DETECTOR_HANDLE)
    os.environ.pop('TF_CPP_MIN_LOG_LEVEL', None) # Restore logging
    print("Model loaded successfully.")
except Exception as e:
    os.environ.pop('TF_CPP_MIN_LOG_LEVEL', None) # Restore logging on error too
    print(f"FATAL ERROR: Could not load the detection model: {e}")
    print("Ensure you have internet access and the model URL is correct.")
    detector = None # Ensure detector is None if loading fails

# --- Model Specific Settings ---
PERSON_CLASS_INDEX = 1
DETECTION_THRESHOLD = 0.22 # Confidence threshold for detections

# --- Density Analysis Settings ---
HIGH_DENSITY_THRESHOLD = 7 # People count per cell to be considered 'High Density'
CRITICAL_DENSITY_THRESHOLD = 9 # People count per cell to be considered 'Critical Density'
HIGH_DENSITY_CELL_COUNT_THRESHOLD = 3 # Number of High Density cells to trigger 'High Density Warning'
CRITICAL_DENSITY_CELL_COUNT_THRESHOLD = 2 # Number of Critical Density cells to trigger 'CRITICAL RISK'
GRID_ROWS = 8
GRID_COLS = 8
# Hierarchy for status messages - higher number means higher priority
STATUS_HIERARCHY = {
    "Normal": 0,
    "High Density Cell Detected": 1,
    "High Density Warning": 2,
    "Critical Density Cell Detected": 3,
    "CRITICAL RISK": 4,
    # Error/Processing statuses (lower priority in comparison, but useful states)
    "Processing Started": -2,
    "Analysis Incomplete": -1,
    "Analysis Incomplete (Tiny Frame)": -1, # Specific incomplete status
    "Error: Model Not Loaded": -10, # Make model loading error very low priority
    "Error: Could not open input video": -5,
    "Error: Failed to initialize VideoWriter": -5,
    "Error: Video writing failed": -5,
    "Error: Output video generation failed": -5,
    "Error: Image processing failed": -5,
    "Error: Unsupported file type": -5,
    "Error: Unexpected failure during video processing": -6 # Generic failure error
}


# --- Fluvio Settings ---
FLUVIO_CROWD_TOPIC = "crowd-data"
# Global variables for Fluvio client and producer
fluvio_client = None
fluvio_producer = None

def connect_fluvio():
    """Attempts to connect to Fluvio and create a topic producer."""
    global fluvio_client, fluvio_producer # Declare intention to modify globals
    if fluvio_producer: # Already connected
         # print("Fluvio producer already initialized.") # Avoid spamming logs
         return True

    print("Attempting to connect to Fluvio...")
    sys.stdout.flush()
    try:
        # Assuming local default connection. Adjust if needed (e.g., Fluvio.connect("cloud_endpoint:9003"))
        fluvio_client = Fluvio.connect()
        print("Fluvio client connected successfully.")
        sys.stdout.flush()

        # Get a producer for the specified topic
        fluvio_producer = fluvio_client.topic_producer(FLUVIO_CROWD_TOPIC)
        print(f"Fluvio producer ready for topic '{FLUVIO_CROWD_TOPIC}'.")
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"!!! FLUVIO ERROR: Could not connect or get producer for topic '{FLUVIO_CROWD_TOPIC}'.")
        print(f"    Error details: {e}")
        print("    Check if Fluvio cluster is running and topic exists.")
        sys.stdout.flush()
        fluvio_client = None # Ensure client is None on failure
        fluvio_producer = None # Ensure producer is None on failure
        return False

def send_to_fluvio(key, data_dict):
    """Sends data dictionary as JSON to the configured Fluvio topic."""
    global fluvio_producer # Access the global producer
    if not fluvio_producer:
        # print("Fluvio producer not available. Cannot send data.") # Can be noisy
        return # Silently fail if producer not ready

    try:
        # Ensure key is bytes
        key_bytes = str(key).encode('utf-8')
        # Ensure data is JSON string, then encode to bytes
        data_json_str = json.dumps(data_dict)
        data_bytes = data_json_str.encode('utf-8')

        # Send the record - basic send is blocking but simpler for this setup
        fluvio_producer.send(key_bytes, data_bytes)
        # print(f"-> Sent data to Fluvio (Key: {key}, Status: {data_dict.get('frame_status')})") # Can be noisy, enable for debug
    except Exception as e:
        print(f"!!! FLUVIO WARNING: Could not send data (Key: {key}) to topic '{FLUVIO_CROWD_TOPIC}'.")
        print(f"    Error details: {e}")
        # In a production app, you might try to reconnect or queue failed messages


# --- Helper Functions ---
def analyze_density_grid(density_grid):
    """Analyzes the grid to determine status and risky cells."""
    high_density_cells = 0
    critical_density_cells = 0
    risky_cell_coords = [] # List of (row, col) for risky cells
    overall_status = "Normal"
    total_people_in_grid = 0

    if not density_grid or len(density_grid) != GRID_ROWS:
        # print("   Warning: Invalid density grid received.") # Can be noisy
        return overall_status, risky_cell_coords, total_people_in_grid, high_density_cells, critical_density_cells

    for r_idx, row in enumerate(density_grid):
        if len(row) != GRID_COLS: continue
        for c_idx, count in enumerate(row):
            try:
                person_count = int(count)
                total_people_in_grid += person_count
                if person_count >= CRITICAL_DENSITY_THRESHOLD:
                    critical_density_cells += 1
                    risky_cell_coords.append((r_idx, c_idx))
                elif person_count >= HIGH_DENSITY_THRESHOLD:
                    high_density_cells += 1
                    # Add high density cells to risky_cell_coords as they also get an overlay
                    risky_cell_coords.append((r_idx, c_idx))
            except (ValueError, TypeError):
                continue # Skip invalid cell counts

    # Determine overall status based on cell counts
    if critical_density_cells >= CRITICAL_DENSITY_CELL_COUNT_THRESHOLD:
        overall_status = "CRITICAL RISK"
    elif critical_density_cells > 0:
        overall_status = "Critical Density Cell Detected"
    elif high_density_cells >= HIGH_DENSITY_CELL_COUNT_THRESHOLD:
        overall_status = "High Density Warning"
    elif high_density_cells > 0:
        overall_status = "High Density Cell Detected"

    # Return all calculated values
    return overall_status, risky_cell_coords, total_people_in_grid, high_density_cells, critical_density_cells


def get_higher_priority_status(status1, status2):
    """Compares two status strings and returns the one with higher priority."""
    p1 = STATUS_HIERARCHY.get(status1, -1)
    p2 = STATUS_HIERARCHY.get(status2, -1)
    return status1 if p1 >= p2 else status2


# --- Frame/Image Processing Function ---
def process_media_content(content, content_width, content_height, frame_or_image_index, current_overall_status_context="Normal"):
    """
    Processes a single image or video frame: detects people, calculates density,
    determines status, draws overlays/text, AND sends data to Fluvio.
    Returns the processed content (image/frame), content status, and confirmed person count.
    """
    if content is None:
        # print(f"Warning: Received None content for processing at index {frame_or_image_index}")
        # Return context status or a generic incomplete status, 0 people
        return None, "Analysis Incomplete (No Content)", 0

    if detector is None:
         # Handle case where model loading failed at startup
         error_frame = content.copy() if content is not None else np.zeros((content_height or 480, content_width or 640, 3), dtype=np.uint8)
         cv2.putText(error_frame, "MODEL ERROR!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
         print(f"!!! Skipping processing for index {frame_or_image_index}: ML model not loaded.")
         # Return a clear error status, but keep the count at 0 as detection didn't happen
         return error_frame, "Error: Model Not Loaded", 0


    start_process_time = time.time()
    # Ensure content is valid before copying
    if content is None:
         return None, "Analysis Incomplete (Invalid Content)", 0
    processed_content = content.copy() # Always work on a copy

    content_status = "Analysis Incomplete" # Default status for this frame
    confirmed_person_count_this_content = 0
    density_grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    risky_coords = [] # Initialize risky coords list
    high_cells = 0
    crit_cells = 0


    try:
        # --- 1. ML Preprocessing & Detection ---
        # Ensure image is in the correct color space for the model (often RGB expected by TF models)
        # OpenCV reads as BGR by default
        if len(content.shape) == 2: rgb_content = cv2.cvtColor(content, cv2.COLOR_GRAY2RGB)
        elif content.shape[2] == 4: rgb_content = cv2.cvtColor(content, cv2.COLOR_BGRA2RGB)
        elif content.shape[2] == 3: rgb_content = cv2.cvtColor(content, cv2.COLOR_BGR2RGB)
        else: rgb_content = content # Fallback - hope it's already RGB or grayscale

        # The detector expects uint8 and specific dimensions (like 320x320),
        # but hub.load often handles resizing internally. We provide the original size image tensor.
        image_tensor = tf.expand_dims(tf.convert_to_tensor(rgb_content, dtype=tf.uint8), axis=0)
        detections = detector(image_tensor)

        # Filter detections based on score threshold and class (PERSON_CLASS_INDEX)
        # Using boolean masking is efficient with TensorFlow outputs
        detection_scores = detections['detection_scores'][0]
        detection_classes = detections['detection_classes'][0].numpy().astype(int)
        detection_boxes = detections['detection_boxes'][0]

        # Create boolean mask: score >= threshold AND class == PERSON
        is_person = (detection_classes == PERSON_CLASS_INDEX)
        is_confident = (detection_scores >= DETECTION_THRESHOLD)
        valid_detections_mask = is_person & is_confident

        # Apply mask to get filtered boxes, classes, and scores
        filtered_boxes = tf.boolean_mask(detection_boxes, valid_detections_mask).numpy()
        # filtered_classes = tf.boolean_mask(detection_classes, valid_detections_mask).numpy() # Not strictly needed for density/count
        # filtered_scores = tf.boolean_mask(detection_scores, valid_detections_mask).numpy() # Not strictly needed for density/count


        # --- 2. Calculate Grid & Filter Detections ---
        confirmed_person_count_this_content = filtered_boxes.shape[0] # Count of persons above threshold

        cell_height = content_height // GRID_ROWS
        cell_width = content_width // GRID_COLS

        if cell_height <= 0 or cell_width <= 0:
            print(f"Warning: Content dimensions ({content_width}x{content_height}) too small for grid size ({GRID_COLS}x{GRID_ROWS}). Skipping density analysis for index {frame_or_image_index}.")
            # Set status to reflect incomplete analysis if grid is not possible
            content_status = "Analysis Incomplete (Tiny Frame)"
            # No density grid calculation or analysis - density_grid remains [[0...]]
        else:
             # Populate density grid
             for i in range(filtered_boxes.shape[0]):
                 # Note: boxes are [ymin, xmin, ymax, xmax] in relative coordinates (0 to 1)
                 ymin, xmin, ymax, xmax = filtered_boxes[i]
                 # Calculate center point in pixel coordinates
                 center_x = int((xmin + xmax) / 2 * content_width)
                 center_y = int((ymin + ymax) / 2 * content_height)

                 # Determine grid cell - ensure bounds are safe
                 row = min(max(0, center_y // cell_height), GRID_ROWS - 1)
                 col = min(max(0, center_x // cell_width), GRID_COLS - 1)

                 # Increment density count for the cell
                 density_grid[row][col] += 1


             # --- 3. Analyze Density Grid ---
             content_status, risky_coords, total_grid_people, high_cells, crit_cells = analyze_density_grid(density_grid)


        # --- 4. Send Data to Fluvio ---
        fluvio_payload = {
            "timestamp": int(time.time()), # Current timestamp
            "frame": frame_or_image_index, # Frame number or image index (e.g., 0 for single image)
            "density_grid": density_grid, # The calculated grid (will be [[0...]] if grid analysis skipped)
            "frame_status": content_status, # Status determined for this frame
            "confirmed_persons": confirmed_person_count_this_content, # Persons detected above threshold
            "high_density_cells": high_cells, # Count of cells >= HIGH_DENSITY_THRESHOLD (0 if grid skipped)
            "critical_density_cells": crit_cells # Count of cells >= CRITICAL_DENSITY_THRESHOLD (0 if grid skipped)
        }
        # Use a composite key including original filename or source identifier if available
        # For simplicity here, using a generic key with index.
        # In a multi-camera setup, the key should include camera ID.
        send_to_fluvio(f"content-{frame_or_image_index}", fluvio_payload)


        # --- 5. Draw Overlays and Text ---
        # --- REMOVED: Grid line drawing ---

        # Draw overlays for individual risky grid cells
        overlay_alpha = 0.4
        overlay_color_critical = (0, 0, 255) # Red (BGR)
        overlay_color_high = (0, 165, 255) # Orange (BGR)

        if cell_height > 0 and cell_width > 0: # Only draw overlays if grid dimensions are valid
             for r, c in risky_coords:
                 cell_y_start = r * cell_height
                 cell_y_end = (r + 1) * cell_height
                 cell_x_start = c * cell_width
                 cell_x_end = (c + 1) * cell_width

                 risk_level_in_cell = "high" # Assume high unless proven critical
                 try:
                     if density_grid[r][c] >= CRITICAL_DENSITY_THRESHOLD: risk_level_in_cell = "critical"
                 except IndexError:
                      continue # Should not happen with robust index calculation

                 color = overlay_color_critical if risk_level_in_cell == "critical" else overlay_color_high
                 overlay = processed_content.copy() # Create a fresh overlay copy for blending
                 cv2.rectangle(overlay, (cell_x_start, cell_y_start), (cell_x_end, cell_y_end), color, -1)
                 # Blend the overlay onto the processed_content
                 cv2.addWeighted(overlay, overlay_alpha, processed_content, 1 - overlay_alpha, 0, processed_content)

                 # Optionally draw person count in the cell
                 # try:
                 #     if density_grid[r][c] > 0:
                 #         count_text = str(density_grid[r][c])
                 #         # Center text within the cell (simple approximate centering)
                 #         text_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                 #         text_x = cell_x_start + (cell_width - text_size[0]) // 2
                 #         text_y = cell_y_start + (cell_height + text_size[1]) // 2
                 #         cv2.putText(processed_content, count_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                 # except IndexError:
                 #     pass # Should not happen


        # Draw Status Text & Stampede Chance Text
        # For the *output frame*, we should show *its* specific status.
        frame_display_status = content_status

        status_text = f"Risk: {frame_display_status}"
        status_color = (0, 128, 0) # Green default
        if "CRITICAL" in frame_display_status: status_color = (0, 0, 255) # Red
        elif "Warning" in frame_display_status or "High" in frame_display_status or "Detected" in frame_display_status: status_color = (0, 165, 255) # Orange
        elif "Error" in frame_display_status or "Incomplete" in frame_display_status: status_color = (0, 0, 255) # Red for errors

        # Stampede Chance Text determination based on frame status
        if "CRITICAL" in frame_display_status: chance_text, chance_color = "Stampede Chance: Critical", (0, 0, 255)
        elif "Warning" in frame_display_status or "High" in frame_display_status or "Detected" in frame_display_status: chance_text, chance_color = "Stampede Chance: High", (0, 165, 255)
        else: chance_text, chance_color = "Stampede Chance: Low", (0, 128, 0)


        # Add text with background rectangles for readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        padding = 5
        bg_color = (50, 50, 50) # Dark gray background
        # text_color = (255, 255, 255) # White text color outline/background - color is set by status/chance

        # Status text - Top Left
        (text_w_s, text_h_s), baseline_s = cv2.getTextSize(status_text, font, font_scale, font_thickness)
        # Add background rectangle
        cv2.rectangle(processed_content, (padding, padding), (padding * 2 + text_w_s, padding * 2 + text_h_s), bg_color, -1)
        # Add text
        cv2.putText(processed_content, status_text, (padding + 5, padding + text_h_s + 5), font, font_scale, status_color, font_thickness, cv2.LINE_AA)


        # Chance text - Bottom Left
        (text_w_c, text_h_c), baseline_c = cv2.getTextSize(chance_text, font, font_scale, font_thickness)
        # Add background rectangle
        cv2.rectangle(processed_content, (padding, content_height - padding * 2 - text_h_c), (padding * 2 + text_w_c, content_height - padding), bg_color, -1)
        # Add text
        cv2.putText(processed_content, chance_text, (padding + 5, content_height - padding - baseline_c), font, font_scale, chance_color, font_thickness, cv2.LINE_AA)

        # Optional: Display person count somewhere (e.g., top right)
        person_count_text = f"Persons: {confirmed_person_count_this_content}"
        (text_w_p, text_h_p), baseline_p = cv2.getTextSize(person_count_text, font, font_scale, font_thickness)
        cv2.rectangle(processed_content, (content_width - text_w_p - padding*2, padding), (content_width - padding, padding * 2 + text_h_p), bg_color, -1)
        cv2.putText(processed_content, person_count_text, (content_width - text_w_p - padding, padding + text_h_p + 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)



    except Exception as e:
        print(f"!!! ERROR during process_media_content for index {frame_or_image_index}: {e}")
        # Draw error text on frame instead of processing results
        # Ensure frame exists before copying/drawing
        error_frame = content.copy() if content is not None else np.zeros((content_height or 480, content_width or 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Processing Error: {e}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        # Return the error frame + error status + 0 count
        return error_frame, f"Error: {e}", 0

    # end_process_time = time.time()
    # print(f" -> Index {frame_or_image_index} processed in {end_process_time - start_process_time:.3f}s. Status: {content_status}")

    # Return processed frame, its specific status, and person count
    return processed_content, content_status, confirmed_person_count_this_content


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index_route():
    """Serves the main page."""
    return render_template('index.html')

@app.route('/upload_media', methods=['POST'])
def upload_media_route():
    print("\n--- Request received for /upload_media ---")
    if fluvio_producer:
         print("   Fluvio Status: Producer is active.")
    else:
         print("   Fluvio Status: Producer is INACTIVE.")
    sys.stdout.flush()

    # Check if model is loaded before proceeding with processing
    if detector is None:
        print("!!! ERROR: ML model not loaded. Cannot process media.")
        # Render results page with an error status
        return render_template('results.html',
                               output_media_type=None,
                               processed_media_url=None,
                               download_video_url=None,
                               prediction_status="Error: Model Not Loaded",
                               max_persons=0,
                               processing_time="N/A")

    start_time = time.time()

    if 'media' not in request.files: return 'No media file part in the request', 400
    media_file = request.files['media']
    if media_file.filename == '': return 'No selected media file', 400

    original_filename = secure_filename(media_file.filename)
    # Use a timestamp or UUID to make upload filenames unique and avoid conflicts
    unique_filename_prefix = str(int(time.time())) # Simple timestamp prefix
    unique_original_filename = f"{unique_filename_prefix}_{original_filename}"

    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_original_filename)
    try:
        media_file.save(upload_path)
        print(f"Media saved temporarily to: {upload_path}")
    except Exception as e:
        print(f"Error saving media file: {e}")
        return f"Error saving media file: {e}", 500


    mimetype = mimetypes.guess_type(upload_path)[0]
    file_type = 'unknown'
    if mimetype:
        if mimetype.startswith('video/'): file_type = 'video'
        elif mimetype.startswith('image/'): file_type = 'image'
    print(f"Detected file type: {file_type}")

    # --- Initialize vars for results template ---
    processed_media_url = None # URL for the image to display (processed image or critical frame)
    download_video_url = None  # URL for the full video (only if video)
    overall_processing_status = "Processing Started" # Overall status for the entire file/video
    max_persons = 0
    output_media_type = file_type # Use detected type for the template

    # --- Process Video ---
    if file_type == 'video':
        output_video_filename = f"processed_{unique_original_filename}"
        # Ensure video filename has an mp4 extension if it doesn't already
        if not output_video_filename.lower().endswith('.mp4'):
             # Remove existing extension and add .mp4
             output_video_filename = os.path.splitext(output_video_filename)[0] + ".mp4"

        output_video_path = os.path.join(PROCESSED_VIDEO_FOLDER, output_video_filename)

        cap = cv2.VideoCapture(upload_path)

        if not cap.isOpened():
            print(f"!!! ERROR: Failed to open input video file: {upload_path}")
            overall_processing_status = "Error: Could not open input video"
            # Fall through to cleanup and render error
        else:
            try: # Wrap processing in a try block
                fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps and fps > 0 else 25.0 # Default to 25 if FPS is zero or None
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Ensure valid dimensions
                if width <= 0 or height <= 0:
                    print(f"!!! ERROR: Invalid video dimensions ({width}x{height}) for {upload_path}.")
                    overall_processing_status = "Error: Invalid video dimensions"
                else:
                    print(f"Video Input: {width}x{height} @ {fps:.2f} FPS")

                    # Use 'mp4v' or 'avc1' (H.264) - 'avc1' generally offers better compression and compatibility
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Default codec
                    # Optional: Try 'avc1' first if supported
                    try:
                         test_writer = cv2.VideoWriter()
                         # Create a dummy file name for the test
                         test_output_path = os.path.join(app.config['PROCESSED_VIDEO_FOLDER'], "codec_test_dummy.mp4")
                         if test_writer.open(test_output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height), True):
                             fourcc = cv2.VideoWriter_fourcc(*'avc1')
                             print("Using avc1 codec (H.264).")
                         else:
                             print("avc1 codec not available, using mp4v.")
                         test_writer.release()
                         if os.path.exists(test_output_path): os.remove(test_output_path) # Clean up dummy file
                     # Use a broad except for codec testing as it can raise various errors
                    except Exception as codec_e:
                         print(f"Codec test failed ({codec_e}), using mp4v.")
                         fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Re-assign if test failed


                    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                    print(f"Attempting to open VideoWriter for: {output_video_path}")

                    if not out_video.isOpened():
                        print(f"!!! ERROR: Failed to initialize VideoWriter for {output_video_path}. Check codecs and permissions.")
                        overall_processing_status = "Error: Failed to initialize VideoWriter"
                    else:
                        print("VideoWriter opened successfully. Starting frame processing loop...")
                        frame_num = 0
                        video_highest_status = "Normal" # Track highest status found in video
                        critical_frame_url_to_save = None # Will store the URL of the image file of the critical frame
                        processed_first_frame_content = None # Keep processed first frame content in memory temporarily

                        # Process the first frame initially to have a fallback image if no detections occur
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Ensure we are at frame 0
                        ret_first, first_frame = cap.read()
                        if ret_first and first_frame is not None:
                            # Process first frame to get initial status and image
                            processed_first_frame_content, first_frame_status, _ = process_media_content(
                                first_frame, width, height, 0, "Normal" # Process first frame with 'Normal' context
                            )
                            if processed_first_frame_content is not None:
                                # This first processed frame is our initial candidate for the critical frame
                                video_highest_status = first_frame_status
                                # We will save this *after* the loop if no higher status frame is found
                                print(f"Processed initial frame (Status: {video_highest_status})")


                        # Reset capture to beginning for the main loop
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_num = 0 # Reset frame counter
                        processed_critical_frame_content = None # Store the actual frame data for the critical frame

                        while True:
                            ret, frame = cap.read()
                            if not ret or frame is None: break # End of video or failed read

                            # Process frame (pass video_highest_status as context for display on frame, though function uses it little)
                            # We get the *current frame's* specific status back
                            processed_frame, current_frame_status, people_count = process_media_content(
                                frame, width, height, frame_num, video_highest_status
                            )

                            if processed_frame is None:
                                print(f"Warning: Skipping frame {frame_num} due to processing failure.")
                                frame_num += 1
                                continue # Skip writing/analyzing this frame if processing failed

                            # Check if this frame's status is higher priority than the highest seen so far
                            new_highest_status = get_higher_priority_status(video_highest_status, current_frame_status)

                            # If a new strictly higher status is found, update the highest status
                            # and store this frame content as the new critical frame candidate
                            if STATUS_HIERARCHY.get(new_highest_status, -1) > STATUS_HIERARCHY.get(video_highest_status, -1):
                                 video_highest_status = new_highest_status
                                 processed_critical_frame_content = processed_frame.copy() # Store the frame data
                                 print(f"New highest status found (Status: {video_highest_status}) at frame {frame_num}. Storing frame content.")


                            # Update max persons found in the entire video
                            max_persons = max(max_persons, people_count)

                            # Write frame to the full output video
                            try:
                                out_video.write(processed_frame)
                            except Exception as write_e:
                               print(f"!!! ERROR writing frame {frame_num} to video: {write_e}")
                               # Stop processing the video if writing fails
                               overall_processing_status = get_higher_priority_status(overall_processing_status, f"Error: Video writing failed frame {frame_num}")
                               break # Exit the processing loop

                            frame_num += 1
                            # Optional: progress update
                            # if frame_num > 0 and frame_num % 100 == 0: print(f"  Processed frame {frame_num}...")


                        print(f"Frame processing loop finished after {frame_num} frames.")
                        # Final status for the video is the highest status encountered across all frames
                        overall_processing_status = video_highest_status

                        # Release VideoWriter FIRST
                        if out_video.isOpened():
                            out_video.release()
                            print("VideoWriter released.")

                        # --- Save the critical frame image (or fallback) ---
                        frame_to_save = processed_critical_frame_content # Use the highest-status frame found
                        if frame_to_save is None: # If no critical frame with a status > Normal was found
                             frame_to_save = processed_first_frame_content # Use the processed first frame as fallback
                             print("No higher-status frame found, using processed first frame as display image.")

                        if frame_to_save is not None:
                             critical_frame_filename = f"display_frame_{unique_filename_prefix}_{os.path.splitext(original_filename)[0]}.jpg"
                             critical_frame_path = os.path.join(PROCESSED_FRAMES_FOLDER, critical_frame_filename)
                             try:
                                 cv2.imwrite(critical_frame_path, frame_to_save)
                                 critical_frame_url_to_save = url_for('static', filename=f'processed_frames/{critical_frame_filename}')
                                 print(f"Saved display frame (Status: {overall_processing_status}) at {critical_frame_path}")
                             except Exception as img_save_e:
                                 print(f"!!! ERROR saving display frame image {critical_frame_path}: {img_save_e}")
                                 critical_frame_url_to_save = None # Ensure it's None on failure
                        else:
                            print("!!! WARNING: No frame content available to save as display image.")
                            critical_frame_url_to_save = None


                        # Check if the output video file was successfully created and is not empty
                        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                             download_video_url = url_for('static', filename=f'processed_videos/{output_video_filename}')
                             print(f"Full processed video created: {output_video_path}")
                        else:
                             print(f"!!! ERROR: Output video file missing or empty after processing: {output_video_path}")
                             overall_processing_status = get_higher_priority_status(overall_processing_status, "Error: Output video generation failed")
                             # If video creation failed, the download URL will be None

                # End of else (valid dimensions) block
            except Exception as video_proc_e:
                print(f"!!! ERROR during video processing: {video_proc_e}")
                overall_processing_status = get_higher_priority_status(overall_processing_status, f"Error: Unexpected failure during video processing - {video_proc_e}")
                # Try to cleanup partially written video file if it exists
                if os.path.exists(output_video_path):
                     try:
                         os.remove(output_video_path)
                         print(f"Removed incomplete video file: {output_video_path}")
                     except OSError as cleanup_e:
                         print(f"Warning: Could not remove incomplete video file {output_video_path}: {cleanup_e}")

            finally:
                # Ensure VideoCapture is released
                if cap.isOpened(): cap.release(); print("VideoCapture released.")

        # Assign the URL of the critical frame image (or initial frame fallback) for display
        processed_media_url = critical_frame_url_to_save


    # --- Process Image ---
    elif file_type == 'image':
        output_image_filename = f"processed_{unique_original_filename}"
        # Ensure filename ends with a common image extension
        # Get original extension or default to jpg if unknown/missing
        ext = os.path.splitext(unique_original_filename)[1].lower()
        if ext not in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'):
             ext = ".jpg" # Default to jpg

        output_image_filename = os.path.splitext(output_image_filename)[0] + ext
        output_image_path = os.path.join(PROCESSED_IMAGE_FOLDER, output_image_filename)

        try:
            image = cv2.imread(upload_path)
            if image is None:
               raise ValueError("Could not read image file with OpenCV.")
            height, width = image.shape[:2] # Get height and width

            print(f"Image Input: {width}x{height}")

            # Process the single image (frame_index=0, status starts 'Normal')
            processed_image, image_status, people_count = process_media_content(
                image, width, height, 0, "Normal" # Process image with 'Normal' context
            )

            if processed_image is None:
               raise ValueError("Image processing function returned None.")

            overall_processing_status = image_status # Overall status is just the result of this one image
            max_persons = people_count

            # Save the processed image
            save_success = cv2.imwrite(output_image_path, processed_image)
            if not save_success:
                raise ValueError(f"Failed to save processed image to {output_image_path}. Check permissions.")

            print(f"Processed image saved to: {output_image_path}")
            processed_media_url = url_for('static', filename=f'processed_images/{output_image_filename}')
            download_video_url = None # No video to download for image upload


        except Exception as img_proc_e:
           print(f"!!! ERROR during image processing: {img_proc_e}")
           overall_processing_status = get_higher_priority_status(overall_processing_status, f"Error: Image processing failed - {img_proc_e}")
           processed_media_url = None # Indicate failure
           download_video_url = None


    # --- Handle Unknown File Type ---
    else:
        print(f"Unsupported file type: {mimetype}")
        overall_processing_status = "Error: Unsupported file type"
        processed_media_url = None
        download_video_url = None

    # --- Cleanup Upload ---
    try:
        os.remove(upload_path)
        print(f"Removed temporary upload: {upload_path}")
    except OSError as e:
        print(f"Warning: Could not remove temporary upload {upload_path}: {e}")
    except Exception as e:
        print(f"Warning: Unexpected error removing temporary upload {upload_path}: {e}")


    processing_time_secs = time.time() - start_time
    print(f"Total request processing time: {processing_time_secs:.2f} seconds")

    # --- Render Results ---
    print(f"---> Rendering results page:")
    print(f"     Media Type: {output_media_type}")
    print(f"     Displayed Media URL (Image/Critical Frame): {processed_media_url}")
    print(f"     Download Video URL: {download_video_url}")
    print(f"     Overall Status: {overall_processing_status}")
    print(f"     Max Persons: {max_persons}")

    # Pass relevant data to the results template
    return render_template('results.html',
                           output_media_type=output_media_type,
                           processed_media_url=processed_media_url, # This is the image URL to DISPLAY
                           download_video_url=download_video_url,   # This is the video URL for DOWNLOAD (or None)
                           prediction_status=overall_processing_status,
                           max_persons=max_persons,
                           processing_time=f"{processing_time_secs:.2f}")


# --- Live Stream Route ---
def generate_live_frames():
    print("\n--- Request received for /video_feed (Live Stream) ---")
    if fluvio_producer:
         print("   Fluvio Status: Producer is active.")
    else:
         print("   Fluvio Status: Producer is INACTIVE.")
    sys.stdout.flush()

    # Check if model is loaded for the live stream
    if detector is None:
        print("!!! ERROR: ML model not loaded. Cannot start live stream.")
        # Yield an error image frame instead of processing frames
        error_msg = "ML Model Loading Failed. Cannot stream."
        # Attempt to get resolution if possible, else default
        width, height = 640, 480
        try:
            # If cap could be opened just to get dimensions, use them
            temp_cap = cv2.VideoCapture(0) # Try opening briefly
            if temp_cap.isOpened():
                 width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                 height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                 temp_cap.release()
        except Exception as dim_e:
            print(f"Warning: Could not get camera dimensions: {dim_e}. Using default {width}x{height}.")

        blank_img = np.zeros((height, width, 3), dtype=np.uint8) # Create a black image
        # Add red error text - scale font based on width
        font_scale = min(width, height) / 600.0 # Adjust font scale based on resolution
        cv2.putText(blank_img, error_msg, (int(width*0.05), int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2, cv2.LINE_AA) # Add red error text
        ret_enc, buffer = cv2.imencode('.jpg', blank_img)
        if ret_enc:
             yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        # Optional: Yield a few times or sleep to keep the error visible before ending
        # for _ in range(3): # Yield 3 times
        #     yield (b'--frame\r\n'
        #            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        #     time.sleep(0.1)
        return # Stop the generator after yielding error frame

    # --- Webcam/Video Source Handling ---
    # Try default camera (0) first on Windows. Add backend hints if needed.
    # If 0 doesn't work, try 1, 2 etc. Experiment or let the code try the fallback.
    # live_cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW) # Explicitly use DirectShow
    # live_cap = cv2.VideoCapture(0 + cv2.CAP_MSMF) # Explicitly use Media Foundation
    live_cap = cv2.VideoCapture(0) # Standard attempt first

    # Fallback to video file if webcam fails
    if not live_cap.isOpened():
        print("!!! WARNING: Cannot open default video source (webcam 0). Attempting to use videoplayback.mp4...")
        # Ensure the fallback video file exists! Place videoplayback.mp4 next to app.py
        fallback_video_path = "videoplayback.mp4"
        if os.path.exists(fallback_video_path):
             live_cap = cv2.VideoCapture(fallback_video_path)
             if not live_cap.isOpened():
                  print(f"!!! ERROR: Failed to open fallback video file: {fallback_video_path}")
             else:
                  print(f"Successfully opened fallback video: {fallback_video_path}")
        else:
             print(f"!!! ERROR: Fallback video file not found: {fallback_video_path}")


    # If neither webcam nor fallback video opened
    if not live_cap.isOpened():
        print("!!! ERROR: Cannot open any video source (webcam/file).")
        error_msg = "Cannot open video source."
        # Default dimensions if unable to get them
        width, height = 640, 480
        try: # Attempt to get some default dimensions if possible
             temp_cap = cv2.VideoCapture(0) # Try opening briefly
             if temp_cap.isOpened():
                 width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                 height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                 temp_cap.release()
        except Exception as dim_e:
            print(f"Warning: Could not get camera dimensions for error frame: {dim_e}. Using default {width}x{height}.")

        blank_img = np.zeros((height, width, 3), dtype=np.uint8) # Create black image
        font_scale = min(width, height) / 600.0
        cv2.putText(blank_img, error_msg, (int(width*0.05), int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2, cv2.LINE_AA) # Add red error text
        ret_enc, buffer = cv2.imencode('.jpg', blank_img)
        if ret_enc:
             yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        # time.sleep(0.5)
        return # Stop the generator

    # Source successfully opened (either webcam or video file)
    frame_width = int(live_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(live_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Final check for valid dimensions just in case
    if frame_width <= 0 or frame_height <= 0:
         print("!!! ERROR: Invalid frame dimensions from opened video source.")
         live_cap.release()
         error_msg = "Invalid video source dimensions."
         # Fallback to a default sized error image
         blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
         font_scale = min(640, 480) / 600.0
         cv2.putText(blank_img, error_msg, (int(640*0.05), int(480/2)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2, cv2.LINE_AA)
         ret_enc, buffer = cv2.imencode('.jpg', blank_img)
         if ret_enc:
             yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
         # time.sleep(0.5)
         return


    print(f"Live source opened: {frame_width}x{frame_height}")

    frame_num = 0
    overall_stream_status = "Normal" # Track highest status seen *during* the current stream session

    while True:
        ret, frame = live_cap.read()
        # Loop video if using a file source for "live" demo
        if not ret or frame is None:
            # If using a video file, loop it
            if live_cap.get(cv2.CAP_PROP_POS_FRAMES) > 0: # Check if we read at least one frame
                 live_cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Set frame position back to beginning
                 print("Looping video source...")
                 ret, frame = live_cap.read() # Read the first frame again
            # If it's still not ret or frame is None (e.g., camera disconnected, empty file)
            if not ret or frame is None:
                print("End of live stream source (or camera disconnected). Stopping generator.")
                break # Really end the stream

        # Process frame (pass overall_stream_status as context for display on frame)
        # We get the *current frame's* specific status back
        processed_frame, frame_status, _ = process_media_content(
            frame, frame_width, frame_height, frame_num, overall_stream_status
        )

        # Update the overall highest status for the stream session
        overall_stream_status = get_higher_priority_status(overall_stream_status, frame_status)

        if processed_frame is None:
            print(f"Warning: Skipping live frame {frame_num} due to processing failure.")
            frame_num += 1
            continue # Skip encoding/yielding this frame


        try:
            # Encode processed frame as JPEG for streaming (MJPEG format)
            ret_enc, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret_enc:
                print(f"Error encoding live frame {frame_num}.")
                frame_num += 1
                continue # Skip if encoding fails

            frame_bytes = buffer.tobytes()
            # Yield the frame in a multi-part response required for MJPEG streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Error encoding/yielding live frame {frame_num}: {e}")
            # Consider yielding an error frame before breaking in a more robust app
            break # Stop stream on error

        frame_num += 1
        # Optional: Add a small delay if processing is too fast and consuming excessive CPU/bandwidth
        # time.sleep(0.01) # Example: 10ms delay results in max ~100 FPS yield


    # Ensure capture is released when the generator stops
    live_cap.release()
    print("Live stream generator finished.")


@app.route('/live')
def live_route():
    """Serves the live stream page."""
    # Attempt Fluvio connection before rendering the live page
    connect_fluvio()
    return render_template('live.html')

@app.route('/video_feed')
def video_feed_route():
    """Provides the MJPEG stream for the live page."""
    # The generator function generate_live_frames handles the actual frame processing and yielding
    return Response(generate_live_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Run App ---
if __name__ == '__main__':
    print("--- Initializing Application ---")
    # Attempt to connect to Fluvio when the application starts
    # This connection will persist for the lifetime of the process
    fluvio_connected = connect_fluvio()
    if not fluvio_connected:
         print("!!! WARNING: Fluvio connection failed on startup. Data will not be sent.")
    else:
         print("+++ Fluvio connection successful on startup.")

    # Attempt to load the ML model on startup
    # The global 'detector' variable is set by the initial load attempt.
    # Error handling for missing detector is present in processing functions.
    if detector is None:
        print("!!! WARNING: ML model failed to load on startup. Media processing and live streams will not work.")

    print("--- Starting Flask Server ---")
    # use_reloader=False is important with background threads/connections (like Fluvio)
    # threaded=True is needed for handling concurrent requests (like /upload_media and /video_feed simultaneously)
    # host='0.0.0.0' makes the server accessible from external IPs
    # debug=True enables Flask's debug mode (auto-reloads on code changes - potentially conflicts with use_reloader=False, but good for development logs)
    # Setting TF log level again just before run in case something resets it
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Use a try-except block around app.run to catch startup errors gracefully
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
    except Exception as run_e:
        print(f"!!! FATAL ERROR: Flask application failed to start: {run_e}")
    finally:
        os.environ.pop('TF_CPP_MIN_LOG_LEVEL', None) # Clean up env variable on exit
        print("--- Application Shutting Down ---")

