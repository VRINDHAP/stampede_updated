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
DETECTION_THRESHOLD = 0.15 # Confidence threshold for detections

# --- Density Analysis Settings ---
HIGH_DENSITY_THRESHOLD = 6 # People count per cell to be considered 'High Density'
CRITICAL_DENSITY_THRESHOLD = 8 # People count per cell to be considered 'Critical Density'
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


'''
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
DETECTION_THRESHOLD = 0.12 # Confidence threshold for detections

# --- Density Analysis Settings ---
HIGH_DENSITY_THRESHOLD = 5 # People count per cell to be considered 'High Density'
CRITICAL_DENSITY_THRESHOLD = 7 # People count per cell to be considered 'Critical Density'
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
        return None, current_overall_status_context, 0 # Return context status, 0 people

    if detector is None:
         # Handle case where model loading failed at startup
         error_frame = content.copy()
         cv2.putText(error_frame, "MODEL ERROR!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
         print(f"!!! Skipping processing for index {frame_or_image_index}: ML model not loaded.")
         # Return a clear error status, but keep the count at 0 as detection didn't happen
         return error_frame, "Error: Model Not Loaded", 0


    start_process_time = time.time()
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
            # No density grid calculation or analysis
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
        send_to_fluvio(f"content-{frame_or_image_index}", fluvio_payload)


        # --- 5. Draw Overlays and Text ---
        # --- REMOVED: Grid line drawing ---
        # Draw grid lines (optional but helpful for visualization)
        # grid_line_color = (200, 200, 200) # Light gray
        # grid_line_thickness = 1
        # for r in range(1, GRID_ROWS):
        #     cv2.line(processed_content, (0, r * cell_height), (content_width, r * cell_height), grid_line_color, grid_line_thickness)
        # for c in range(1, GRID_COLS):
        #     cv2.line(processed_content, (c * cell_width, 0), (c * cell_width, content_height), grid_line_color, grid_line_thickness)


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
        text_color = (255, 255, 255) # White text color outline/background

        # Status text
        (text_w_s, text_h_s), baseline_s = cv2.getTextSize(status_text, font, font_scale, font_thickness)
        # Add background rectangle
        cv2.rectangle(processed_content, (padding, padding), (padding * 2 + text_w_s, padding * 2 + text_h_s), bg_color, -1)
        # Add text
        cv2.putText(processed_content, status_text, (padding + 5, padding + text_h_s + 5), font, font_scale, status_color, font_thickness, cv2.LINE_AA)


        # Chance text
        (text_w_c, text_h_c), baseline_c = cv2.getTextSize(chance_text, font, font_scale, font_thickness)
        # Add background rectangle
        cv2.rectangle(processed_content, (padding, content_height - padding * 2 - text_h_c), (padding * 2 + text_w_c, content_height - padding), bg_color, -1)
        # Add text
        cv2.putText(processed_content, chance_text, (padding + 5, content_height - padding - baseline_c), font, font_scale, chance_color, font_thickness, cv2.LINE_AA)


    except Exception as e:
        print(f"!!! ERROR during process_media_content for index {frame_or_image_index}: {e}")
        # Draw error text on frame instead of processing results
        error_frame = content.copy() # Start with original again to avoid drawing on potentially corrupted processed_content
        cv2.putText(error_frame, f"Processing Error: {e}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        # Return original content + error status + 0 count
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

                        # Process the first frame initially to have a fallback image if no detections occur
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Ensure we are at frame 0
                        ret_first, first_frame = cap.read()
                        if ret_first and first_frame is not None:
                            # Process first frame to get initial status and image
                            processed_first_frame, first_frame_status, _ = process_media_content(
                                first_frame, width, height, 0, "Normal" # Process first frame with 'Normal' context
                            )
                            if processed_first_frame is not None:
                                # This first processed frame is our initial candidate for the critical frame
                                video_highest_status = first_frame_status
                                fallback_frame_filename = f"initial_frame_{unique_filename_prefix}_frame0_{os.path.splitext(original_filename)[0]}.jpg"
                                fallback_frame_path = os.path.join(PROCESSED_FRAMES_FOLDER, fallback_frame_filename)
                                try:
                                    cv2.imwrite(fallback_frame_path, processed_first_frame)
                                    critical_frame_url_to_save = url_for('static', filename=f'processed_frames/{fallback_frame_filename}')
                                    print(f"Saved initial frame image (Status: {video_highest_status}) at {fallback_frame_path}")
                                except Exception as fb_save_e:
                                    print(f"!!! ERROR saving initial frame image {fallback_frame_path}: {fb_save_e}")
                                    critical_frame_url_to_save = None # Ensure it's None on failure

                        # Reset capture to beginning for the main loop
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_num = 0 # Reset frame counter

                        while True:
                            ret, frame = cap.read()
                            if not ret or frame is None: break # End of video or failed read

                            # Process frame (pass video_highest_status as context for display on the frame itself)
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

                            # If a new strictly higher status is found, save this frame as the "critical" one
                            if STATUS_HIERARCHY.get(new_highest_status, -1) > STATUS_HIERARCHY.get(video_highest_status, -1):
                                 video_highest_status = new_highest_status
                                 # Save this frame image, overwriting the previous critical frame candidate
                                 critical_frame_filename = f"critical_{unique_filename_prefix}_frame{frame_num}_{os.path.splitext(original_filename)[0]}.jpg"
                                 critical_frame_path = os.path.join(PROCESSED_FRAMES_FOLDER, critical_frame_filename)
                                 try:
                                     cv2.imwrite(critical_frame_path, processed_frame)
                                     critical_frame_url_to_save = url_for('static', filename=f'processed_frames/{critical_frame_filename}')
                                     print(f"Saved new critical frame (Status: {video_highest_status}) at {critical_frame_path}")
                                 except Exception as img_save_e:
                                     print(f"!!! ERROR saving critical frame image {critical_frame_path}: {img_save_e}")
                                     # Continue video processing but critical frame might be missing

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

# Optional: Add a route to serve files for download if static doesn't handle the 'download' attribute well
# @app.route('/download/processed_video/<filename>')
# def download_processed_video(filename):
#     try:
#         # Ensure the filename is safe before sending
#         safe_filename = secure_filename(filename)
#         return send_from_directory(app.config['PROCESSED_VIDEO_FOLDER'], safe_filename, as_attachment=True)
#     except FileNotFoundError:
#         print(f"Download file not found: {filename}")
#         return "File not found.", 404
#     except Exception as e:
#          print(f"Error serving download file {filename}: {e}")
#          return "Error serving file.", 500


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
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8) # Create a black image
        cv2.putText(blank_img, error_msg, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) # Add red error text
        ret_enc, buffer = cv2.imencode('.jpg', blank_img)
        if ret_enc:
             yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        # Optionally yield a few times or sleep to keep the error visible before ending
        # time.sleep(0.5) # Yield for 0.5 seconds
        return # Stop the generator after yielding error frame

    # Use 0 for default webcam, or provide a video file path for testing
    live_cap = cv2.VideoCapture(0)
    #live_cap = cv2.VideoCapture("videoplayback.mp4") # Example using local video file (replace with your file or use 0)


    if not live_cap.isOpened():
        print("!!! ERROR: Cannot open video source (webcam/file).")
        # Yield an error image frame
        error_msg = "Cannot open video source."
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8) # Create black image
        cv2.putText(blank_img, error_msg, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) # Add red error text
        ret_enc, buffer = cv2.imencode('.jpg', blank_img)
        if ret_enc:
             yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        # time.sleep(0.5)
        return

    frame_width = int(live_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(live_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Ensure valid dimensions
    if frame_width <= 0 or frame_height <= 0:
         print("!!! ERROR: Invalid frame dimensions from video source.")
         live_cap.release() # Release the capture object
         error_msg = "Invalid video source dimensions."
         blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
         cv2.putText(blank_img, error_msg, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
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
        if not ret or frame is None:
             print("End of live stream source (or camera disconnected).")
             break # End of stream (for file input) or camera disconnection

        # Process frame (pass overall_stream_status as context for display on frame, though function uses it little)
        # We get the *current frame's* specific status back
        processed_frame, frame_status, _ = process_media_content(
            frame, frame_width, frame_height, frame_num, overall_stream_status # Pass current highest for display context
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
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
    os.environ.pop('TF_CPP_MIN_LOG_LEVEL', None) # Clean up env variable on exit

'''

# # app.py

# # Required Imports
# from flask import Flask, render_template, request, url_for, Response, stream_with_context, redirect, send_from_directory # Added send_from_directory for download
# import os
# import cv2 # OpenCV
# import tensorflow as tf
# import tensorflow_hub as hub
# import numpy as np
# import sys
# import time
# from fluvio import Fluvio
# import json
# from werkzeug.utils import secure_filename
# import mimetypes
# import shutil # Import shutil for potential cleanup


# # --- Flask Application Setup ---
# app = Flask(__name__)

# # --- Configuration for Folders ---
# UPLOAD_FOLDER = 'uploads'
# STATIC_FOLDER = 'static'
# PROCESSED_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'processed_frames') # Will store critical frames
# PROCESSED_VIDEO_FOLDER = os.path.join(STATIC_FOLDER, 'processed_videos') # Will store full processed videos
# PROCESSED_IMAGE_FOLDER = os.path.join(STATIC_FOLDER, 'processed_images') # Will store processed images
# DEBUG_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'debug_frames')

# for folder in [UPLOAD_FOLDER, PROCESSED_FRAMES_FOLDER, PROCESSED_VIDEO_FOLDER, PROCESSED_IMAGE_FOLDER, DEBUG_FRAMES_FOLDER]:
#     if not os.path.exists(folder):
#         os.makedirs(folder)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['PROCESSED_VIDEO_FOLDER'] = PROCESSED_VIDEO_FOLDER # Add this for send_from_directory


# # --- Load Machine Learning Model ---
# DETECTOR_HANDLE = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
# detector = None
# try:
#     print(f"Loading detection model from: {DETECTOR_HANDLE}...")
#     detector = hub.load(DETECTOR_HANDLE)
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"FATAL ERROR: Could not load the detection model: {e}")
#     # sys.exit("Model loading failed. Exiting.") # Don't exit in webapp, just log error
#     detector = None # Ensure detector is None if loading fails

# # --- Model Specific Settings ---
# PERSON_CLASS_INDEX = 1
# DETECTION_THRESHOLD = 0.25

# # --- Density Analysis Settings ---
# HIGH_DENSITY_THRESHOLD = 6
# CRITICAL_DENSITY_THRESHOLD = 9
# HIGH_DENSITY_CELL_COUNT_THRESHOLD = 3
# CRITICAL_DENSITY_CELL_COUNT_THRESHOLD = 2
# GRID_ROWS = 8
# GRID_COLS = 8
# STATUS_HIERARCHY = {
#     "Normal": 0, "High Density Cell Detected": 1, "High Density Warning": 2,
#     "Critical Density Cell Detected": 3, "CRITICAL RISK": 4,
#     "Processing Started": -2, "Analysis Incomplete": -1, # Add processing states
#     "Error: Could not open input video": -3, "Error: Failed to initialize VideoWriter": -3,
#     "Error: Video writing failed": -3, "Error: Output video generation failed": -3,
#     "Error: Image processing failed": -3, "Error: Unsupported file type": -3
# }


# # --- Fluvio Settings ---
# FLUVIO_CROWD_TOPIC = "crowd-data"
# # Global variables for Fluvio client and producer
# fluvio_client = None
# fluvio_producer = None

# def connect_fluvio():
#     """Attempts to connect to Fluvio and create a topic producer."""
#     global fluvio_client, fluvio_producer # Declare intention to modify globals
#     if fluvio_producer: # Already connected
#          print("Fluvio producer already initialized.")
#          return True

#     print("Attempting to connect to Fluvio...")
#     sys.stdout.flush()
#     try:
#         # Assuming local default connection. Adjust if needed (e.g., Fluvio.connect("cloud_endpoint:9003"))
#         fluvio_client = Fluvio.connect()
#         print("Fluvio client connected successfully.")
#         sys.stdout.flush()

#         # Get a producer for the specified topic
#         fluvio_producer = fluvio_client.topic_producer(FLUVIO_CROWD_TOPIC)
#         print(f"Fluvio producer ready for topic '{FLUVIO_CROWD_TOPIC}'.")
#         sys.stdout.flush()
#         return True
#     except Exception as e:
#         print(f"!!! FLUVIO ERROR: Could not connect or get producer for topic '{FLUVIO_CROWD_TOPIC}'.")
#         print(f"    Error details: {e}")
#         print("    Check if Fluvio cluster is running and topic exists.")
#         sys.stdout.flush()
#         fluvio_client = None # Ensure client is None on failure
#         fluvio_producer = None # Ensure producer is None on failure
#         return False

# def send_to_fluvio(key, data_dict):
#     """Sends data dictionary as JSON to the configured Fluvio topic."""
#     global fluvio_producer # Access the global producer
#     if not fluvio_producer:
#         # print("Fluvio producer not available. Cannot send data.") # Can be noisy
#         return # Silently fail if producer not ready

#     try:
#         # Ensure key is bytes
#         key_bytes = str(key).encode('utf-8')
#         # Ensure data is JSON string, then encode to bytes
#         data_json_str = json.dumps(data_dict)
#         data_bytes = data_json_str.encode('utf-8')

#         # Send the record
#         # Using aio_send allows it to happen in the background without blocking the main thread
#         # Requires async context or event loop, which Flask doesn't have by default in this simple setup.
#         # For this simple setup, use blocking send or handle async explicitly if needed for high load.
#         # Let's stick to the basic send for simplicity here.
#         fluvio_producer.send(key_bytes, data_bytes)
#         # print(f"-> Sent data to Fluvio (Key: {key})") # Can be noisy, enable for debug
#     except Exception as e:
#         print(f"!!! FLUVIO WARNING: Could not send data (Key: {key}) to topic '{FLUVIO_CROWD_TOPIC}'.")
#         print(f"    Error details: {e}")
#         # Consider attempting to reconnect or flag the producer as potentially broken


# # --- Helper Functions ---
# def analyze_density_grid(density_grid):
#     """Analyzes the grid to determine status and risky cells."""
#     high_density_cells = 0
#     critical_density_cells = 0
#     risky_cell_coords = [] # List of (row, col) for risky cells
#     overall_status = "Normal"
#     total_people_in_grid = 0

#     if not density_grid or len(density_grid) != GRID_ROWS:
#         # print("   Warning: Invalid density grid received.") # Can be noisy
#         return overall_status, risky_cell_coords, total_people_in_grid, high_density_cells, critical_density_cells

#     for r_idx, row in enumerate(density_grid):
#         if len(row) != GRID_COLS: continue
#         for c_idx, count in enumerate(row):
#             try:
#                 person_count = int(count)
#                 total_people_in_grid += person_count
#                 if person_count >= CRITICAL_DENSITY_THRESHOLD:
#                     critical_density_cells += 1
#                     risky_cell_coords.append((r_idx, c_idx))
#                 elif person_count >= HIGH_DENSITY_THRESHOLD:
#                     high_density_cells += 1
#                     risky_cell_coords.append((r_idx, c_idx))
#             except (ValueError, TypeError):
#                 continue # Skip invalid cell counts

#     # Determine overall status based on cell counts
#     if critical_density_cells >= CRITICAL_DENSITY_CELL_COUNT_THRESHOLD:
#         overall_status = "CRITICAL RISK"
#     elif critical_density_cells > 0:
#         overall_status = "Critical Density Cell Detected"
#     elif high_density_cells >= HIGH_DENSITY_CELL_COUNT_THRESHOLD:
#         overall_status = "High Density Warning"
#     elif high_density_cells > 0:
#         overall_status = "High Density Cell Detected"

#     # Return all calculated values
#     return overall_status, risky_cell_coords, total_people_in_grid, high_density_cells, critical_density_cells


# def get_higher_priority_status(status1, status2):
#     """Compares two status strings and returns the one with higher priority."""
#     p1 = STATUS_HIERARCHY.get(status1, -1)
#     p2 = STATUS_HIERARCHY.get(status2, -1)
#     return status1 if p1 >= p2 else status2


# # --- Frame/Image Processing Function ---
# def process_media_content(content, content_width, content_height, frame_or_image_index, current_overall_status_context="Normal"):
#     """
#     Processes a single image or video frame: detects people, calculates density,
#     determines status, draws overlays/text, AND sends data to Fluvio.
#     Returns the processed content (image/frame), content status, and confirmed person count.
#     """
#     if content is None:
#         # print(f"Warning: Received None content for processing at index {frame_or_image_index}")
#         return None, current_overall_status_context, 0 # Return context status, 0 people

#     if detector is None:
#          # Handle case where model loading failed at startup
#          error_frame = content.copy()
#          cv2.putText(error_frame, "MODEL ERROR!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#          print(f"!!! Skipping processing for index {frame_or_image_index}: ML model not loaded.")
#          return error_frame, "Error: Model Not Loaded", 0


#     start_process_time = time.time()
#     processed_content = content.copy() # Always work on a copy
#     content_status = "Analysis Incomplete" # Default status for this frame
#     confirmed_person_count_this_content = 0
#     density_grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
#     risky_coords = [] # Initialize risky coords list


#     try:
#         # --- 1. ML Preprocessing & Detection ---
#         # Ensure image is in the correct color space for the model (often RGB)
#         if len(content.shape) == 2: rgb_content = cv2.cvtColor(content, cv2.COLOR_GRAY2RGB)
#         elif content.shape[2] == 4: rgb_content = cv2.cvtColor(content, cv2.COLOR_BGRA2RGB)
#         elif content.shape[2] == 3: rgb_content = cv2.cvtColor(content, cv2.COLOR_BGR2RGB) # Assuming typical BGR input from OpenCV
#         else: rgb_content = content # Fallback, hope for the best

#         image_tensor = tf.expand_dims(tf.convert_to_tensor(rgb_content, dtype=tf.uint8), axis=0)
#         detections = detector(image_tensor)

#         # Filter detections based on score threshold
#         # Using list comprehension for clarity
#         valid_detections_indices = [
#             i for i in range(detections['detection_scores'][0].shape[0])
#             if detections['detection_scores'][0].numpy()[i] >= DETECTION_THRESHOLD
#             and detections['detection_classes'][0].numpy().astype(int)[i] == PERSON_CLASS_INDEX
#         ]

#         boxes = detections['detection_boxes'][0].numpy()[valid_detections_indices]
#         classes = detections['detection_classes'][0].numpy().astype(int)[valid_detections_indices]
#         scores = detections['detection_scores'][0].numpy()[valid_detections_indices]


#         # --- 2. Calculate Grid & Filter Detections ---
#         cell_height = content_height // GRID_ROWS
#         cell_width = content_width // GRID_COLS
#         if cell_height <= 0 or cell_width <= 0:
#             print(f"Error: Content dimensions ({content_width}x{content_height}) too small for grid size ({GRID_COLS}x{GRID_ROWS}). Skipping density analysis.")
#             # Still draw detections if any, but density grid won't be accurate/possible
#             content_status = "Analysis Incomplete" # Or specific error status
#             confirmed_person_count_this_content = len(boxes) # Count valid detections even without grid
#             # Skip density grid processing below
#         else:
#              confirmed_person_count_this_content = len(boxes) # Count of persons above threshold

#              for i in range(boxes.shape[0]):
#                  # Note: boxes are [ymin, xmin, ymax, xmax] in relative coordinates (0 to 1)
#                  ymin, xmin, ymax, xmax = boxes[i]
#                  # Calculate center point in pixel coordinates
#                  center_x = int((xmin + xmax) / 2 * content_width)
#                  center_y = int((ymin + ymax) / 2 * content_height)

#                  # Determine grid cell
#                  row = min(max(0, center_y // cell_height), GRID_ROWS - 1)
#                  col = min(max(0, center_x // cell_width), GRID_COLS - 1)

#                  # Increment density count for the cell
#                  if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
#                      density_grid[row][col] += 1


#              # --- 3. Analyze Density Grid ---
#              content_status, risky_coords, total_grid_people, high_cells, crit_cells = analyze_density_grid(density_grid)


#         # --- 4. Send Data to Fluvio ---
#         fluvio_payload = {
#             "timestamp": int(time.time()), # Current timestamp
#             "frame": frame_or_image_index, # Frame number or image index (e.g., 0 for single image)
#             "density_grid": density_grid, # The calculated grid (will be [[0...]] if grid analysis skipped)
#             "frame_status": content_status, # Status determined for this frame
#             "confirmed_persons": confirmed_person_count_this_content, # Persons detected above threshold
#             "high_density_cells": high_cells if 'high_cells' in locals() else 0, # Count of cells >= HIGH_DENSITY_THRESHOLD
#             "critical_density_cells": crit_cells if 'crit_cells' in locals() else 0 # Count of cells >= CRITICAL_DENSITY_THRESHOLD
#         }
#         # Use a composite key including original filename or source identifier if available
#         # For simplicity here, using a generic key with index.
#         send_to_fluvio(f"content-{frame_or_image_index}", fluvio_payload)


#         # --- 5. Draw Overlays and Text ---
#         # Draw grid lines (optional but helpful for visualization)
#         grid_line_color = (200, 200, 200) # Light gray
#         grid_line_thickness = 1
#         for r in range(1, GRID_ROWS):
#             cv2.line(processed_content, (0, r * cell_height), (content_width, r * cell_height), grid_line_color, grid_line_thickness)
#         for c in range(1, GRID_COLS):
#             cv2.line(processed_content, (c * cell_width, 0), (c * cell_width, content_height), grid_line_color, grid_line_thickness)


#         # Draw overlays for individual risky grid cells
#         overlay_alpha = 0.4
#         overlay_color_critical = (0, 0, 255) # Red (BGR)
#         overlay_color_high = (0, 165, 255) # Orange (BGR)

#         for r, c in risky_coords:
#             cell_y_start = r * cell_height
#             cell_y_end = (r + 1) * cell_height
#             cell_x_start = c * cell_width
#             cell_x_end = (c + 1) * cell_width

#             risk_level_in_cell = "unknown"
#             try:
#                 if density_grid[r][c] >= CRITICAL_DENSITY_THRESHOLD: risk_level_in_cell = "critical"
#                 elif density_grid[r][c] >= HIGH_DENSITY_THRESHOLD: risk_level_in_cell = "high"
#             except IndexError:
#                  continue # Should not happen with robust index calculation, but safety first

#             color = overlay_color_critical if risk_level_in_cell == "critical" else overlay_color_high
#             overlay = processed_content.copy() # Create a fresh overlay copy
#             cv2.rectangle(overlay, (cell_x_start, cell_y_start), (cell_x_end, cell_y_end), color, -1)
#             # Blend the overlay onto the processed_content
#             cv2.addWeighted(overlay, overlay_alpha, processed_content, 1 - overlay_alpha, 0, processed_content)

#             # Optionally draw person count in the cell
#             # if density_grid[r][c] > 0:
#             #     count_text = str(density_grid[r][c])
#             #     text_origin = (cell_x_start + 5, cell_y_end - 5) # Bottom-left corner
#             #     cv2.putText(processed_content, count_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


#         # Draw Detections (Optional, can clutter the view with overlays)
#         # for i in range(boxes.shape[0]):
#         #     ymin, xmin, ymax, xmax = boxes[i]
#         #     x1, y1 = int(xmin * content_width), int(ymin * content_height)
#         #     x2, y2 = int(xmax * content_width), int(ymax * content_height)
#         #     cv2.rectangle(processed_content, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green rectangle


#         # Draw Status Text & Stampede Chance Text
#         # Use overall status context for potentially sticky status display across frames
#         # However, for the *output frame*, we should show *its* specific status.
#         frame_display_status = content_status # Status specific to this frame

#         status_text = f"Risk: {frame_display_status}"
#         status_color = (0, 128, 0) # Green default
#         if "CRITICAL" in frame_display_status: status_color = (0, 0, 255) # Red
#         elif "Warning" in frame_display_status or "High" in frame_display_status or "Detected" in frame_display_status: status_color = (0, 165, 255) # Orange
#         elif "Error" in frame_display_status or "Incomplete" in frame_display_status: status_color = (0, 0, 255) # Red for errors

#         # Stampede Chance Text determination based on frame status
#         if "CRITICAL" in frame_display_status: chance_text, chance_color = "Stampede Chance: Critical", (0, 0, 255)
#         elif "Warning" in frame_display_status or "High" in frame_display_status or "Detected" in frame_display_status: chance_text, chance_color = "Stampede Chance: High", (0, 165, 255)
#         else: chance_text, chance_color = "Stampede Chance: Low", (0, 128, 0)


#         # Add text with background rectangles
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.7
#         font_thickness = 2
#         padding = 5
#         bg_color = (50, 50, 50) # Dark gray background

#         # Status text
#         (text_w_s, text_h_s), baseline_s = cv2.getTextSize(status_text, font, font_scale, font_thickness)
#         cv2.rectangle(processed_content, (padding, padding), (padding * 2 + text_w_s, padding * 2 + text_h_s), bg_color, -1)
#         cv2.putText(processed_content, status_text, (padding + 5, padding + text_h_s + 5), font, font_scale, status_color, font_thickness, cv2.LINE_AA)

#         # Chance text
#         (text_w_c, text_h_c), baseline_c = cv2.getTextSize(chance_text, font, font_scale, font_thickness)
#         cv2.rectangle(processed_content, (padding, content_height - padding * 2 - text_h_c), (padding * 2 + text_w_c, content_height - padding), bg_color, -1)
#         cv2.putText(processed_content, chance_text, (padding + 5, content_height - padding - baseline_c), font, font_scale, chance_color, font_thickness, cv2.LINE_AA)


#     except Exception as e:
#         print(f"!!! ERROR during process_media_content for index {frame_or_image_index}: {e}")
#         # Draw error text on frame
#         error_frame = content.copy() # Start with original again to avoid drawing on potentially corrupted processed_content
#         cv2.putText(error_frame, f"Processing Error: {e}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
#         # Return original content + error status + 0 count
#         return error_frame, f"Error: {e}", 0

#     # end_process_time = time.time()
#     # print(f" -> Index {frame_or_image_index} processed in {end_process_time - start_process_time:.3f}s. Status: {content_status}")

#     # Return processed frame, its specific status, and person count
#     return processed_content, content_status, confirmed_person_count_this_content


# # --- Flask Routes ---
# @app.route('/', methods=['GET'])
# def index_route():
#     """Serves the main page."""
#     return render_template('index.html')

# @app.route('/upload_media', methods=['POST'])
# def upload_media_route():
#     print("\n--- Request received for /upload_media ---")
#     if fluvio_producer:
#          print("   Fluvio Status: Producer is active.")
#     else:
#          print("   Fluvio Status: Producer is INACTIVE.")
#     sys.stdout.flush()

#     if detector is None:
#         print("!!! ERROR: ML model not loaded. Cannot process media.")
#         return render_template('results.html',
#                                output_media_type=None,
#                                processed_media_url=None,
#                                download_video_url=None,
#                                prediction_status="Error: Model Not Loaded",
#                                max_persons=0,
#                                processing_time="N/A")

#     start_time = time.time()

#     if 'media' not in request.files: return 'No media file part in the request', 400
#     media_file = request.files['media']
#     if media_file.filename == '': return 'No selected media file', 400

#     original_filename = secure_filename(media_file.filename)
#     # Use a timestamp or UUID to make upload filenames unique and avoid conflicts
#     unique_filename_prefix = str(int(time.time())) # Simple timestamp prefix
#     unique_original_filename = f"{unique_filename_prefix}_{original_filename}"

#     upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_original_filename)
#     try: media_file.save(upload_path)
#     except Exception as e:
#         print(f"Error saving media file: {e}")
#         return f"Error saving media file: {e}", 500
#     print(f"Media saved temporarily to: {upload_path}")

#     mimetype = mimetypes.guess_type(upload_path)[0]
#     file_type = 'unknown'
#     if mimetype:
#         if mimetype.startswith('video/'): file_type = 'video'
#         elif mimetype.startswith('image/'): file_type = 'image'
#     print(f"Detected file type: {file_type}")

#     # --- Initialize vars for results template ---
#     processed_media_url = None # URL for the image to display (processed image or critical frame)
#     download_video_url = None  # URL for the full video (only if video)
#     overall_processing_status = "Processing Started" # Overall status for the entire file/video
#     max_persons = 0
#     output_media_type = file_type # Use detected type for the template

#     # --- Process Video ---
#     if file_type == 'video':
#         output_video_filename = f"processed_{unique_original_filename}"
#         # Ensure video filename has an mp4 extension
#         if not output_video_filename.lower().endswith('.mp4'):
#              output_video_filename = os.path.splitext(output_video_filename)[0] + ".mp4"

#         output_video_path = os.path.join(PROCESSED_VIDEO_FOLDER, output_video_filename)

#         cap = cv2.VideoCapture(upload_path)

#         if not cap.isOpened():
#             print(f"!!! ERROR: Failed to open input video file: {upload_path}")
#             overall_processing_status = "Error: Could not open input video"
#             # Fall through to cleanup and render error
#         else:
#             try: # Wrap processing
#                 fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps and fps > 0 else 25.0
#                 width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                 height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                 print(f"Video Input: {width}x{height} @ {fps:.2f} FPS")

#                 # Use 'mp4v' or 'avc1' (H.264) - 'avc1' is generally better supported
#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                 # Check if 'avc1' is available/preferred - might require specific OpenCV build or system codecs
#                 # try:
#                 #     test_writer = cv2.VideoWriter()
#                 #     if test_writer.open(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height)):
#                 #         fourcc = cv2.VideoWriter_fourcc(*'avc1')
#                 #         test_writer.release()
#                 #         print("Using avc1 codec.")
#                 #     else:
#                 #          print("avc1 codec not available, falling back to mp4v.")
#                 # except Exception as codec_e:
#                 #      print(f"Codec test failed ({codec_e}), falling back to mp4v.")
#                 #      fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Re-assign if test failed


#                 out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
#                 print(f"Attempting to open VideoWriter for: {output_video_path} with FOURCC: {fourcc}")

#                 if not out_video.isOpened():
#                     print(f"!!! ERROR: Failed to initialize VideoWriter for {output_video_path}. Check codecs and permissions.")
#                     overall_processing_status = "Error: Failed to initialize VideoWriter"
#                 else:
#                     print("VideoWriter opened successfully. Starting frame processing loop...")
#                     frame_num = 0
#                     video_highest_status = "Normal" # Track highest status found in video
#                     critical_frame_url_to_save = None # Will store the path to the image file of the critical frame

#                     while True:
#                         ret, frame = cap.read()
#                         if not ret: break # End of video

#                         # Process frame (pass video_highest_status for context, though process_media_content uses it for text drawing)
#                         # We get the *current frame's* status back
#                         processed_frame, current_frame_status, people_count = process_media_content(
#                             frame, width, height, frame_num, video_highest_status # Pass current highest status for display context
#                         )

#                         if processed_frame is None:
#                             print(f"Warning: Skipping frame {frame_num} due to processing failure.")
#                             frame_num += 1
#                             continue # Skip writing/analyzing this frame if processing failed

#                         # Check if this frame's status is higher priority than the highest seen so far
#                         new_highest_status = get_higher_priority_status(video_highest_status, current_frame_status)

#                         # If a new highest status is found, save this frame as the "critical" one
#                         # We overwrite the previous critical frame if a higher status is found later
#                         if STATUS_HIERARCHY.get(new_highest_status, -1) > STATUS_HIERARCHY.get(video_highest_status, -1):
#                              video_highest_status = new_highest_status
#                              # Save this frame image
#                              critical_frame_filename = f"critical_{unique_filename_prefix}_frame{frame_num}_{os.path.splitext(original_filename)[0]}.jpg"
#                              critical_frame_path = os.path.join(PROCESSED_FRAMES_FOLDER, critical_frame_filename)
#                              try:
#                                  cv2.imwrite(critical_frame_path, processed_frame)
#                                  critical_frame_url_to_save = url_for('static', filename=f'processed_frames/{critical_frame_filename}')
#                                  print(f"Saved new critical frame (Status: {video_highest_status}) at {critical_frame_path}")
#                              except Exception as img_save_e:
#                                  print(f"!!! ERROR saving critical frame image {critical_frame_path}: {img_save_e}")
#                                  # Continue video processing but critical frame might be missing

#                         # Update max persons found in the entire video
#                         max_persons = max(max_persons, people_count)

#                         # Write frame to the full output video
#                         try:
#                             out_video.write(processed_frame)
#                         except Exception as write_e:
#                            print(f"!!! ERROR writing frame {frame_num} to video: {write_e}")
#                            # Stop processing the video if writing fails
#                            overall_processing_status = f"Error: Video writing failed frame {frame_num}"
#                            break

#                         frame_num += 1
#                         # if frame_num % 50 == 0: print(f"  Processed frame {frame_num}...") # Optional progress


#                     print(f"Frame processing loop finished after {frame_num} frames.")
#                     overall_processing_status = video_highest_status # Final video status is the highest encountered

#                     # Release VideoWriter FIRST
#                     if out_video.isOpened():
#                         out_video.release()
#                         print("VideoWriter released.")

#                     # Check if the output video file was successfully created and is not empty
#                     if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
#                          download_video_url = url_for('static', filename=f'processed_videos/{output_video_filename}')
#                          print(f"Full processed video created: {output_video_path}")
#                          # If no critical frame was found (e.g., video had no detections above threshold),
#                          # maybe save the first frame as a fallback image to display?
#                          if critical_frame_url_to_save is None and frame_num > 0:
#                              print("No critical frame saved, attempting to save the first frame as fallback display.")
#                              cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Go back to first frame
#                              ret_first, first_frame = cap.read()
#                              if ret_first:
#                                  # Reprocess the first frame to get overlays if needed, or just save raw?
#                                  # Saving processed first frame is better.
#                                  processed_first_frame, _, _ = process_media_content(first_frame, width, height, 0, "Normal")
#                                  if processed_first_frame is not None:
#                                     fallback_frame_filename = f"fallback_{unique_filename_prefix}_frame0_{os.path.splitext(original_filename)[0]}.jpg"
#                                     fallback_frame_path = os.path.join(PROCESSED_FRAMES_FOLDER, fallback_frame_filename)
#                                     try:
#                                         cv2.imwrite(fallback_frame_path, processed_first_frame)
#                                         critical_frame_url_to_save = url_for('static', filename=f'processed_frames/{fallback_frame_filename}')
#                                         print(f"Saved fallback display frame at {fallback_frame_path}")
#                                     except Exception as fb_save_e:
#                                          print(f"!!! ERROR saving fallback frame: {fb_save_e}")


#                     else:
#                          print(f"!!! ERROR: Output video file missing or empty after processing: {output_video_path}")
#                          # If video creation failed, maybe the critical frame image is still useful to show?
#                          if critical_frame_url_to_save is None: # If critical frame also failed to save
#                               overall_processing_status = get_higher_priority_status(overall_processing_status, "Error: Output video generation failed")
#                          # Else, keep the critical frame status/url, but add a warning about the video download

#                     processed_media_url = critical_frame_url_to_save # This is the URL for the *display* image (critical frame)


#             except Exception as video_proc_e:
#                 print(f"!!! ERROR during video processing: {video_proc_e}")
#                 overall_processing_status = get_higher_priority_status(overall_processing_status, f"Error: Unexpected failure during video processing - {video_proc_e}")
#                 # Try to cleanup partially written video file
#                 if os.path.exists(output_video_path):
#                      try:
#                          os.remove(output_video_path)
#                          print(f"Removed incomplete video file: {output_video_path}")
#                      except OSError as cleanup_e:
#                          print(f"Warning: Could not remove incomplete video file {output_video_path}: {cleanup_e}")

#             finally:
#                 if cap.isOpened(): cap.release(); print("VideoCapture released.")


#     # --- Process Image ---
#     elif file_type == 'image':
#         output_image_filename = f"processed_{unique_original_filename}"
#         # Ensure filename ends with a common image extension like .jpg
#         if not output_image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#              # Get original extension or default to jpg
#              ext = os.path.splitext(unique_original_filename)[1]
#              if ext and ext.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'):
#                  output_image_filename = os.path.splitext(output_image_filename)[0] + ext # Keep original extension
#              else:
#                 output_image_filename = os.path.splitext(output_image_filename)[0] + ".jpg" # Default to jpg

#         output_image_path = os.path.join(PROCESSED_IMAGE_FOLDER, output_image_filename)

#         try:
#             image = cv2.imread(upload_path)
#             if image is None:
#                raise ValueError("Could not read image file with OpenCV.")
#             height, width = image.shape[:2] # Get height and width

#             print(f"Image Input: {width}x{height}")

#             # Process the single image (frame_index=0, status starts 'Normal')
#             processed_image, image_status, people_count = process_media_content(
#                 image, width, height, 0, "Normal"
#             )

#             if processed_image is None:
#                raise ValueError("Image processing function returned None.")

#             overall_processing_status = image_status # Overall status is just the result of this one image
#             max_persons = people_count

#             # Save the processed image
#             save_success = cv2.imwrite(output_image_path, processed_image)
#             if not save_success:
#                 raise ValueError(f"Failed to save processed image to {output_image_path}. Check permissions.")

#             print(f"Processed image saved to: {output_image_path}")
#             processed_media_url = url_for('static', filename=f'processed_images/{output_image_filename}')
#             download_video_url = None # No video to download for image upload


#         except Exception as img_proc_e:
#            print(f"!!! ERROR during image processing: {img_proc_e}")
#            overall_processing_status = get_higher_priority_status(overall_processing_status, f"Error: Image processing failed - {img_proc_e}")
#            processed_media_url = None # Indicate failure
#            download_video_url = None


#     # --- Handle Unknown File Type ---
#     else:
#         print(f"Unsupported file type: {mimetype}")
#         overall_processing_status = "Error: Unsupported file type"
#         processed_media_url = None
#         download_video_url = None

#     # --- Cleanup Upload ---
#     try:
#         os.remove(upload_path)
#         print(f"Removed temporary upload: {upload_path}")
#     except OSError as e:
#         print(f"Warning: Could not remove temporary upload {upload_path}: {e}")

#     processing_time_secs = time.time() - start_time
#     print(f"Total request processing time: {processing_time_secs:.2f} seconds")

#     # --- Render Results ---
#     print(f"---> Rendering results page:")
#     print(f"     Media Type: {output_media_type}")
#     print(f"     Displayed Media URL (Image/Critical Frame): {processed_media_url}")
#     print(f"     Download Video URL: {download_video_url}")
#     print(f"     Overall Status: {overall_processing_status}")
#     print(f"     Max Persons: {max_persons}")

#     return render_template('results.html',
#                            output_media_type=output_media_type,
#                            processed_media_url=processed_media_url, # This is the image URL
#                            download_video_url=download_video_url,   # This is the video URL (or None)
#                            prediction_status=overall_processing_status,
#                            max_persons=max_persons,
#                            processing_time=f"{processing_time_secs:.2f}")

# # --- Add a route to serve files for download ---
# # This is needed if the download link doesn't work directly from static
# # Flask's static handler should work, but this provides an alternative if needed.
# # @app.route('/download/processed_video/<filename>')
# # def download_processed_video(filename):
# #     try:
# #         return send_from_directory(app.config['PROCESSED_VIDEO_FOLDER'], filename, as_attachment=True)
# #     except FileNotFoundError:
# #         return "File not found.", 404
# #     except Exception as e:
# #          print(f"Error serving download file {filename}: {e}")
# #          return "Error serving file.", 500


# # --- Live Stream Route ---
# def generate_live_frames():
#     print("\n--- Request received for /video_feed (Live Stream) ---")
#     if fluvio_producer:
#          print("   Fluvio Status: Producer is active.")
#     else:
#          print("   Fluvio Status: Producer is INACTIVE.")
#     sys.stdout.flush()

#     if detector is None:
#         print("!!! ERROR: ML model not loaded. Cannot start live stream.")
#         # Yield an error image or message instead of frames
#         error_msg = "ML Model Loading Failed. Cannot stream."
#         blank_img = np.zeros((480, 640, 3), dtype=np.uint8) # Blank black image
#         cv2.putText(blank_img, error_msg, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#         ret_enc, buffer = cv2.imencode('.jpg', blank_img)
#         if ret_enc:
#              yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#         time.sleep(5) # Keep error frame visible for a moment
#         return # Stop the generator


#     # Use 0 for webcam, or provide a video file path
#     # live_cap = cv2.VideoCapture(0)
#     live_cap = cv2.VideoCapture("videoplayback.mp4") # Example using local video file

#     if not live_cap.isOpened():
#         print("!!! ERROR: Cannot open video source (webcam/file).")
#         error_msg = "Cannot open video source."
#         blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
#         cv2.putText(blank_img, error_msg, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#         ret_enc, buffer = cv2.imencode('.jpg', blank_img)
#         if ret_enc:
#              yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#         time.sleep(5) # Keep error frame visible for a moment
#         return

#     frame_width = int(live_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(live_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     # Ensure valid dimensions for processing
#     if frame_width <= 0 or frame_height <= 0:
#          print("!!! ERROR: Invalid frame dimensions from video source.")
#          live_cap.release()
#          error_msg = "Invalid video source dimensions."
#          blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
#          cv2.putText(blank_img, error_msg, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#          ret_enc, buffer = cv2.imencode('.jpg', blank_img)
#          if ret_enc:
#              yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#          time.sleep(5)
#          return


#     print(f"Live source opened: {frame_width}x{frame_height}")

#     frame_num = 0
#     overall_stream_status = "Normal" # Track highest status seen *during* the stream session

#     while True:
#         ret, frame = live_cap.read()
#         if not ret:
#              print("End of live stream source.")
#              break # End of stream (for file input) or camera disconnected

#         # Process frame (pass overall_stream_status as context for display)
#         processed_frame, frame_status, _ = process_media_content(
#             frame, frame_width, frame_height, frame_num, overall_stream_status
#         )

#         # Update the overall status for the stream session
#         overall_stream_status = get_higher_priority_status(overall_stream_status, frame_status)

#         if processed_frame is None:
#             print(f"Warning: Skipping live frame {frame_num} due to processing failure.")
#             frame_num += 1
#             continue # Skip encoding/yielding this frame


#         try:
#             # Encode processed frame as JPEG for streaming
#             ret_enc, buffer = cv2.imencode('.jpg', processed_frame)
#             if not ret_enc:
#                 print(f"Error encoding live frame {frame_num}.")
#                 frame_num += 1
#                 continue # Skip if encoding fails

#             frame_bytes = buffer.tobytes()
#             # Yield the frame in a multi-part response
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#         except Exception as e:
#             print(f"Error encoding/yielding live frame {frame_num}: {e}")
#             # Consider yielding an error frame before breaking
#             break # Stop stream on error

#         frame_num += 1
#         # Optional: Add a small delay if processing is too fast and consuming excessive CPU/bandwidth
#         # time.sleep(0.01) # Example: 10ms delay


#     live_cap.release()
#     print("Live stream generator finished.")


# @app.route('/live')
# def live_route():
#     """Serves the live stream page."""
#     # Ensure Fluvio connection is attempted before rendering live page
#     connect_fluvio()
#     return render_template('live.html')

# @app.route('/video_feed')
# def video_feed_route():
#     """Provides the MJPEG stream for the live page."""
#     # The generator function handles the actual frame processing and yielding
#     return Response(generate_live_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# # --- Run App ---
# if __name__ == '__main__':
#     print("--- Initializing Application ---")
#     # Attempt to connect to Fluvio when the application starts
#     # This connection will persist for the lifetime of the process
#     fluvio_connected = connect_fluvio()
#     if not fluvio_connected:
#          print("!!! WARNING: Fluvio connection failed on startup. Data will not be sent.")
#     else:
#          print("+++ Fluvio connection successful on startup.")

#     # Attempt to load the ML model on startup
#     # The global 'detector' variable is set by the initial load attempt.
#     if detector is None:
#         print("!!! WARNING: ML model failed to load on startup. Media processing and live streams will not work.")

#     print("--- Starting Flask Server ---")
#     # use_reloader=False is often important with background threads/connections (like Fluvio)
#     # threaded=True is needed for handling concurrent requests (like /upload_media and /video_feed simultaneously)
#     # host='0.0.0.0' makes the server accessible from external IPs, useful in Docker/deployments
#     app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)

























# # app.py

# # Required Imports
# from flask import Flask, render_template, request, url_for, Response, stream_with_context, redirect
# import os
# import cv2 # OpenCV
# import tensorflow as tf
# import tensorflow_hub as hub
# import numpy as np
# import sys
# import time
# from fluvio import Fluvio # <-- UNCOMMENTED
# import json
# from werkzeug.utils import secure_filename
# import mimetypes

# # --- Flask Application Setup ---
# app = Flask(__name__)

# # --- Configuration for Folders ---
# # ... (folder setup remains the same) ...
# UPLOAD_FOLDER = 'uploads'
# STATIC_FOLDER = 'static'
# PROCESSED_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'processed_frames')
# PROCESSED_VIDEO_FOLDER = os.path.join(STATIC_FOLDER, 'processed_videos')
# PROCESSED_IMAGE_FOLDER = os.path.join(STATIC_FOLDER, 'processed_images') # New folder for images
# DEBUG_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'debug_frames')

# for folder in [UPLOAD_FOLDER, PROCESSED_FRAMES_FOLDER, PROCESSED_VIDEO_FOLDER, PROCESSED_IMAGE_FOLDER, DEBUG_FRAMES_FOLDER]:
#     if not os.path.exists(folder):
#         os.makedirs(folder)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # --- Load Machine Learning Model ---
# # ... (model loading remains the same) ...
# DETECTOR_HANDLE = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
# detector = None
# try:
#     print(f"Loading detection model from: {DETECTOR_HANDLE}...")
#     detector = hub.load(DETECTOR_HANDLE)
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"FATAL ERROR: Could not load the detection model: {e}")
#     sys.exit("Model loading failed. Exiting.")

# # --- Model Specific Settings ---
# PERSON_CLASS_INDEX = 1
# DETECTION_THRESHOLD = 0.25
# # DEBUG_LOGGING_THRESHOLD = 0.1 # Keep if used, otherwise remove

# # --- Density Analysis Settings ---
# # ... (density settings remain the same) ...
# HIGH_DENSITY_THRESHOLD = 6
# CRITICAL_DENSITY_THRESHOLD = 9
# HIGH_DENSITY_CELL_COUNT_THRESHOLD = 3
# CRITICAL_DENSITY_CELL_COUNT_THRESHOLD = 2
# GRID_ROWS = 8
# GRID_COLS = 8
# STATUS_HIERARCHY = {
#     "Normal": 0, "High Density Cell Detected": 1, "High Density Warning": 2,
#     "Critical Density Cell Detected": 3, "CRITICAL RISK": 4
# }

# # --- Fluvio Settings ---
# FLUVIO_CROWD_TOPIC = "crowd-data"
# # Global variables for Fluvio client and producer
# fluvio_client = None
# fluvio_producer = None

# def connect_fluvio():
#     """Attempts to connect to Fluvio and create a topic producer."""
#     global fluvio_client, fluvio_producer # Declare intention to modify globals
#     if fluvio_producer: # Already connected
#          print("Fluvio producer already initialized.")
#          return True

#     print("Attempting to connect to Fluvio...")
#     sys.stdout.flush()
#     try:
#         # Assuming local default connection. Adjust if needed (e.g., Fluvio.connect("cloud_endpoint:9003"))
#         fluvio_client = Fluvio.connect()
#         print("Fluvio client connected successfully.")
#         sys.stdout.flush()

#         # Get a producer for the specified topic
#         fluvio_producer = fluvio_client.topic_producer(FLUVIO_CROWD_TOPIC)
#         print(f"Fluvio producer ready for topic '{FLUVIO_CROWD_TOPIC}'.")
#         sys.stdout.flush()
#         return True
#     except Exception as e:
#         print(f"!!! FLUVIO ERROR: Could not connect or get producer for topic '{FLUVIO_CROWD_TOPIC}'.")
#         print(f"    Error details: {e}")
#         print("    Check if Fluvio cluster is running and topic exists.")
#         sys.stdout.flush()
#         fluvio_client = None # Ensure client is None on failure
#         fluvio_producer = None # Ensure producer is None on failure
#         return False

# def send_to_fluvio(key, data_dict):
#     """Sends data dictionary as JSON to the configured Fluvio topic."""
#     global fluvio_producer # Access the global producer
#     if not fluvio_producer:
#         # print("Fluvio producer not available. Cannot send data.") # Can be noisy
#         return # Silently fail if producer not ready

#     try:
#         # Ensure key is bytes
#         key_bytes = str(key).encode('utf-8')
#         # Ensure data is JSON string, then encode to bytes
#         data_json_str = json.dumps(data_dict)
#         data_bytes = data_json_str.encode('utf-8')

#         # Send the record
#         fluvio_producer.send(key_bytes, data_bytes)
#         # print(f"-> Sent data to Fluvio (Key: {key})") # Can be noisy, enable for debug
#     except Exception as e:
#         print(f"!!! FLUVIO WARNING: Could not send data (Key: {key}) to topic '{FLUVIO_CROWD_TOPIC}'.")
#         print(f"    Error details: {e}")
#         # Consider attempting to reconnect or flag the producer as potentially broken
#         # global fluvio_producer # If you want to reset it
#         # fluvio_producer = None

# # --- Helper Functions ---
# # (analyze_density_grid, get_higher_priority_status remain the same)
# def analyze_density_grid(density_grid):
#     # ... (existing code - make sure it returns all needed values) ...
#     high_density_cells = 0
#     critical_density_cells = 0
#     risky_cell_coords = [] # List of (row, col) for risky cells
#     overall_status = "Normal"
#     total_people_in_grid = 0

#     if not density_grid or len(density_grid) != GRID_ROWS:
#         # print("   Warning: Invalid density grid received.") # Can be noisy
#         return overall_status, risky_cell_coords, total_people_in_grid, high_density_cells, critical_density_cells

#     for r_idx, row in enumerate(density_grid):
#         if len(row) != GRID_COLS: continue
#         for c_idx, count in enumerate(row):
#             try:
#                 person_count = int(count)
#                 total_people_in_grid += person_count
#                 if person_count >= CRITICAL_DENSITY_THRESHOLD:
#                     critical_density_cells += 1
#                     risky_cell_coords.append((r_idx, c_idx))
#                 elif person_count >= HIGH_DENSITY_THRESHOLD:
#                     high_density_cells += 1
#                     risky_cell_coords.append((r_idx, c_idx))
#             except (ValueError, TypeError):
#                 continue

#     # Determine overall status
#     if critical_density_cells >= CRITICAL_DENSITY_CELL_COUNT_THRESHOLD:
#         overall_status = "CRITICAL RISK"
#     elif critical_density_cells > 0:
#         overall_status = "Critical Density Cell Detected"
#     elif high_density_cells >= HIGH_DENSITY_CELL_COUNT_THRESHOLD:
#         overall_status = "High Density Warning"
#     elif high_density_cells > 0:
#         overall_status = "High Density Cell Detected"

#     # Return all calculated values, including cell counts
#     return overall_status, risky_cell_coords, total_people_in_grid, high_density_cells, critical_density_cells


# def get_higher_priority_status(status1, status2):
#     # ... (existing code) ...
#     p1 = STATUS_HIERARCHY.get(status1, -1)
#     p2 = STATUS_HIERARCHY.get(status2, -1)
#     return status1 if p1 >= p2 else status2


# # --- Frame/Image Processing Function ---
# def process_media_content(content, content_width, content_height, frame_or_image_index, current_overall_status="Normal"):
#     """
#     Processes a single image or video frame: detects people, calculates density,
#     determines status, draws overlays/text, AND sends data to Fluvio.
#     Returns the processed content (image/frame), content status, and confirmed person count.
#     """
#     if content is None:
#         # print(f"Warning: Received None content for processing at index {frame_or_image_index}")
#         return None, current_overall_status, 0

#     start_process_time = time.time()
#     processed_content = content # Start with original in case of early error
#     content_status = current_overall_status
#     confirmed_person_count_this_content = 0

#     try:
#         # --- 1. ML Preprocessing & Detection ---
#         # ... (existing detection code) ...
#         if len(content.shape) == 2: rgb_content = cv2.cvtColor(content, cv2.COLOR_GRAY2RGB)
#         elif content.shape[2] == 4: rgb_content = cv2.cvtColor(content, cv2.COLOR_BGRA2RGB)
#         elif content.shape[2] == 3: rgb_content = cv2.cvtColor(content, cv2.COLOR_BGR2RGB)
#         else: rgb_content = content # Fallback

#         image_tensor = tf.expand_dims(tf.convert_to_tensor(rgb_content, dtype=tf.uint8), axis=0)
#         detections = detector(image_tensor)
#         boxes = detections['detection_boxes'][0].numpy()
#         classes = detections['detection_classes'][0].numpy().astype(int)
#         scores = detections['detection_scores'][0].numpy()

#         # --- 2. Calculate Grid & Filter Detections ---
#         cell_height = content_height // GRID_ROWS
#         cell_width = content_width // GRID_COLS
#         if cell_height <= 0 or cell_width <= 0:
#             print("Error: Content dimensions too small for grid size.")
#             return content, current_overall_status, 0

#         density_grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
#         person_bboxes_this_content = []
#         confirmed_person_count_this_content = 0

#         for i in range(boxes.shape[0]):
#             if classes[i] == PERSON_CLASS_INDEX and scores[i] >= DETECTION_THRESHOLD:
#                 confirmed_person_count_this_content += 1
#                 ymin, xmin, ymax, xmax = boxes[i]
#                 x1, y1 = int(xmin * content_width), int(ymin * content_height)
#                 x2, y2 = int(xmax * content_width), int(ymax * content_height)
#                 person_bboxes_this_content.append((x1, y1, x2, y2))

#                 center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
#                 row = min(center_y // cell_height, GRID_ROWS - 1)
#                 col = min(center_x // cell_width, GRID_COLS - 1)
#                 if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
#                     density_grid[row][col] += 1

#         # --- 3. Analyze Density Grid ---
#         # Ensure analyze_density_grid returns all the needed values
#         content_status, risky_coords, total_grid_people, high_cells, crit_cells = analyze_density_grid(density_grid)

#         # --- 4. Send Data to Fluvio --- # <-- ADDED/MODIFIED
#         fluvio_payload = {
#             "timestamp": int(time.time()), # Current timestamp
#             "frame": frame_or_image_index, # Frame number or image index (e.g., 0 for single image)
#             "density_grid": density_grid, # The calculated grid
#             "frame_status": content_status, # Status determined for this frame
#             "confirmed_persons": confirmed_person_count_this_content, # Persons detected above threshold
#             "high_density_cells": high_cells, # Count of cells >= HIGH_DENSITY_THRESHOLD
#             "critical_density_cells": crit_cells # Count of cells >= CRITICAL_DENSITY_THRESHOLD
#         }
#         # Use frame index or a unique ID as the key
#         send_to_fluvio(f"content-{frame_or_image_index}", fluvio_payload)

#         # --- 5. Draw Overlays and Text ---
#         processed_content = content.copy() # Work on a copy
#         overlay_alpha = 0.4
#         overlay_color_critical = (0, 0, 255) # Red (BGR)
#         overlay_color_high = (0, 165, 255) # Orange (BGR)

#         # Draw overlays for individual risky grid cells (using the refined logic)
#         for r, c in risky_coords:
#             cell_y_start = r * cell_height
#             cell_y_end = (r + 1) * cell_height
#             cell_x_start = c * cell_width
#             cell_x_end = (c + 1) * cell_width

#             risk_level_in_cell = "unknown"
#             try: # Add try-except for safety accessing grid element
#                 if density_grid[r][c] >= CRITICAL_DENSITY_THRESHOLD: risk_level_in_cell = "critical"
#                 elif density_grid[r][c] >= HIGH_DENSITY_THRESHOLD: risk_level_in_cell = "high"
#             except IndexError:
#                  continue # Skip if coords are somehow out of bounds

#             color = overlay_color_critical if risk_level_in_cell == "critical" else overlay_color_high
#             overlay = processed_content.copy()
#             cv2.rectangle(overlay, (cell_x_start, cell_y_start), (cell_x_end, cell_y_end), color, -1)
#             cv2.addWeighted(overlay, overlay_alpha, processed_content, 1 - overlay_alpha, 0, processed_content)

#         # Draw Status Text & Stampede Chance Text
#         # Use effective_status which considers the overall video status for consistency
#         effective_status = get_higher_priority_status(current_overall_status, content_status)
#         status_text = f"Risk: {effective_status}"
#         status_color = (0, 128, 0) # Green default
#         if "CRITICAL" in effective_status: status_color = (0, 0, 255) # Red
#         elif "Warning" in effective_status or "High" in effective_status or "Detected" in effective_status: status_color = (0, 165, 255) # Orange

#         # Stampede Chance Text determination
#         if "CRITICAL" in effective_status: chance_text, chance_color = "Stampede Chance: Critical", (0, 0, 255)
#         elif "Warning" in effective_status or "High" in effective_status or "Detected" in effective_status: chance_text, chance_color = "Stampede Chance: High", (0, 165, 255)
#         else: chance_text, chance_color = "Stampede Chance: Low", (0, 128, 0)

#         # Add text with background rectangles
#         text_size_s, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
#         cv2.rectangle(processed_content, (5, 5), (15 + text_size_s[0], 30 + text_size_s[1]), (50, 50, 50), -1)
#         cv2.putText(processed_content, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

#         text_size_c, _ = cv2.getTextSize(chance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
#         cv2.rectangle(processed_content, (5, content_height - 35), (10 + text_size_c[0], content_height - 10), (50, 50, 50), -1)
#         cv2.putText(processed_content, chance_text, (10, content_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, chance_color, 2, cv2.LINE_AA)

#     except Exception as e:
#         print(f"!!! ERROR during process_media_content for index {frame_or_image_index}: {e}")
#         # Return original content on major processing error, maintain status, count 0
#         return content, current_overall_status, 0

#     # end_process_time = time.time()
#     # print(f" -> Index {frame_or_image_index} processed in {end_process_time - start_process_time:.3f}s. Status: {content_status}")

#     return processed_content, content_status, confirmed_person_count_this_content


# # --- Flask Routes ---
# # (index_route remains the same)
# @app.route('/', methods=['GET'])
# def index_route():
#     """Serves the main page."""
#     return render_template('index.html')

# # (upload_media_route: Minor change to print Fluvio status)
# @app.route('/upload_media', methods=['POST'])
# def upload_media_route():
#     print("\n--- Request received for /upload_media ---")
#     if fluvio_producer:
#          print("   Fluvio Status: Producer is active.")
#     else:
#          print("   Fluvio Status: Producer is INACTIVE.")
#     sys.stdout.flush()

#     start_time = time.time()
#     # ... (rest of file handling, type detection remains the same) ...
#     if 'media' not in request.files: return 'No media file part in the request', 400
#     media_file = request.files['media']
#     if media_file.filename == '': return 'No selected media file', 400
#     original_filename = secure_filename(media_file.filename)
#     upload_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
#     try: media_file.save(upload_path)
#     except Exception as e: return f"Error saving media file: {e}", 500
#     print(f"Media saved temporarily to: {upload_path}")

#     mimetype = mimetypes.guess_type(upload_path)[0]
#     file_type = 'unknown'
#     if mimetype:
#         if mimetype.startswith('video/'): file_type = 'video'
#         elif mimetype.startswith('image/'): file_type = 'image'
#     print(f"Detected file type: {file_type}")

#     # --- Initialize vars ---
#     output_url = None
#     overall_status = "Processing Started" # Initial status before processing starts
#     max_persons = 0
#     processing_time_secs = 0
#     output_media_type = None

#     # --- Process Video ---
#     if file_type == 'video':
#         output_media_type = 'video'
#         output_filename = f"processed_{os.path.splitext(original_filename)[0]}.mp4"
#         output_path = os.path.join(PROCESSED_VIDEO_FOLDER, output_filename)
#         cap = cv2.VideoCapture(upload_path)

#         if not cap.isOpened():
#             print(f"!!! ERROR: Failed to open input video file: {upload_path}")
#             overall_status = "Error: Could not open input video"
#             # Fall through to cleanup and render error
#         else:
#             try: # Wrap processing
#                 fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps and fps > 0 else 25.0
#                 width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                 height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                 print(f"Video Input: {width}x{height} @ {fps:.2f} FPS")

#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v'); # Or 'avc1' if mp4v fails
#                 out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#                 print(f"Attempting to open VideoWriter for: {output_path}")

#                 if not out_video.isOpened():
#                     print(f"!!! ERROR: Failed to initialize VideoWriter for {output_path}. Check codecs.")
#                     overall_status = "Error: Failed to initialize VideoWriter"
#                 else:
#                     print("VideoWriter opened successfully. Starting frame processing loop...")
#                     frame_num = 0
#                     video_overall_status = "Normal" # Track status specifically for this video
#                     while True:
#                         ret, frame = cap.read()
#                         if not ret: break # End of video

#                         # Process frame (pass video_overall_status for context)
#                         processed_frame, frame_status, people_count = process_media_content(
#                             frame, width, height, frame_num, video_overall_status
#                         )

#                         if processed_frame is None: continue # Skip if processing failed

#                         # Update overall status for the video and max persons
#                         video_overall_status = get_higher_priority_status(video_overall_status, frame_status)
#                         max_persons = max(max_persons, people_count)

#                         # Write frame
#                         try: out_video.write(processed_frame)
#                         except Exception as write_e:
#                             print(f"!!! ERROR writing frame {frame_num}: {write_e}")
#                             video_overall_status = f"Error: Video writing failed frame {frame_num}"
#                             break # Stop processing

#                         frame_num += 1
#                         # if frame_num % 50 == 0: print(f"  Processed frame {frame_num}...") # Optional progress

#                     print("Frame processing loop finished.")
#                     overall_status = video_overall_status # Assign final video status
#                     out_video.release()
#                     print("VideoWriter released.")

#                     # Check output file
#                     if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
#                         output_url = url_for('static', filename=f'processed_videos/{output_filename}')
#                         print(f"Output video created: {output_path}")
#                         if overall_status == "Processing Started": overall_status = "Completed" # Update if no issues arose
#                     else:
#                         print(f"!!! ERROR: Output video file missing or empty: {output_path}")
#                         overall_status = "Error: Output video generation failed"
#                         output_url = None

#             except Exception as video_proc_e:
#                 print(f"!!! ERROR during video processing: {video_proc_e}")
#                 overall_status = "Error: Unexpected failure during video processing"
#                 output_url = None
#             finally:
#                 if cap.isOpened(): cap.release(); print("VideoCapture released.")

#     # --- Process Image ---
#     elif file_type == 'image':
#         output_media_type = 'image'
#         output_filename = f"processed_{original_filename}"
#         # Ensure filename ends with a common image extension like .jpg
#         if not output_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#              output_filename += ".jpg"
#         output_path = os.path.join(PROCESSED_IMAGE_FOLDER, output_filename)

#         try:
#             image = cv2.imread(upload_path)
#             if image is None:
#                  raise ValueError("Could not read image file with OpenCV.")
#             height, width, _ = image.shape
#             print(f"Image Input: {width}x{height}")

#             # Process the single image (frame_index=0, status starts 'Normal')
#             processed_image, image_status, people_count = process_media_content(
#                 image, width, height, 0, "Normal"
#             )

#             if processed_image is None:
#                 raise ValueError("Image processing function returned None.")

#             overall_status = image_status # Status is just the result of this one image
#             max_persons = people_count

#             # Save the processed image
#             save_success = cv2.imwrite(output_path, processed_image)
#             if not save_success:
#                  raise ValueError("Failed to save processed image.")

#             print(f"Processed image saved to: {output_path}")
#             output_url = url_for('static', filename=f'processed_images/{output_filename}')

#         except Exception as img_proc_e:
#             print(f"!!! ERROR during image processing: {img_proc_e}")
#             overall_status = "Error: Image processing failed"
#             output_url = None

#     # --- Handle Unknown File Type ---
#     else:
#         print(f"Unsupported file type: {mimetype}")
#         overall_status = "Error: Unsupported file type"

#     # --- Cleanup Upload ---
#     try:
#         os.remove(upload_path)
#         print(f"Removed temporary upload: {upload_path}")
#     except OSError as e:
#         print(f"Warning: Could not remove temporary upload {upload_path}: {e}")

#     processing_time_secs = time.time() - start_time
#     print(f"Total request processing time: {processing_time_secs:.2f} seconds")

#     # --- Render Results ---
#     print(f"---> Rendering results page:")
#     print(f"     Media Type: {output_media_type}")
#     print(f"     Output URL: {output_url}")
#     print(f"     Status    : {overall_status}")
#     print(f"     Max Persons: {max_persons}")
#     return render_template('results.html',
#                            output_media_type=output_media_type,
#                            output_url=output_url,
#                            prediction_status=overall_status,
#                            max_persons=max_persons,
#                            processing_time=f"{processing_time_secs:.2f}")


# # --- Live Stream Route ---
# # (generate_live_frames needs slight modification to print Fluvio status)
# def generate_live_frames():
#     print("\n--- Request received for /video_feed (Live Stream) ---")
#     if fluvio_producer:
#          print("   Fluvio Status: Producer is active.")
#     else:
#          print("   Fluvio Status: Producer is INACTIVE.")
#     sys.stdout.flush()

#     live_cap = cv2.VideoCapture(0) # Or path to video file
#     # ... (rest of generate_live_frames remains largely the same, ensuring it calls process_media_content) ...
#     if not live_cap.isOpened():
#         print("!!! ERROR: Cannot open video source (webcam/file).")
#         # Handle error frame yield...
#         return

#     frame_width = int(live_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(live_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     print(f"Live source opened: {frame_width}x{frame_height}")

#     frame_num = 0
#     overall_stream_status = "Normal"

#     while True:
#         ret, frame = live_cap.read()
#         if not ret: break # End of stream

#         processed_frame, frame_status, _ = process_media_content(
#             frame, frame_width, frame_height, frame_num, overall_stream_status
#         )
#         if processed_frame is None: continue # Skip frame if processing fails

#         overall_stream_status = get_higher_priority_status(overall_stream_status, frame_status)

#         try:
#             ret_enc, buffer = cv2.imencode('.jpg', processed_frame)
#             if not ret_enc: continue # Skip if encoding fails
#             frame_bytes = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#         except Exception as e:
#             print(f"Error encoding/yielding live frame: {e}")
#             break # Stop stream on error

#         frame_num += 1

#     live_cap.release()
#     print("Live stream stopped.")


# @app.route('/live')
# def live_route():
#     return render_template('live.html')

# @app.route('/video_feed')
# def video_feed_route():
#     return Response(generate_live_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# # --- Run App ---
# if __name__ == '__main__':
#     print("--- Initializing Application ---")
#     # Attempt to connect to Fluvio when the application starts
#     fluvio_connected = connect_fluvio()
#     if not fluvio_connected:
#          print("!!! WARNING: Fluvio connection failed on startup. Data will not be sent.")
#     else:
#          print("+++ Fluvio connection successful on startup.")

#     print("--- Starting Flask Server ---")
#     # use_reloader=False is often important with background threads/connections
#     # threaded=True is needed for handling concurrent requests like the live stream
#     app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)




















# # Required Imports
# from flask import Flask, render_template, request, url_for, Response, stream_with_context, redirect
# import os
# import cv2 # OpenCV
# import tensorflow as tf
# import tensorflow_hub as hub
# import numpy as np
# import sys
# import time
# # from fluvio import Fluvio # Commenting out Fluvio if not essential for basic functionality test
# import json
# from werkzeug.utils import secure_filename # For safer filenames
# import mimetypes # To check file type

# # --- Flask Application Setup ---
# app = Flask(__name__)

# # --- Configuration for Folders ---
# UPLOAD_FOLDER = 'uploads'
# STATIC_FOLDER = 'static'
# PROCESSED_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'processed_frames')
# PROCESSED_VIDEO_FOLDER = os.path.join(STATIC_FOLDER, 'processed_videos')
# PROCESSED_IMAGE_FOLDER = os.path.join(STATIC_FOLDER, 'processed_images') # New folder for images
# DEBUG_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'debug_frames')

# # Create folders if they don't exist
# # Added PROCESSED_IMAGE_FOLDER
# for folder in [UPLOAD_FOLDER, PROCESSED_FRAMES_FOLDER, PROCESSED_VIDEO_FOLDER, PROCESSED_IMAGE_FOLDER, DEBUG_FRAMES_FOLDER]:
#     if not os.path.exists(folder):
#         os.makedirs(folder)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # --- Load Machine Learning Model ---
# # (Keep existing model loading code)
# DETECTOR_HANDLE = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
# detector = None
# try:
#     print(f"Loading detection model from: {DETECTOR_HANDLE}...")
#     # os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#     detector = hub.load(DETECTOR_HANDLE)
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"FATAL ERROR: Could not load the detection model: {e}")
#     sys.exit("Model loading failed. Exiting.")

# # --- Model Specific Settings ---
# PERSON_CLASS_INDEX = 1
# DETECTION_THRESHOLD = 0.25
# DEBUG_LOGGING_THRESHOLD = 0.1

# # --- Fluvio Settings (Optional - Keep commented out for now if debugging playback) ---
# # FLUVIO_CROWD_TOPIC = "crowd-data"
# # fluvio_client = None
# # fluvio_producer = None
# # (Keep fluvio functions if needed, but ensure they handle unavailability gracefully)
# def connect_fluvio(): pass # Placeholder
# def send_to_fluvio(key, data_dict): pass # Placeholder

# # --- Density Analysis Settings ---
# # (Keep existing settings: HIGH_DENSITY_THRESHOLD, etc.)
# HIGH_DENSITY_THRESHOLD = 6
# CRITICAL_DENSITY_THRESHOLD = 9
# HIGH_DENSITY_CELL_COUNT_THRESHOLD = 3
# CRITICAL_DENSITY_CELL_COUNT_THRESHOLD = 2
# GRID_ROWS = 8
# GRID_COLS = 8
# STATUS_HIERARCHY = {
#     "Normal": 0, "High Density Cell Detected": 1, "High Density Warning": 2,
#     "Critical Density Cell Detected": 3, "CRITICAL RISK": 4
# }

# # --- Helper Functions ---
# # (Keep analyze_density_grid, get_higher_priority_status)
# def analyze_density_grid(density_grid):
#     # ... (existing code) ...
#     high_density_cells = 0
#     critical_density_cells = 0
#     risky_cell_coords = [] # List of (row, col) for risky cells
#     overall_status = "Normal"
#     total_people_in_grid = 0

#     if not density_grid or len(density_grid) != GRID_ROWS:
#         print("   Warning: Invalid density grid received.")
#         return overall_status, risky_cell_coords, total_people_in_grid, high_density_cells, critical_density_cells

#     for r_idx, row in enumerate(density_grid):
#         if len(row) != GRID_COLS: continue
#         for c_idx, count in enumerate(row):
#             try:
#                 person_count = int(count)
#                 total_people_in_grid += person_count
#                 if person_count >= CRITICAL_DENSITY_THRESHOLD:
#                     critical_density_cells += 1
#                     risky_cell_coords.append((r_idx, c_idx))
#                 elif person_count >= HIGH_DENSITY_THRESHOLD:
#                     high_density_cells += 1
#                     risky_cell_coords.append((r_idx, c_idx))
#             except (ValueError, TypeError):
#                 continue

#     if critical_density_cells >= CRITICAL_DENSITY_CELL_COUNT_THRESHOLD:
#         overall_status = "CRITICAL RISK"
#     elif critical_density_cells > 0:
#         overall_status = "Critical Density Cell Detected"
#     elif high_density_cells >= HIGH_DENSITY_CELL_COUNT_THRESHOLD:
#         overall_status = "High Density Warning"
#     elif high_density_cells > 0:
#         overall_status = "High Density Cell Detected"

#     return overall_status, risky_cell_coords, total_people_in_grid, high_density_cells, critical_density_cells

# def get_higher_priority_status(status1, status2):
#     p1 = STATUS_HIERARCHY.get(status1, -1)
#     p2 = STATUS_HIERARCHY.get(status2, -1)
#     return status1 if p1 >= p2 else status2

# # --- Frame/Image Processing Function ---
# # Modified to handle both single images and video frames
# # Added 'is_video_frame' flag (though not strictly needed if logic is identical)
# def process_media_content(content, content_width, content_height, frame_or_image_index, current_overall_status="Normal"):
#     """
#     Processes a single image or video frame: detects people, calculates density,
#     determines status, draws overlays and text.
#     Returns the processed content (image/frame), content status, and total people count.
#     """
#     if content is None:
#         print(f"Warning: Received None content for processing at index {frame_or_image_index}")
#         return None, current_overall_status, 0

#     start_process_time = time.time()

#     # --- 1. ML Preprocessing & Detection ---
#     try:
#         # Ensure content is in RGB format for TensorFlow Hub model
#         if len(content.shape) == 2: # Grayscale? Convert
#             rgb_content = cv2.cvtColor(content, cv2.COLOR_GRAY2RGB)
#         elif content.shape[2] == 4: # BGRA? Convert
#              rgb_content = cv2.cvtColor(content, cv2.COLOR_BGRA2RGB)
#         elif content.shape[2] == 3: # Assume BGR, convert
#              rgb_content = cv2.cvtColor(content, cv2.COLOR_BGR2RGB)
#         else:
#              print(f"Warning: Unexpected number of channels ({content.shape[2]}) for content {frame_or_image_index}. Trying to proceed.")
#              rgb_content = content # Hope for the best? Or return error?

#         image_tensor = tf.expand_dims(tf.convert_to_tensor(rgb_content, dtype=tf.uint8), axis=0)
#         detections = detector(image_tensor) # Run detection
#         boxes = detections['detection_boxes'][0].numpy()
#         classes = detections['detection_classes'][0].numpy().astype(int)
#         scores = detections['detection_scores'][0].numpy()
#     except Exception as e:
#         print(f"Index {frame_or_image_index}: Error in ML processing: {e}")
#         return content, current_overall_status, 0 # Return original on error

#     # --- 2. Calculate Grid & Filter Detections ---
#     cell_height = content_height // GRID_ROWS
#     cell_width = content_width // GRID_COLS
#     if cell_height <= 0 or cell_width <= 0:
#         print("Error: Content dimensions too small for grid size.")
#         return content, current_overall_status, 0

#     density_grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
#     person_bboxes_this_content = [] # BBoxes for drawing later
#     confirmed_person_count_this_content = 0
#     # debug_content = content.copy() # If debug saving needed

#     # --- Filter for Persons and Populate Density Grid ---
#     for i in range(boxes.shape[0]):
#         if classes[i] == PERSON_CLASS_INDEX and scores[i] >= DETECTION_THRESHOLD:
#             confirmed_person_count_this_content += 1
#             ymin, xmin, ymax, xmax = boxes[i]
#             x1, y1 = int(xmin * content_width), int(ymin * content_height)
#             x2, y2 = int(xmax * content_width), int(ymax * content_height)
#             person_bboxes_this_content.append((x1, y1, x2, y2))

#             # Assign to grid cell based on center
#             center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
#             row = min(center_y // cell_height, GRID_ROWS - 1)
#             col = min(center_x // cell_width, GRID_COLS - 1)
#             if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
#                 density_grid[row][col] += 1

#     # --- 3. Analyze Density Grid ---
#     content_status, risky_coords, total_grid_people, high_cells, crit_cells = analyze_density_grid(density_grid)

#     # --- 4. Send Data to Fluvio (Optional) ---
#     # fluvio_payload = { ... }
#     # send_to_fluvio(f"content-{frame_or_image_index}", fluvio_payload)

#     # --- 5. Draw Overlays and Text ---
#     final_content = content.copy() # Work on a copy to avoid modifying original input
#     overlay_alpha = 0.3

#     # Draw Crowd Area Overlay if needed
#     if "CRITICAL" in content_status or "High" in content_status or "Detected" in content_status:
#         if person_bboxes_this_content:
#             min_crowd_x = min(box[0] for box in person_bboxes_this_content)
#             min_crowd_y = min(box[1] for box in person_bboxes_this_content)
#             max_crowd_x = max(box[2] for box in person_bboxes_this_content)
#             max_crowd_y = max(box[3] for box in person_bboxes_this_content)

#             overlay = final_content.copy()
#             overlay_color = (0, 0, 255) if "CRITICAL" in content_status else (0, 165, 255) # Red or Orange
#             cv2.rectangle(overlay, (min_crowd_x, min_crowd_y), (max_crowd_x, max_crowd_y), overlay_color, -1)
#             cv2.addWeighted(overlay, overlay_alpha, final_content, 1 - overlay_alpha, 0, final_content)

#     # Draw Status Text
#     # For video, use effective_status; for image, use content_status directly
#     effective_status = get_higher_priority_status(current_overall_status, content_status)
#     status_text = f"Risk: {effective_status}"
#     status_color = (0, 128, 0) # Green default
#     if "CRITICAL" in effective_status: status_color = (0, 0, 255) # Red
#     elif "Warning" in effective_status or "High" in effective_status or "Detected" in effective_status: status_color = (0, 165, 255) # Orange

#     # Add background rectangle for text
#     text_size, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
#     text_w, text_h = text_size
#     cv2.rectangle(final_content, (5, 5), (15 + text_w, 30 + text_h), (50, 50, 50), -1)
#     cv2.putText(final_content, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

#     # Draw Stampede Chance Text
#     if "CRITICAL" in effective_status:
#         chance_text = f"Stampede Chance: Critical"
#         chance_color = (0, 0, 255)
#     elif "Warning" in effective_status or "High" in effective_status or "Detected" in effective_status:
#         chance_text = f"Stampede Chance: High"
#         chance_color = (0, 165, 255)
#     else:
#         chance_text = "Stampede Chance: Low"
#         chance_color = (0, 128, 0)

#     text_size_b, _ = cv2.getTextSize(chance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
#     text_w_b, text_h_b = text_size_b
#     cv2.rectangle(final_content, (5, content_height - 35), (10 + text_w_b, content_height - 10), (50, 50, 50), -1)
#     cv2.putText(final_content, chance_text, (10, content_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, chance_color, 2, cv2.LINE_AA)

#     end_process_time = time.time()
#     # print(f" -> Index {frame_or_image_index} processed in {end_process_time - start_process_time:.3f}s. Status: {content_status}")

#     # Return processed image/frame, its status, and person count found *in it*
#     return final_content, content_status, confirmed_person_count_this_content

# # --- Flask Routes ---

# @app.route('/', methods=['GET'])
# def index_route():
#     """Serves the main page."""
#     return render_template('index.html')

# # Renamed route and function, handles 'media' input
# @app.route('/upload_media', methods=['POST'])
# def upload_media_route():
#     print("\n--- Request received for /upload_media ---")
#     start_time = time.time()

#     if 'media' not in request.files:
#         return 'No media file part in the request', 400
#     media_file = request.files['media']
#     if media_file.filename == '':
#         return 'No selected media file', 400

#     # Use secure_filename for safety
#     original_filename = secure_filename(media_file.filename)
#     upload_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)

#     try:
#         media_file.save(upload_path)
#         print(f"Media saved temporarily to: {upload_path}")
#     except Exception as e:
#         return f"Error saving media file: {e}", 500

#     # --- Determine File Type ---
#     mimetype = mimetypes.guess_type(upload_path)[0]
#     file_type = 'unknown'
#     if mimetype:
#         if mimetype.startswith('video/'):
#             file_type = 'video'
#         elif mimetype.startswith('image/'):
#             file_type = 'image'

#     print(f"Detected file type: {file_type} (MIME: {mimetype})")

#     # --- Process Based on Type ---
#     output_url = None
#     overall_status = "Normal"
#     max_persons = 0
#     processing_time_secs = 0
#     output_media_type = None # To tell the template what to render

#     if file_type == 'video':
# # Inside the 'if file_type == 'video':' block in upload_media_route

#         output_media_type = 'video' # Set this early
#         output_url = None # Initialize to None
#         overall_status = "Processing Started" # Initial status
#         max_persons = 0
#         output_filename = f"processed_{os.path.splitext(original_filename)[0]}.mp4"
#         output_path = os.path.join(PROCESSED_VIDEO_FOLDER, output_filename)

#         cap = cv2.VideoCapture(upload_path)
#         if not cap.isOpened():
#             print(f"!!! CRITICAL ERROR: Failed to open input video file: {upload_path}")
#             try: os.remove(upload_path)
#             except OSError: pass
#             # Ensure status reflects failure before returning
#             processing_time_secs = time.time() - start_time
#             return render_template('results.html', prediction_status="Error: Could not open input video", processing_time=f"{processing_time_secs:.2f}")

#         try: # Wrap video processing in a try block for better error catching
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             if not fps or fps <= 0: fps = 25.0
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             print(f"Video Input: {width}x{height} @ {fps:.2f} FPS")

#             # Video Writer Setup
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or 'avc1'
#             out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#             print(f"Attempting to open VideoWriter for: {output_path}")
#             print(f" -> VideoWriter Opened Successfully: {out_video.isOpened()}") # *** CHECK THIS LOG ***

#             if not out_video.isOpened():
#                 # Specific error if writer failed
#                 overall_status = "Error: Failed to initialize VideoWriter (Check Codecs?)"
#                 print(f"!!! {overall_status} !!!")
#                 # No output_url can be generated
#             else:
#                 print(f"VideoWriter ready. Starting frame processing loop...")
#                 frame_num = 0
#                 loop_start_time = time.time()
#                 while True:
#                     ret, frame = cap.read()
#                     if not ret:
#                         print(f"End of video stream reached after frame {frame_num-1}.")
#                         break # Exit loop cleanly

#                     # Use the unified processing function
#                     processed_frame, frame_status, people_count = process_media_content(
#                         frame, width, height, frame_num, overall_status
#                     )

#                     if processed_frame is None:
#                         print(f"Warning: process_media_content returned None for frame {frame_num}. Skipping write.")
#                         frame_num += 1
#                         continue # Skip writing this frame

#                     # Update overall status and max people count
#                     overall_status = get_higher_priority_status(overall_status, frame_status)
#                     max_persons = max(max_persons, people_count)

#                     # Write the processed frame to the output video
#                     try:
#                         out_video.write(processed_frame)
#                     except Exception as write_e:
#                         print(f"!!! CRITICAL ERROR writing frame {frame_num}: {write_e}")
#                         overall_status = f"Error: Failed during video writing frame {frame_num}"
#                         break # Stop processing if writing fails

#                     frame_num += 1
#                     if frame_num % 50 == 0: print(f"  Processed frame {frame_num}...")

#                 loop_end_time = time.time()
#                 print(f"Frame processing loop finished. Time taken: {loop_end_time - loop_start_time:.2f}s")

#                 # Release VideoWriter *only if it was opened*
#                 print("Releasing VideoWriter...")
#                 out_video.release() # Release should happen regardless of success/failure after opening
#                 print("VideoWriter released.")

#                 # *** Crucial Check: Does the file exist NOW? ***
#                 print(f"Checking for output file existence: {output_path}")
#                 if os.path.exists(output_path):
#                     file_size = os.path.getsize(output_path)
#                     print(f" -> Output file FOUND. Size: {file_size} bytes.")
#                     if file_size > 0: # Check if file is not empty
#                          output_url = url_for('static', filename=f'processed_videos/{output_filename}')
#                          print(f" -> Generated output_url: {output_url}")
#                          if overall_status.startswith("Error:") or overall_status == "Processing Started":
#                               overall_status = "Completed with potential issues" # Or use highest status found
#                     else:
#                          print("!!! ERROR: Output file exists but is EMPTY (0 bytes). Setting URL to None.")
#                          overall_status = "Error: Output video file is empty"
#                          output_url = None
#                 else:
#                     print(f"!!! CRITICAL ERROR: Output file {output_path} DOES NOT EXIST after processing.")
#                     overall_status = "Error: Output video file missing"
#                     output_url = None

#         except Exception as video_proc_e:
#             print(f"!!! An unexpected error occurred during video processing: {video_proc_e}")
#             overall_status = "Error: Unexpected failure during video processing"
#             output_url = None # Ensure URL is None on major error
#         finally:
#              # Ensure capture device is always released
#              if cap.isOpened():
#                   print("Releasing VideoCapture...")
#                   cap.release()
#                   print("VideoCapture released.")

#         # --- Upload Cleanup (remains the same) ---
#         # ...

#         processing_time_secs = time.time() - start_time
#         print(f"Total processing time: {processing_time_secs:.2f} seconds")

#         # --- Render Results ---
#         # Final log before rendering
#         print(f"---> Rendering results page:")
#         print(f"     Media Type: {output_media_type}")
#         print(f"     Output URL: {output_url}")
#         print(f"     Status    : {overall_status}")
#         print(f"     Max Persons: {max_persons}")
#         return render_template('results.html',
#                                output_media_type=output_media_type,
#                                output_url=output_url,
#                                prediction_status=overall_status,
#                                max_persons=max_persons,
#                                processing_time=f"{processing_time_secs:.2f}")


# # --- Live Stream Route (Keep existing generate_live_frames and routes) ---
# def generate_live_frames():
#     # ... (existing code, ensure it uses process_media_content) ...
#     print("Starting live stream...")
#     # Use 0 for default webcam, or path to a video file for testing
#     # live_cap = cv2.VideoCapture(0)
#     live_cap = cv2.VideoCapture(0) # Example: using a file

#     if not live_cap.isOpened():
#         print("!!! ERROR: Cannot open video source (webcam/file).")
#         # Yield an error message frame
#         error_img = np.zeros((480, 640, 3), dtype=np.uint8)
#         cv2.putText(error_img, "ERROR: Cannot open video source", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         ret, buffer = cv2.imencode('.jpg', error_img)
#         if ret:
#             frame_bytes = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#         return

#     frame_width = int(live_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(live_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     print(f"Live source opened: {frame_width}x{frame_height}")

#     frame_num = 0
#     overall_stream_status = "Normal"

#     while True:
#         ret, frame = live_cap.read()
#         if not ret:
#             print("End of live stream source or error grabbing frame.")
#             # Optional: yield a 'stream ended' frame?
#             break # Exit loop if stream ends or fails

#         # Use the unified processing function
#         processed_frame, frame_status, _ = process_media_content(
#             frame, frame_width, frame_height, frame_num, overall_stream_status
#         )

#         if processed_frame is None:
#              print("Warning: Frame processing returned None. Skipping frame in stream.")
#              # Send original frame instead?
#              # processed_frame = frame
#              continue # Skip this frame if processing failed

#         overall_stream_status = get_higher_priority_status(overall_stream_status, frame_status)

#         try:
#             ret_enc, buffer = cv2.imencode('.jpg', processed_frame)
#             if not ret_enc:
#                 print("Error encoding frame to JPEG")
#                 continue
#             frame_bytes = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#         except Exception as e:
#             print(f"Error encoding or yielding frame: {e}")
#             break

#         frame_num += 1
#         # time.sleep(0.01) # Optional delay

#     live_cap.release()
#     print("Live stream stopped.")


# @app.route('/live')
# def live_route():
#     return render_template('live.html')

# @app.route('/video_feed')
# def video_feed_route():
#     return Response(generate_live_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


# # --- Run App ---
# if __name__ == '__main__':
#     print("--- Starting Flask Server ---")
#     connect_fluvio() # Optional
#     # Make sure reloader is False if TF gives issues
#     app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True) # Added threaded=True





















'''
# Required Imports
from flask import Flask, render_template, request, url_for, Response, stream_with_context
import os
import cv2 # OpenCV
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sys
import time
from fluvio import Fluvio
import json

# --- Flask Application Setup ---
app = Flask(__name__)

# --- Configuration for Folders ---
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
PROCESSED_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'processed_frames') # Still used for potential single-frame output or debugging
PROCESSED_VIDEO_FOLDER = os.path.join(STATIC_FOLDER, 'processed_videos') # For video output
DEBUG_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'debug_frames') # For debugging detections

# Create folders if they don't exist
for folder in [UPLOAD_FOLDER, PROCESSED_FRAMES_FOLDER, PROCESSED_VIDEO_FOLDER, DEBUG_FRAMES_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load Machine Learning Model ---
DETECTOR_HANDLE = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
detector = None
try:
    print(f"Loading detection model from: {DETECTOR_HANDLE}...")
    # Set TF_ENABLE_ONEDNN_OPTS=0 if you encounter issues like "Illegal instruction" on some CPUs
    # os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Uncomment if needed
    detector = hub.load(DETECTOR_HANDLE)
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load the detection model: {e}")
    print("If you see 'Illegal instruction', try uncommenting 'os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'' near the model loading section.")
    sys.exit("Model loading failed. Exiting.")

# --- Model Specific Settings ---
PERSON_CLASS_INDEX = 1
# *** TUNABLE: Adjust this based on detection results ***
DETECTION_THRESHOLD = 0.25 # Threshold for considering a detection a 'person'
DEBUG_LOGGING_THRESHOLD = 0.1 # Threshold for logging *any* detection in debug frames

# --- Fluvio Settings ---
FLUVIO_CROWD_TOPIC = "crowd-data"
fluvio_client = None
fluvio_producer = None

def connect_fluvio():
    global fluvio_client, fluvio_producer
    if fluvio_client and fluvio_producer:
        return True
    try:
        print("Connecting to Fluvio...")
        fluvio_client = Fluvio.connect()
        print("Getting Fluvio producer...")
        fluvio_producer = fluvio_client.topic_producer(FLUVIO_CROWD_TOPIC)
        print("Fluvio ready.")
        return True
    except Exception as e:
        print(f"!!! FLUVIO WARNING: Could not connect or get producer: {e}")
        fluvio_client = None
        fluvio_producer = None
        return False

def send_to_fluvio(key, data_dict):
    if not fluvio_producer:
        if not connect_fluvio():
             print("!!! FLUVIO Send ERROR: Producer not available.")
             return # Skip sending if connection failed

    try:
        fluvio_data = json.dumps(data_dict)
        fluvio_producer.send(key.encode('utf-8'), fluvio_data.encode('utf-8'))
        # print(f"Fluvio: Sent data for {key}") # Uncomment for verbose logging
    except Exception as e:
        print(f"!!! FLUVIO Send WARNING: {e}")
        # Attempt to reconnect on error might be needed in robust scenarios
        # global fluvio_client, fluvio_producer
        # fluvio_client = None
        # fluvio_producer = None


# --- Density Analysis Settings ---
# *** TUNABLE: Adjust these based on your video/camera view and desired sensitivity ***
HIGH_DENSITY_THRESHOLD = 5      # People per cell for 'high' (Orange Overlay)
CRITICAL_DENSITY_THRESHOLD = 8  # People per cell for 'critical' (Red Overlay)

# Thresholds for overall status based on *number* of dense cells
HIGH_DENSITY_CELL_COUNT_THRESHOLD = 3   # Number of 'high' cells for "High Density Warning" status
CRITICAL_DENSITY_CELL_COUNT_THRESHOLD = 2 # Number of 'critical' cells for "CRITICAL RISK" status

# Grid dimensions
GRID_ROWS = 8
GRID_COLS = 8

# Status Hierarchy
STATUS_HIERARCHY = {
    "Normal": 0, "High Density Cell Detected": 1, "High Density Warning": 2,
    "Critical Density Cell Detected": 3, "CRITICAL RISK": 4
}

# --- Helper Functions ---

def analyze_density_grid(density_grid):
    """Analyzes density grid, returns status, risky cell details, and counts."""
    high_density_cells = 0
    critical_density_cells = 0
    risky_cell_coords = [] # List of (row, col) for risky cells
    overall_status = "Normal"
    total_people_in_grid = 0

    if not density_grid or len(density_grid) != GRID_ROWS:
        print("   Warning: Invalid density grid received.")
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
                    # print(f"    Cell ({r_idx},{c_idx}): Count={person_count} -> CRITICAL") # Debug
                elif person_count >= HIGH_DENSITY_THRESHOLD:
                    high_density_cells += 1
                    risky_cell_coords.append((r_idx, c_idx))
                    # print(f"    Cell ({r_idx},{c_idx}): Count={person_count} -> HIGH") # Debug
            except (ValueError, TypeError):
                continue # Skip non-integer counts

    # Determine overall status based on cell counts
    if critical_density_cells >= CRITICAL_DENSITY_CELL_COUNT_THRESHOLD:
        overall_status = "CRITICAL RISK"
    elif critical_density_cells > 0:
         overall_status = "Critical Density Cell Detected" # Status for 1 critical cell
    elif high_density_cells >= HIGH_DENSITY_CELL_COUNT_THRESHOLD:
        overall_status = "High Density Warning"
    elif high_density_cells > 0:
        overall_status = "High Density Cell Detected" # Status for 1+ high cells but below warning threshold

    # print(f"   Analysis: Status='{overall_status}', Risky Cells={len(risky_cell_coords)}, Total People={total_people_in_grid}")
    return overall_status, risky_cell_coords, total_people_in_grid, high_density_cells, critical_density_cells

def get_higher_priority_status(status1, status2):
    """Returns the status with higher priority based on STATUS_HIERARCHY."""
    p1 = STATUS_HIERARCHY.get(status1, -1)
    p2 = STATUS_HIERARCHY.get(status2, -1)
    return status1 if p1 >= p2 else status2

def process_frame(frame, frame_width, frame_height, frame_count, current_overall_status):
    """
    Processes a single frame: detects people, calculates density, determines status,
    draws overlays and text, and sends data to Fluvio.
    Returns the processed frame, frame status, and total people count.
    """
    if frame is None:
        return None, current_overall_status, 0

    # --- 1. ML Preprocessing & Detection ---
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = tf.expand_dims(tf.convert_to_tensor(rgb_frame, dtype=tf.uint8), axis=0)
        detections = detector(image_tensor)
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(int)
        scores = detections['detection_scores'][0].numpy()
    except Exception as e:
        print(f"Frame {frame_count}: Error in ML processing: {e}")
        return frame, current_overall_status, 0 # Return original frame on error

    # --- 2. Calculate Grid & Filter Detections ---
    cell_height = frame_height // GRID_ROWS
    cell_width = frame_width // GRID_COLS
    if cell_height <= 0 or cell_width <= 0:
        print("Error: Frame dimensions too small for grid size.")
        return frame, current_overall_status, 0

    density_grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    person_bboxes_this_frame = [] # Store (x1, y1, x2, y2) of detected persons
    confirmed_person_count_this_frame = 0
    debug_frame = frame.copy() # For saving raw detections separately

    # --- Optional: Save Debug Frame with ALL detections ---
    save_debug = True # Set to False to disable saving debug frames
    if save_debug:
        total_detections_logged = 0
        class_colors = {PERSON_CLASS_INDEX: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255)} # Green, Red, Blue
        default_color = (255, 255, 0) # Yellow
        for i in range(boxes.shape[0]):
             if scores[i] >= DEBUG_LOGGING_THRESHOLD:
                total_detections_logged += 1
                ymin, xmin, ymax, xmax = boxes[i]
                x1, y1 = int(xmin * frame_width), int(ymin * frame_height)
                x2, y2 = int(xmax * frame_width), int(ymax * frame_height)
                class_id = classes[i]
                box_color = class_colors.get(class_id, default_color)
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), box_color, 1)
                label = f"ID:{class_id} S:{scores[i]:.2f}"
                cv2.putText(debug_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
        debug_output_filename = f"debug_frame_{frame_count:05d}.jpg"
        debug_output_filepath = os.path.join(DEBUG_FRAMES_FOLDER, debug_output_filename)
        try: cv2.imwrite(debug_output_filepath, debug_frame)
        except Exception as e: print(f"Error saving DEBUG frame {frame_count}: {e}")
    # --- End Debug Frame Section ---


    # --- Filter for Persons and Populate Density Grid ---
    for i in range(boxes.shape[0]):
        if classes[i] == PERSON_CLASS_INDEX and scores[i] >= DETECTION_THRESHOLD:
            confirmed_person_count_this_frame += 1
            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1 = int(xmin * frame_width), int(ymin * frame_height)
            x2, y2 = int(xmax * frame_width), int(ymax * frame_height)
            person_bboxes_this_frame.append((x1, y1, x2, y2))

            # Calculate center and assign to grid cell
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            row = min(center_y // cell_height, GRID_ROWS - 1)
            col = min(center_x // cell_width, GRID_COLS - 1)
            if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
                density_grid[row][col] += 1

    # print(f"Frame {frame_count}: Confirmed Persons={confirmed_person_count_this_frame}")
    # print(f"  Density Grid: {density_grid}")

    # --- 3. Analyze Density Grid for this Frame ---
    frame_status, risky_cell_coords, total_people_in_grid, high_cells, critical_cells = analyze_density_grid(density_grid)

    # --- 4. Send Data to Fluvio ---
    fluvio_payload = {
        "timestamp": int(time.time()),
        "frame": frame_count,
        "density_grid": density_grid,
        "confirmed_persons": confirmed_person_count_this_frame,
        "frame_status": frame_status, # Send frame status too
        "high_density_cells": high_cells,
        "critical_density_cells": critical_cells
    }
    send_to_fluvio(f"frame-{frame_count}", fluvio_payload)

    # --- 5. Draw Overlays and Text on Final Frame ---
    final_frame = frame # Start with the original frame
    overlay_alpha = 0.3 # Transparency of the overlay

    # Draw Crowd Area Overlay (Phase 3 modification)
    if "CRITICAL" in frame_status or "High" in frame_status or "Detected" in frame_status:
        if person_bboxes_this_frame:
            min_crowd_x = min(box[0] for box in person_bboxes_this_frame)
            min_crowd_y = min(box[1] for box in person_bboxes_this_frame)
            max_crowd_x = max(box[2] for box in person_bboxes_this_frame)
            max_crowd_y = max(box[3] for box in person_bboxes_this_frame)

            overlay = final_frame.copy()
            overlay_color = (0, 0, 255) if "CRITICAL" in frame_status else (0, 165, 255)
            cv2.rectangle(overlay, (min_crowd_x, min_crowd_y), (max_crowd_x, max_crowd_y), overlay_color, -1)
            cv2.addWeighted(overlay, overlay_alpha, final_frame, 1 - overlay_alpha, 0, final_frame)
            # print(f"    Drawing {frame_status} overlay over crowd area")

    # # Draw Green Bounding Boxes for confirmed persons
    # person_box_color = (0, 255, 0); thickness = 1 # Thinner boxes
    # for x1, y1, x2, y2 in person_bboxes_this_frame:
    #      cv2.rectangle(final_frame, (x1, y1), (x2, y2), person_box_color, thickness)

    # Draw Status Text (Phase 3 modification)
    # Use the higher priority status seen so far for consistent text in video output
    effective_status = get_higher_priority_status(current_overall_status, frame_status)
    text_to_display = f"Risk: {effective_status}"
    status_color = (0, 128, 0) # Default Green

    if "CRITICAL" in effective_status:
        status_color = (0, 0, 255) # Red
    elif "Warning" in effective_status or "High" in effective_status or "Detected" in effective_status: # Orange for High/Warning/Detected
        status_color = (0, 165, 255) # Orange

    # Add background rectangle for text
    text_size, _ = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_w, text_h = text_size
    cv2.rectangle(final_frame, (5, 5), (15 + text_w, 30 + text_h), (50, 50, 50), -1) # Dark grey background

    # Draw black outline (optional, can make text clearer)
    # cv2.putText(final_frame, text_to_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    # Draw colored text
    cv2.putText(final_frame, text_to_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

    if "CRITICAL" in effective_status:
        text_to_display = f"Stampede Chance: Critical" # Or with percentage if available
        status_color = (0, 0, 255) # Red
    elif "Warning" in effective_status or "High" in effective_status or "Detected" in effective_status:
        text_to_display = f"Stampede Chance: High" # Or Moderate, etc.
        status_color = (0, 165, 255) # Orange
    else:
        text_to_display = "Stampede Chance: Low" # Or Normal
        status_color = (0, 128, 0) # Green

    text_size, _ = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_w, text_h = text_size
    cv2.rectangle(final_frame, (5, frame_height - 35), (10 + text_w, frame_height - 10), (50, 50, 50), -1)

    cv2.putText(final_frame, text_to_display, (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

    return final_frame, frame_status, total_people_in_grid

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index_route():
    """Serves the main page with upload form and live link."""
    return render_template('index.html')

# --- Flask Routes ---

@app.route('/upload', methods=['POST'])
def upload_video_route():
    """Handles video upload, processing, and returns results page with video."""
    print("\n--- Request received for /upload ---")
    start_time = time.time()
    sys.stdout.flush()

    if 'video' not in request.files:
        return 'No video file part in the request', 400
    video_file = request.files['video']
    if video_file.filename == '':
        return 'No selected video file', 400

    print(f"Received video file: {video_file.filename}")

    # --- Video Saving and Opening ---
    safe_filename = os.path.basename(video_file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    try:
        video_file.save(upload_path)
        print(f"Video saved to: {upload_path}")
    except Exception as e:
        return f"Error saving video: {e}", 500

    cap = cv2.VideoCapture(upload_path)
    if not cap.isOpened():
        try: os.remove(upload_path)
        except OSError: pass
        return f"Error opening video file: {upload_path}", 500

    fps = cap.get(cv2.CAP_PROP_FPS)
    # Handle cases where FPS might be 0 or invalid
    if not fps or fps <= 0:
        print("Warning: Invalid FPS detected, defaulting to 25.0")
        fps = 25.0 # Or another sensible default
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video opened: {width}x{height} @ {fps:.2f} FPS, ~{frame_count_total} frames")

    # --- Video Writer Setup (MODIFIED) ---
    # Change extension to .mp4
    output_video_filename = f"processed_{os.path.splitext(safe_filename)[0]}.mp4"
    output_video_path = os.path.join(PROCESSED_VIDEO_FOLDER, output_video_filename)
    # Change FourCC to 'mp4v' for H.264 in MP4. Try 'avc1' or 'H264' if this fails.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out_video.isOpened():
        print(f"!!! ERROR: Could not open VideoWriter for {output_video_path}")
        print("!!! CHECK if necessary codecs (like H.264) are installed in your environment (e.g., ffmpeg, libx264-dev).")
        cap.release()
        try: os.remove(upload_path);
        except OSError: pass
        return "Error initializing video writer.", 500
    print(f"Output video will be saved to: {output_video_path} with FourCC 'mp4v'")
    # --- End Video Writer Setup Modification ---

    # --- Processing Loop (Keep as is) ---
    frame_num = 0
    overall_video_status = "Normal"
    max_people_in_frame = 0

    print("Starting frame processing loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video. Frames processed: {frame_num}")
            break

        processed_frame, frame_status, people_count = process_frame(frame, width, height, frame_num, overall_video_status)

        if processed_frame is None:
            frame_num += 1
            continue

        overall_video_status = get_higher_priority_status(overall_video_status, frame_status)
        if people_count > max_people_in_frame:
            max_people_in_frame = people_count

        # Write frame
        try:
            out_video.write(processed_frame)
        except Exception as e:
            print(f"!!! ERROR writing frame {frame_num}: {e}")
            # Decide if you want to break or continue on write error
            break # Safer to break if writing fails consistently

        frame_num += 1
        if frame_num % 50 == 0:
            print(f"  Processed frame {frame_num}...")
        sys.stdout.flush()
    # --- End Processing Loop ---


    # --- Cleanup (Keep as is, ensure release happens) ---
    cap.release()
    if out_video: out_video.release() # Make sure release is called!
    print("Video readers and writers released.")
    try:
        os.remove(upload_path)
        print(f"Removed temporary upload file: {upload_path}")
    except OSError as e:
        print(f"Warning: Could not remove temporary upload file: {e}")

    processing_time = time.time() - start_time
    print(f"Total processing time: {processing_time:.2f} seconds")

    # --- Render Results (URL generation will use the new .mp4 filename) ---
    print(f"Rendering results page. Overall Status: {overall_video_status}, Max Persons Detected: {max_people_in_frame}")
    video_url = url_for('static', filename=f'processed_videos/{output_video_filename}') # This automatically uses the new filename
    return render_template('results.html',
                           output_video_url=video_url,
                           prediction_status=overall_video_status,
                           max_persons=max_people_in_frame,
                           processing_time=f"{processing_time:.2f}")

# --- Live Stream Route (Phase 5) ---

def generate_live_frames():
    """Generator function for streaming live processed frames."""
    print("Starting live stream...")
    # Use 0 for default webcam
    # *** TRY CHANGING 0 TO 1 or 2 if webcam doesn't start ***
    live_cap = cv2.VideoCapture("videoplayback.mp4")
    if not live_cap.isOpened():
        print("!!! ERROR: Cannot open webcam. Check index (0, 1, etc.) and permissions.")
        # Yield an error message frame (optional)
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "ERROR: Cannot open webcam", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_img)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return # Stop the generator

    frame_width = int(live_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(live_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam opened: {frame_width}x{frame_height}")

    frame_num = 0
    overall_stream_status = "Normal" # Track status over the stream duration

    while True:
        ret, frame = live_cap.read()
        if not ret:
            print("Warning: Failed to grab frame from webcam.")
            # Send a frame indicating the issue? Or just wait?
            time.sleep(0.1) # Wait briefly before trying again
            continue

        # Process the frame
        processed_frame, frame_status, _ = process_frame(frame, frame_width, frame_height, frame_num, overall_stream_status)

        if processed_frame is None:
            print("Warning: Frame processing returned None. Using original frame for stream.")
            processed_frame = frame # Send original frame if processing fails

        # Update overall status for the stream
        overall_stream_status = get_higher_priority_status(overall_stream_status, frame_status)

        # Encode frame as JPEG
        try:
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                print("Error encoding frame to JPEG")
                continue
            frame_bytes = buffer.tobytes()

            # Yield the frame in the required format for MJPEG stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error encoding or yielding frame: {e}")
            break # Exit loop on encoding error

        frame_num += 1
        # Add a small delay - sometimes helps prevent overwhelming the processing/network
        # time.sleep(0.01) # Optional: control frame rate slightly

    # Cleanup
    live_cap.release()
    print("Live stream stopped.")


@app.route('/live')
def live_route():
    """Serves the page that will display the live stream."""
    return render_template('live.html')

@app.route('/video_feed')
def video_feed_route():
    """Provides the MJPEG stream."""
    # Uses a generator function and Flask's Response object for streaming
    return Response(generate_live_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --- Run App ---
if __name__ == '__main__':
    print("--- Starting Flask Server ---")
    # Connect to Fluvio once at the start if possible
    connect_fluvio()
    # Set use_reloader=False if TF model loading causes issues with reloading
    # Set debug=False for production/actual hackathon demo if stability is key
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
'''
'''
# app.py

# Required Imports
from flask import Flask, render_template, request, url_for
import os
import cv2 # OpenCV
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sys
import time
from fluvio import Fluvio
import json

# --- Flask Application Setup ---
app = Flask(__name__)

# --- Configuration for Folders ---
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

STATIC_FOLDER = 'static'
PROCESSED_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'processed_frames')
if not os.path.exists(PROCESSED_FRAMES_FOLDER): os.makedirs(PROCESSED_FRAMES_FOLDER)
DEBUG_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'debug_frames')
if not os.path.exists(DEBUG_FRAMES_FOLDER): os.makedirs(DEBUG_FRAMES_FOLDER)

# --- Load Machine Learning Model ---
DETECTOR_HANDLE = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
detector = None
try:
    print(f"Loading detection model from: {DETECTOR_HANDLE}...")
    detector = hub.load(DETECTOR_HANDLE)
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load the detection model: {e}")
    sys.exit("Model loading failed. Exiting.")

# --- Model Specific Settings ---
PERSON_CLASS_INDEX = 1
# *** TUNABLE: Adjust this based on detection results ***
DETECTION_THRESHOLD = 0.2 # Example: Lowered threshold slightly
DEBUG_LOGGING_THRESHOLD = 0.1

# --- Fluvio Settings ---
FLUVIO_CROWD_TOPIC = "crowd-data"

# --- Density Analysis Settings ---
# *** TUNABLE: LOWER THESE THRESHOLDS if overlays aren't appearing ***
# You need counts >= threshold to trigger the color. Check console output!
HIGH_DENSITY_THRESHOLD = 5   # People per cell for 'high' (Orange)
CRITICAL_DENSITY_THRESHOLD = 7 # People per cell for 'critical' (Red)

# Thresholds for overall status based on *number* of dense cells
HIGH_DENSITY_CELL_COUNT_THRESHOLD = 3 # Number of 'high' cells for "High Density Warning"
CRITICAL_DENSITY_CELL_COUNT_THRESHOLD = 2 # Number of 'critical' cells for "CRITICAL RISK"

# Grid dimensions
GRID_ROWS = 10
GRID_COLS = 10

# --- Helper Function: Analyze Density Grid ---
def analyze_density_grid_for_frame(density_grid):
    high_density_cells = 0
    critical_density_cells = 0
    risky_cell_details = []
    overall_status = "Normal"
    total_people_in_grid = 0

    print("  Analyzing Density Grid:") # DEBUG PRINT
    if not density_grid or len(density_grid) != GRID_ROWS:
        print("    Warning: Invalid density grid received.")
        return overall_status, risky_cell_details, total_people_in_grid

    for r_idx, row in enumerate(density_grid):
        if len(row) != GRID_COLS: continue
        for c_idx, count in enumerate(row):
            try:
                person_count = int(count)
                total_people_in_grid += person_count
                is_risky = False # Flag to check if we print this cell's count
                if person_count >= CRITICAL_DENSITY_THRESHOLD:
                    critical_density_cells += 1
                    risky_cell_details.append((r_idx, c_idx, "critical"))
                    print(f"    Cell ({r_idx},{c_idx}): Count={person_count} >= CRITICAL ({CRITICAL_DENSITY_THRESHOLD}) -> Marked CRITICAL") # DEBUG PRINT
                    is_risky = True
                elif person_count >= HIGH_DENSITY_THRESHOLD:
                    high_density_cells += 1
                    risky_cell_details.append((r_idx, c_idx, "high"))
                    print(f"    Cell ({r_idx},{c_idx}): Count={person_count} >= HIGH ({HIGH_DENSITY_THRESHOLD}) -> Marked HIGH") # DEBUG PRINT
                    is_risky = True
                # Optional: Print counts even for non-risky cells if needed for tuning
                # elif person_count > 0:
                #    print(f"    Cell ({r_idx},{c_idx}): Count={person_count} (Below High Threshold)")
            except (ValueError, TypeError):
                continue # Skip non-integer counts

    # Determine overall status
    if critical_density_cells >= CRITICAL_DENSITY_CELL_COUNT_THRESHOLD:
        overall_status = "CRITICAL RISK"
    elif critical_density_cells > 0:
        overall_status = "Critical Density Cell Detected"
    elif high_density_cells >= HIGH_DENSITY_CELL_COUNT_THRESHOLD:
        overall_status = "High Density Warning"
    elif high_density_cells > 0:
        overall_status = "High Density Cell Detected"

    print(f"  Analysis Result: Status='{overall_status}', Risky Cells Found={len(risky_cell_details)}, Total People={total_people_in_grid}") # DEBUG PRINT
    return overall_status, risky_cell_details, total_people_in_grid

# --- Status Priority Helper ---
STATUS_HIERARCHY = {
    "Normal": 0, "High Density Cell Detected": 1, "High Density Warning": 2,
    "Critical Density Cell Detected": 3, "CRITICAL RISK": 4
}
def get_higher_priority_status(status1, status2):
    p1 = STATUS_HIERARCHY.get(status1, -1); p2 = STATUS_HIERARCHY.get(status2, -1)
    return status1 if p1 >= p2 else status2

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    print("\n--- Request received for /upload ---")
    sys.stdout.flush()

    # Check 1: Does the 'video' key exist in the uploaded files?
    if 'video' not in request.files:
        print("Upload Error: 'video' key not found in request.files.")
        return 'No video file part in the request', 400

    # If it exists, assign it to the variable
    video_file = request.files['video']

    # Check 2: Now that we have video_file, is its filename empty?
    if video_file.filename == '':
        print("Upload Error: No file selected (filename is empty).")
        return 'No selected video file', 400

    # If we passed both checks, it's safe to proceed
    print(f"Received video file: {video_file.filename}")

    # --- Fluvio Setup ---
    fluvio_producer = None; fluvio_client = None
    try:
        print("Connecting to Fluvio..."); fluvio_client = Fluvio.connect()
        print("Getting Fluvio producer..."); fluvio_producer = fluvio_client.topic_producer(FLUVIO_CROWD_TOPIC)
        print("Fluvio ready.")
    except Exception as e: print(f"!!! FLUVIO WARNING: {e}")

    # --- Video Saving and Opening ---
    filename = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    try: video_file.save(filename); print(f"Video saved: {filename}")
    except Exception as e: return f"Error saving video: {e}", 500

    video_path = filename
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return f"Error opening video: {video_path}", 500

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video opened: {width}x{height} @ {fps:.2f} FPS")

    processed_frame_filenames = []
    frame_count = 0
    overall_video_status = "Normal"
    max_people_in_frame = 0

    cell_height = height // GRID_ROWS; cell_width = width // GRID_COLS
    if cell_height <= 0 or cell_width <= 0: return "Video dimensions incompatible with grid size.", 500

    print("Starting frame processing loop...")
    while True:
        ret, frame = cap.read()
        if not ret: print(f"End of video. Frames processed: {frame_count}"); break

        # --- ML Preprocessing & Detection ---
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_tensor = tf.expand_dims(tf.convert_to_tensor(rgb_frame, dtype=tf.uint8), axis=0)
            detections = detector(image_tensor)
            boxes = detections['detection_boxes'][0].numpy()
            classes = detections['detection_classes'][0].numpy().astype(int)
            scores = detections['detection_scores'][0].numpy()
        except Exception as e: print(f"Frame {frame_count}: Error in ML processing: {e}"); frame_count+=1; continue

        # --- Calculate Grid & Debug ---
        density_grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        person_bboxes_in_frame_for_final_drawing = []
        confirmed_person_count_this_frame = 0
        debug_frame = frame.copy()
        print(f"--- Frame {frame_count} Raw Detections (Score > {DEBUG_LOGGING_THRESHOLD}) ---")
        class_colors = {PERSON_CLASS_INDEX: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255)}
        default_color = (255, 255, 0)
        total_detections_logged = 0

        for i in range(boxes.shape[0]):
            class_id = classes[i]; score = scores[i]
            if score >= DEBUG_LOGGING_THRESHOLD:
                total_detections_logged += 1
                ymin, xmin, ymax, xmax = boxes[i]
                xmin_abs, xmax_abs = int(xmin * width), int(xmax * width)
                ymin_abs, ymax_abs = int(ymin * height), int(ymax * height)
                # print(f"  Det {i}: ID={class_id}, S={score:.2f}, Box=[{xmin_abs},{ymin_abs},{xmax_abs},{ymax_abs}]") # Optionally uncomment
                box_color = class_colors.get(class_id, default_color)
                cv2.rectangle(debug_frame, (xmin_abs, ymin_abs), (xmax_abs, ymax_abs), box_color, 1)
                label = f"ID:{class_id} S:{score:.2f}"
                cv2.putText(debug_frame, label, (xmin_abs, ymin_abs - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

                if class_id == PERSON_CLASS_INDEX and score >= DETECTION_THRESHOLD:
                    confirmed_person_count_this_frame += 1
                    person_bboxes_in_frame_for_final_drawing.append((xmin_abs, ymin_abs, xmax_abs, ymax_abs))
                    center_x, center_y = (xmin_abs + xmax_abs) // 2, (ymin_abs + ymax_abs) // 2
                    row, col = min(center_y // cell_height, GRID_ROWS - 1), min(center_x // cell_width, GRID_COLS - 1)
                    if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS: density_grid[row][col] += 1

        print(f"  Total Dets Logged: {total_detections_logged}, Confirmed Persons: {confirmed_person_count_this_frame}")
        print(f"  Density Grid (Frame {frame_count}): {density_grid}") # Print grid directly

        # Save Debug Frame
        debug_output_filename = f"debug_frame_{frame_count:05d}.jpg"
        debug_output_filepath = os.path.join(DEBUG_FRAMES_FOLDER, debug_output_filename)
        try: cv2.imwrite(debug_output_filepath, debug_frame)
        except Exception as e: print(f"Error saving DEBUG frame {frame_count}: {e}")

        # --- Fluvio Send ---
        if fluvio_producer:
            fluvio_data = json.dumps({"timestamp": int(time.time()), "frame": frame_count, "density_grid": density_grid, "confirmed_persons": confirmed_person_count_this_frame})
            try: fluvio_producer.send(f"frame-{frame_count}".encode('utf-8'), fluvio_data.encode('utf-8'))
            except Exception as e: print(f"!!! FLUVIO Send WARNING: {e}")

        # --- Analyze Density Grid ---
        frame_status, risky_cells, total_people_in_grid = analyze_density_grid_for_frame(density_grid)
        if total_people_in_grid > max_people_in_frame: max_people_in_frame = total_people_in_grid
        overall_video_status = get_higher_priority_status(overall_video_status, frame_status)
        print(f"  Frame {frame_count} Status: {frame_status}, Risky Cells Details: {risky_cells}") # DEBUG PRINT risky_cells

        # --- Draw Overlays and Final Boxes ---
        final_frame = frame # Start fresh from original frame
        overlay = final_frame.copy()
        alpha = 0.4
        overlay_color_critical = (0, 0, 255) # Red (BGR)
        overlay_color_high = (0, 165, 255) # Orange (BGR)

        # print(f"  Attempting to draw overlays for {len(risky_cells)} risky cells...") # DEBUG PRINT
        # for r, c, risk_level in risky_cells: # Iterates only through HIGH or CRITICAL cells
        #     cell_y_start, cell_y_end = r * cell_height, min((r + 1) * cell_height, height)
        #     cell_x_start, cell_x_end = c * cell_width, min((c + 1) * cell_width, width)
        #     color = overlay_color_critical if risk_level == "critical" else overlay_color_high
        #     print(f"    Drawing {risk_level} overlay at Cell({r},{c}) with color {color}") # DEBUG PRINT
        #     cv2.rectangle(overlay, (cell_x_start, cell_y_start), (cell_x_end, cell_y_end), color, -1)

                # --- Draw Crowd Area Overlay (Instead of Grid Cells) ---
        overlay = final_frame.copy() # Start fresh overlay layer
        alpha = 0.4
        overlay_color_critical = (0, 0, 255) # Red (BGR)
        overlay_color_high = (0, 165, 255) # Orange (BGR)

        if "CRITICAL" in frame_status or "High" in frame_status: # Draw if any warning/critical status
            # Find bounding box of ALL confirmed persons for simplicity
            all_boxes = person_bboxes_in_frame_for_final_drawing
            if all_boxes: # Ensure there are people detected
                min_crowd_x = min(box[0] for box in all_boxes)
                min_crowd_y = min(box[1] for box in all_boxes)
                max_crowd_x = max(box[2] for box in all_boxes)
                max_crowd_y = max(box[3] for box in all_boxes)

                overlay_color = overlay_color_critical if "CRITICAL" in frame_status else overlay_color_high
                print(f"    Drawing {frame_status} overlay over crowd area: ({min_crowd_x},{min_crowd_y}) to ({max_crowd_x},{max_crowd_y})")
                cv2.rectangle(overlay, (min_crowd_x, min_crowd_y), (max_crowd_x, max_crowd_y), overlay_color, -1)

        # Apply the overlay (whether it was drawn on or is still blank)
        cv2.addWeighted(overlay, alpha, final_frame, 1 - alpha, 0, final_frame)
        # --- End of New Crowd Area Overlay ---

        cv2.addWeighted(overlay, alpha, final_frame, 1 - alpha, 0, final_frame)

        # Draw Green Person Boxes
        box_color = (0, 255, 0); thickness = 2
        for x1, y1, x2, y2 in person_bboxes_in_frame_for_final_drawing:
             cv2.rectangle(final_frame, (x1, y1), (x2, y2), box_color, thickness)

        # Add Status Text
        status_color = (0,0,0)
        if "CRITICAL" in frame_status: status_color = (0, 0, 255)
        elif "Warning" in frame_status or "High" in frame_status: status_color = (0, 165, 255)
        elif "Normal" in frame_status: status_color = (0, 128, 0)
        cv2.putText(final_frame, f"F:{frame_count} S:{frame_status} P:{total_people_in_grid}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(final_frame, f"F:{frame_count} S:{frame_status} P:{total_people_in_grid}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)


        # --- Save Final Processed Frame ---
        output_filename_base = f"frame_{frame_count:05d}.jpg"
        output_filepath = os.path.join(PROCESSED_FRAMES_FOLDER, output_filename_base)
        try: cv2.imwrite(output_filepath, final_frame)
        except Exception as e: print(f"Error saving processed frame {frame_count}: {e}")
        processed_frame_filenames.append(output_filename_base)

        frame_count += 1
        sys.stdout.flush()

    # --- Cleanup ---
    cap.release()
    if fluvio_client:
        if fluvio_producer: del fluvio_producer
        del fluvio_client
        print("Fluvio cleanup done.")
    try: os.remove(filename); print(f"Removed temp file: {filename}")
    except OSError as e: print(f"Warning removing temp file: {e}")

    # --- Render Results ---
    print(f"Rendering results. Overall Status: {overall_video_status}, Max Persons in Grid: {max_people_in_frame}")
    return render_template('results.html',
                           processed_frames=processed_frame_filenames,
                           prediction_status=overall_video_status,
                           max_persons=max_people_in_frame)

# --- Run App ---
if __name__ == '__main__':
    print("--- Starting Flask Server ---")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
'''


'''
# app.py

# Required Imports
from flask import Flask, render_template, request, url_for # Added url_for
import os
import cv2 # OpenCV for video/image processing
import tensorflow as tf # TensorFlow for the ML model
import tensorflow_hub as hub # For loading models from TensorFlow Hub
import numpy as np # Added numpy
import sys # For system-level operations like printing/exiting
import time # To get timestamps for Fluvio data
from fluvio import Fluvio # Fluvio Python client
import json # For handling JSON data

# --- Flask Application Setup ---
app = Flask(__name__)

# --- Configuration for Folders ---
UPLOAD_FOLDER = 'uploads' # Folder to temporarily store uploaded videos
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

STATIC_FOLDER = 'static' # Standard folder for static files (CSS, JS, images)
PROCESSED_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'processed_frames') # Subfolder for output frames
if not os.path.exists(PROCESSED_FRAMES_FOLDER):
    os.makedirs(PROCESSED_FRAMES_FOLDER)
DEBUG_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'debug_frames') # Subfolder for debug frames
if not os.path.exists(DEBUG_FRAMES_FOLDER):
        os.makedirs(DEBUG_FRAMES_FOLDER)

# --- Load Machine Learning Model ---
# Using the standard TF Hub URL for SSD MobileNet V2 FPNLite 320x320
DETECTOR_HANDLE = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
detector = None # Initialize detector
try:
    print(f"Loading detection model from: {DETECTOR_HANDLE}...")
    # Set TF_HUB_CACHE_DIR environment variable *before* loading if needed
    # os.environ['TF_HUB_CACHE_DIR'] = '/path/to/your/cache' # Optional: specify cache dir
    detector = hub.load(DETECTOR_HANDLE)
    print("Model loaded successfully.")
    # You could optionally print model signatures here to understand inputs/outputs
    # print(list(detector.signatures.keys())) # E.g., ['serving_default']
    # concrete_func = detector.signatures['serving_default']
    # print(concrete_func.structured_outputs)
except Exception as e:
    print(f"FATAL ERROR: Could not load the detection model from {DETECTOR_HANDLE}")
    print(f"Error details: {e}")
    print("Ensure TensorFlow Hub can access the URL, the model format is compatible,")
    print("and you have internet connectivity. Check TF Hub cache permissions if applicable.")
    if "Connection refused" in str(e) or "Temporary failure in name resolution" in str(e):
         print("Hint: Check your internet connection or firewall settings.")
    sys.exit("Model loading failed. Exiting.")

# --- Model Specific Settings ---
# IMPORTANT: Verify the class index for 'person' in your chosen model's dataset (COCO). It's typically 1.
PERSON_CLASS_INDEX = 1
# Minimum confidence score for a detection to be considered valid for the *final analysis*
# *** TUNABLE PARAMETER: Try lowering this (e.g., 0.2, 0.15) if too few persons are detected ***
DETECTION_THRESHOLD = 0.10
# Lower threshold for logging/debugging purposes to see *all* potential detections
DEBUG_LOGGING_THRESHOLD = 0.1 # Log anything model detects with >10% confidence

# --- Fluvio Settings ---
FLUVIO_CROWD_TOPIC = "crowd-data" # Topic to send person counts to

# --- Density Analysis Settings ---
# *** TUNABLE PARAMETERS: Adjust these based on observed grid counts and desired sensitivity ***
MODERATE_DENSITY_THRESHOLD = 3 # People per cell for 'moderate' (currently unused, but could add later)
HIGH_DENSITY_THRESHOLD = 5     # People per cell for 'high'
CRITICAL_DENSITY_THRESHOLD = 8 # People per cell for 'critical'
# How many cells of a certain density trigger overall warnings
HIGH_DENSITY_CELL_COUNT_THRESHOLD = 4 # Number of 'high' cells needed for "High Density Warning"
CRITICAL_DENSITY_CELL_COUNT_THRESHOLD = 2 # Number of 'critical' cells needed for "CRITICAL RISK"

# Grid dimensions for density analysis
GRID_ROWS = 10
GRID_COLS = 10

# --- Helper Function: Analyze Density Grid ---
def analyze_density_grid_for_frame(density_grid):
    """
    Analyzes the density grid for a single frame.
    Returns:
        tuple: (frame_status_string, list_of_risky_cells)
               frame_status_string: e.g., "Normal", "High Density Warning", "CRITICAL RISK"
               list_of_risky_cells: list of tuples [(row, col, 'critical'/'high'), ...]
    """
    high_density_cells = 0
    critical_density_cells = 0
    risky_cell_details = [] # Store tuples: (row, col, 'critical' or 'high')
    overall_status = "Normal"
    total_people_in_grid = 0

    if not density_grid or len(density_grid) != GRID_ROWS: # Basic validation
        print(f"Warning: Received invalid density grid for analysis.")
        return overall_status, risky_cell_details, total_people_in_grid

    for r_idx, row in enumerate(density_grid):
        if len(row) != GRID_COLS:
            print(f"Warning: Row {r_idx} has incorrect number of columns.")
            continue # Skip malformed row
        for c_idx, count in enumerate(row):
            try:
                person_count = int(count) # Ensure it's an integer
                total_people_in_grid += person_count
                if person_count >= CRITICAL_DENSITY_THRESHOLD:
                    critical_density_cells += 1
                    risky_cell_details.append((r_idx, c_idx, "critical"))
                elif person_count >= HIGH_DENSITY_THRESHOLD:
                    high_density_cells += 1
                    risky_cell_details.append((r_idx, c_idx, "high"))
            except (ValueError, TypeError):
                print(f"Warning: Invalid count '{count}' at cell ({r_idx},{c_idx}). Skipping.")
                continue

    # --- Determine overall status based on cell counts ---
    if critical_density_cells >= CRITICAL_DENSITY_CELL_COUNT_THRESHOLD:
        overall_status = "CRITICAL RISK"
    elif critical_density_cells > 0: # Even one critical cell might be worth escalating
        overall_status = "Critical Density Cell Detected"
    elif high_density_cells >= HIGH_DENSITY_CELL_COUNT_THRESHOLD:
        overall_status = "High Density Warning"
    elif high_density_cells > 0:
        overall_status = "High Density Cell Detected"
    # else: status remains "Normal"

    # print(f"Analyzed Grid: Status='{overall_status}', Risky Cells={len(risky_cell_details)}, Total People={total_people_in_grid}") # Debug print
    return overall_status, risky_cell_details, total_people_in_grid

# --- Status Priority Helper ---
STATUS_HIERARCHY = {
    "Normal": 0,
    "High Density Cell Detected": 1,
    "High Density Warning": 2,
    "Critical Density Cell Detected": 3,
    "CRITICAL RISK": 4
}

def get_higher_priority_status(status1, status2):
    """Returns the status with the higher priority based on STATUS_HIERARCHY."""
    priority1 = STATUS_HIERARCHY.get(status1, -1)
    priority2 = STATUS_HIERARCHY.get(status2, -1)
    return status1 if priority1 >= priority2 else status2

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page (index.html)."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Handles video upload, processes frames, sends data, and renders results.
    """
    print("\n--- Request received for /upload ---")
    sys.stdout.flush()

    if 'video' not in request.files:
        print("Upload Error: 'video' part not found.")
        return 'No video file part in the request', 400
    video_file = request.files['video']
    if video_file.filename == '':
        print("Upload Error: No file selected.")
        return 'No selected video file', 400

    print(f"Received video file: {video_file.filename}")

    # --- Initialize Fluvio Producer ---
    fluvio_producer = None
    fluvio_client = None
    try:
        print("Attempting to connect to Fluvio...")
        fluvio_client = Fluvio.connect()
        print("Fluvio client connected.")
        fluvio_producer = fluvio_client.topic_producer(FLUVIO_CROWD_TOPIC)
        print(f"Fluvio producer ready for topic '{FLUVIO_CROWD_TOPIC}'.")
    except Exception as e:
        print(f"!!! FLUVIO WARNING: Could not connect or get producer: {e}")
        print("!!! Video processing will continue, but crowd data will NOT be sent to Fluvio.")

    # --- Process the video file ---
    filename = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    try:
        video_file.save(filename)
        print(f"Video temporarily saved to: {filename}")
    except Exception as e:
        print(f"Error saving video file '{filename}': {e}")
        if fluvio_client: del fluvio_client
        return f"Error saving video file: {e}", 500

    video_path = filename
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file with OpenCV: {video_path}")
        if fluvio_client: del fluvio_client
        try: os.remove(filename) # Clean up
        except OSError: pass
        return f"Error opening video file: {video_path}", 500
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video opened: {video_path} ({width}x{height} @ {fps:.2f} FPS)")

    processed_frame_filenames = []
    frame_count = 0
    overall_video_status = "Normal"
    max_people_in_frame = 0 # Track max people detected in any single frame grid

    # Pre-calculate cell dimensions
    cell_height = height // GRID_ROWS
    cell_width = width // GRID_COLS
    if cell_height <= 0 or cell_width <= 0: # Check for valid dimensions
        print(f"ERROR: Video dimensions ({width}x{height}) too small or grid ({GRID_ROWS}x{GRID_COLS}) too large. Cell dimensions are zero or negative.")
        cap.release()
        if fluvio_client: del fluvio_client
        try: os.remove(filename)
        except OSError: pass
        return "Video dimensions incompatible with grid size.", 500

    print("Starting frame processing loop...")
    while True:
        try:
            ret, frame = cap.read()
        except Exception as e:
            print(f"Error reading frame {frame_count} from video: {e}")
            break

        if not ret:
            print(f"End of video or cannot read frame {frame_count}. Total frames processed: {frame_count}")
            break

        # --- Prepare frame for the ML model ---
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Model expects uint8 [1, height, width, 3]
            image_tensor = tf.convert_to_tensor(rgb_frame, dtype=tf.uint8)
            image_tensor = tf.expand_dims(image_tensor, axis=0)
        except Exception as e:
            print(f"Error converting frame {frame_count} for TF: {e}. Skipping frame.")
            frame_count += 1
            continue

        # --- Run Object Detection ---
        try:
            # start_time = time.time() # Optional timing
            detections = detector(image_tensor) # Use the globally loaded detector
            # end_time = time.time()
            # print(f"Frame {frame_count} detection time: {end_time - start_time:.4f}s")
        except Exception as e:
            print(f"Error running ML model detection on frame {frame_count}: {e}. Skipping frame.")
            frame_count += 1
            continue

        # --- Extract detection results ---
        try:
            # Output tensors are typically float32, need conversion
            boxes = detections['detection_boxes'][0].numpy()
            classes = detections['detection_classes'][0].numpy().astype(int)
            scores = detections['detection_scores'][0].numpy()
        except Exception as e:
            print(f"Error processing detection results for frame {frame_count}: {e}. Skipping frame.")
            frame_count += 1
            continue

        # --- Calculate Density Grid & Debug Detections ---
        density_grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        person_bboxes_in_frame_for_final_drawing = [] # For final green boxes
        confirmed_person_count_this_frame = 0

        print(f"--- Frame {frame_count} Raw Detections (Score > {DEBUG_LOGGING_THRESHOLD}) ---") # Log detections above debug threshold

        # Create a temporary frame copy for drawing ALL detected boxes (debugging)
        debug_frame = frame.copy() # Start with original BGR frame
        debug_thickness = 1

        # Define colors for different classes (add more if needed from COCO map)
        class_colors = {
            PERSON_CLASS_INDEX: (0, 255, 0), # Green for Person (BGR)
            2: (255, 0, 0),    # Blue for Bicycle
            3: (0, 0, 255),    # Red for Car
            # ... add others if you identify common misclassifications
        }
        default_color = (255, 255, 0) # Cyan for other classes (BGR)

        total_detections_logged = 0
        for i in range(boxes.shape[0]): # Iterate through all detected boxes
            class_id = classes[i]
            score = scores[i]

            # Log any detection above the debug threshold
            if score >= DEBUG_LOGGING_THRESHOLD:
                total_detections_logged += 1
                ymin, xmin, ymax, xmax = boxes[i] # These are normalized (0.0 to 1.0)

                # Convert normalized coords to absolute pixel coords
                xmin_abs = int(xmin * width)
                xmax_abs = int(xmax * width)
                ymin_abs = int(ymin * height)
                ymax_abs = int(ymax * height)

                # Print details of this detection
                print(f"  Detection {i}: ClassID={class_id}, Score={score:.2f}, Box=[{xmin_abs},{ymin_abs},{xmax_abs},{ymax_abs}]")

                # Draw bounding box on the debug frame, color-coded by class
                box_color = class_colors.get(class_id, default_color)
                cv2.rectangle(debug_frame, (xmin_abs, ymin_abs), (xmax_abs, ymax_abs), box_color, debug_thickness)
                # Label with Class ID and Score
                label = f"ID:{class_id} S:{score:.2f}"
                cv2.putText(debug_frame, label, (xmin_abs, ymin_abs - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

                # --- Logic for Confirmed Persons for Density Grid & Final Output ---
                # Check if it's the PERSON class AND meets the main DETECTION_THRESHOLD
                if class_id == PERSON_CLASS_INDEX and score >= DETECTION_THRESHOLD:
                    confirmed_person_count_this_frame += 1
                    # Store coords for the final green box drawing
                    person_bboxes_in_frame_for_final_drawing.append((xmin_abs, ymin_abs, xmax_abs, ymax_abs))

                    # Calculate center point for grid assignment
                    center_x = (xmin_abs + xmax_abs) // 2
                    center_y = (ymin_abs + ymax_abs) // 2

                    # Assign person to grid cell based on center point
                    row = min(center_y // cell_height, GRID_ROWS - 1) # Clamp to grid bounds
                    col = min(center_x // cell_width, GRID_COLS - 1)  # Clamp to grid bounds

                    # Check if indices are valid before assignment
                    if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
                         density_grid[row][col] += 1
                    else:
                         # This should ideally not happen if clamping works, but good to check
                         print(f"    WARNING: Invalid grid indices ({row},{col}) calculated for person at ({center_x},{center_y}). Skipping assignment.")


        print(f"  Total detections logged this frame (Score > {DEBUG_LOGGING_THRESHOLD}): {total_detections_logged}")
        print(f"  Confirmed PERSONS this frame (ClassID={PERSON_CLASS_INDEX}, Score >= {DETECTION_THRESHOLD}): {confirmed_person_count_this_frame}")

        # --- Print the Density Grid (based only on confirmed persons) ---
        print(f"  Density Grid (Frame {frame_count}):")
        for row_idx, row_data in enumerate(density_grid):
            print(f"    Row {row_idx}: {row_data}")
        # sys.stdout.flush() # Flush output buffer to see prints immediately

        # --- Save the DEBUG frame (shows ALL potential objects with IDs/Scores) ---
        debug_output_filename = f"debug_frame_{frame_count:05d}.jpg"
        debug_output_filepath = os.path.join(DEBUG_FRAMES_FOLDER, debug_output_filename) # Save to separate debug folder
        try:
            cv2.imwrite(debug_output_filepath, debug_frame)
        except Exception as e:
            print(f"Error saving DEBUG frame {frame_count} to {debug_output_filepath}: {e}")


        # --- Send density grid to Fluvio ---
        if fluvio_producer:
            current_timestamp = int(time.time())
            # Send the grid calculated from *confirmed* persons
            fluvio_data = json.dumps({
                "timestamp": current_timestamp,
                "frame": frame_count,
                "density_grid": density_grid,
                "confirmed_persons": confirmed_person_count_this_frame # Also send count for easy checking
            })
            try:
                # Using frame number as key (simple approach)
                fluvio_producer.send(f"frame-{frame_count}".encode('utf-8'), fluvio_data.encode('utf-8'))
            except Exception as e:
                print(f"!!! FLUVIO WARNING: Could not send density grid for frame {frame_count}: {e}")


        # --- Analyze density grid for this frame ---
        # Use the grid calculated from confirmed persons
        frame_status, risky_cells, total_people_in_grid = analyze_density_grid_for_frame(density_grid)
        print(f"  Analysis: Frame Status='{frame_status}', Risky Cells={len(risky_cells)}, Total Persons in Grid={total_people_in_grid}")
        if total_people_in_grid > max_people_in_frame:
             max_people_in_frame = total_people_in_grid # Update max count seen


        # --- Update Overall Video Status ---
        overall_video_status = get_higher_priority_status(overall_video_status, frame_status)


        # --- Draw Red/Orange Overlays for High-Risk Cells on FINAL frame ---
        # Start with the original frame for final output modifications
        final_frame = frame # We'll draw overlays and green boxes on this
        overlay = final_frame.copy() # Create a copy for blending overlays
        alpha = 0.4 # Transparency factor
        overlay_color_critical = (0, 0, 255) # Red (BGR)
        overlay_color_high = (0, 165, 255) # Orange (BGR)

        for r, c, risk_level in risky_cells:
            cell_y_start = r * cell_height
            cell_y_end = min((r + 1) * cell_height, height) # Ensure not exceeding frame bounds
            cell_x_start = c * cell_width
            cell_x_end = min((c + 1) * cell_width, width)   # Ensure not exceeding frame bounds
            color = overlay_color_critical if risk_level == "critical" else overlay_color_high
            # Draw filled rectangle on the overlay copy
            cv2.rectangle(overlay, (cell_x_start, cell_y_start), (cell_x_end, cell_y_end), color, -1) # Use overlay frame

        # Blend the overlay with the final frame
        cv2.addWeighted(overlay, alpha, final_frame, 1 - alpha, 0, final_frame)
        # 'final_frame' now contains the blended overlay


        # --- Draw CONFIRMED Person Bounding Boxes (Green) on top of overlay ---
        box_color = (0, 255, 0) # Green (BGR)
        thickness = 2
        for x1, y1, x2, y2 in person_bboxes_in_frame_for_final_drawing: # Use the filtered list
             cv2.rectangle(final_frame, (x1, y1), (x2, y2), box_color, thickness)


        # --- Add Status Text to the FINAL frame ---
        status_color = (0, 0, 0) # Default Black
        if "CRITICAL" in frame_status:
            status_color = (0, 0, 255) # Red
        elif "Warning" in frame_status or "High" in frame_status:
            status_color = (0, 165, 255) # Orange
        elif "Normal" in frame_status:
             status_color = (0, 128, 0) # Dark Green

        cv2.putText(final_frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 3, cv2.LINE_AA) # White Outline
        cv2.putText(final_frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA) # Black Text
        cv2.putText(final_frame, f"Status: {frame_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 3, cv2.LINE_AA) # White Outline
        cv2.putText(final_frame, f"Status: {frame_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA) # Status Color Text
        cv2.putText(final_frame, f"Persons in Grid: {total_people_in_grid}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 3, cv2.LINE_AA) # White Outline
        cv2.putText(final_frame, f"Persons in Grid: {total_people_in_grid}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA) # Black Text


        # --- Save the PROCESSED FINAL frame (with overlays and boxes) ---
        output_filename_base = f"frame_{frame_count:05d}.jpg" # Use padding for sorting
        output_filepath = os.path.join(PROCESSED_FRAMES_FOLDER, output_filename_base)
        try:
            cv2.imwrite(output_filepath, final_frame) # Save the final frame with overlays/boxes
            processed_frame_filenames.append(output_filename_base)
        except Exception as e:
            print(f"Error saving processed frame {frame_count} to {output_filepath}: {e}")

        frame_count += 1
        sys.stdout.flush() # Flush after processing each frame
        # Optional: time.sleep(0.01)

    # --- End of video processing loop ---
    print("Finished processing all frames.")

    # --- Clean up resources ---
    cap.release()
    print("Video capture released.")
    if fluvio_client:
        # Proper cleanup might involve specific methods if available
        # For now, just delete references
        if fluvio_producer: del fluvio_producer
        del fluvio_client
        print("Fluvio client reference deleted.")

    # Remove the original uploaded file
    try:
        os.remove(filename)
        print(f"Removed temporary uploaded file: {filename}")
    except OSError as e:
        print(f"Warning: Error removing uploaded file {filename}: {e}")

    # --- Render the results page ---
    print(f"Rendering results page. Overall video status: {overall_video_status}")
    print(f"Maximum number of people detected in the grid for any single frame: {max_people_in_frame}")
    return render_template('results.html',
                           processed_frames=processed_frame_filenames,
                           prediction_status=overall_video_status,
                           max_persons=max_people_in_frame) # Pass the final overall status and max count

# --- Run the Flask Application ---
if __name__ == '__main__':
    # Set debug=False for production/hackathon submission
    # use_reloader=False helps prevent issues with TF models loading twice in debug mode
    print("--- Starting Flask Development Server ---")
    print("Access the application via http://127.0.0.1:5000 or http://<your-ip>:5000")
    # Ensure HOST is 0.0.0.0 to be accessible on your network if needed
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
'''







# # app.py

# # Required Imports
# from flask import Flask, render_template, request, url_for # Added url_for
# import os
# import cv2 # OpenCV for video/image processing
# import tensorflow as tf # TensorFlow for the ML model
# import tensorflow_hub as hub # For loading models from TensorFlow Hub
# import numpy as np # Added numpy for array operations if needed later
# import sys # For system-level operations like printing/exiting
# import time # To get timestamps for Fluvio data
# from fluvio import Fluvio # Fluvio Python client
# import json # For handling JSON data

# # --- Flask Application Setup ---
# # It's conventional to use __name__ for the Flask app instance
# app = Flask(__name__)

# # --- Configuration for Folders ---
# UPLOAD_FOLDER = 'uploads' # Folder to temporarily store uploaded videos
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# STATIC_FOLDER = 'static' # Standard folder for static files (CSS, JS, images)
# PROCESSED_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'processed_frames') # Subfolder for output frames
# if not os.path.exists(PROCESSED_FRAMES_FOLDER):
#     os.makedirs(PROCESSED_FRAMES_FOLDER)

# # --- Load Machine Learning Model ---
# # Using the standard TF Hub URL for SSD MobileNet V2 FPNLite 320x320
# DETECTOR_HANDLE = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
# # Note: If you previously confirmed the Kaggle handle worked and prefer it, you can switch back.
# # DETECTOR_HANDLE = "https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/TensorFlow2/fpnlite-320x320/1"
# detector = None # Initialize detector
# try:
#     print(f"Loading detection model from: {DETECTOR_HANDLE}...")
#     # Set TF_HUB_CACHE_DIR environment variable *before* loading if needed
#     # os.environ['TF_HUB_CACHE_DIR'] = '/path/to/your/cache' # Optional: specify cache dir
#     detector = hub.load(DETECTOR_HANDLE)
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"FATAL ERROR: Could not load the detection model from {DETECTOR_HANDLE}")
#     print(f"Error details: {e}")
#     print("Ensure TensorFlow Hub can access the URL, the model format is compatible,")
#     print("and you have internet connectivity. Check TF Hub cache permissions if applicable.")
#     # Optionally provide guidance on specific errors (e.g., network, cache)
#     if "Connection refused" in str(e) or "Temporary failure in name resolution" in str(e):
#          print("Hint: Check your internet connection or firewall settings.")
#     sys.exit("Model loading failed. Exiting.")

# # --- Model Specific Settings ---
# # IMPORTANT: Verify the class index for 'person' in your chosen model's dataset (COCO). It's typically 1.
# PERSON_CLASS_INDEX = 1
# # Minimum confidence score for a detection to be considered valid
# DETECTION_THRESHOLD = 0.3 # You might need to adjust this

# # --- Fluvio Settings ---
# FLUVIO_CROWD_TOPIC = "crowd-data" # Topic to send person counts to

# # --- Density Analysis Settings (Moved from predict_stampede.py) ---
# MODERATE_DENSITY_THRESHOLD = 5
# HIGH_DENSITY_THRESHOLD = 10
# CRITICAL_DENSITY_THRESHOLD = 15
# HIGH_DENSITY_CELL_COUNT_THRESHOLD = 3
# CRITICAL_DENSITY_CELL_COUNT_THRESHOLD = 2 # Min number of critical cells to trigger highest alert

# # Grid dimensions for density analysis
# GRID_ROWS = 10
# GRID_COLS = 10

# # --- Helper Function: Analyze Density Grid (Adapted from predict_stampede.py) ---
# def analyze_density_grid_for_frame(density_grid):
#     """
#     Analyzes the density grid for a single frame.
#     Returns:
#         tuple: (frame_status_string, list_of_risky_cells)
#                frame_status_string: e.g., "Normal", "High Density Warning", "CRITICAL RISK"
#                list_of_risky_cells: list of tuples [(row, col, 'critical'/'high'), ...]
#     """
#     high_density_cells = 0
#     critical_density_cells = 0
#     risky_cell_details = [] # Store tuples: (row, col, 'critical' or 'high')
#     overall_status = "Normal"

#     if not density_grid or len(density_grid) != GRID_ROWS: # Basic validation
#          print(f"Warning: Received invalid density grid for analysis.")
#          return overall_status, risky_cell_details

#     for r_idx, row in enumerate(density_grid):
#          if len(row) != GRID_COLS:
#               print(f"Warning: Row {r_idx} has incorrect number of columns.")
#               continue # Skip malformed row
#          for c_idx, count in enumerate(row):
#             try:
#                  person_count = int(count) # Ensure it's an integer
#                  if person_count >= CRITICAL_DENSITY_THRESHOLD:
#                       critical_density_cells += 1
#                       risky_cell_details.append((r_idx, c_idx, "critical"))
#                  elif person_count >= HIGH_DENSITY_THRESHOLD:
#                       high_density_cells += 1
#                       risky_cell_details.append((r_idx, c_idx, "high"))
#             except (ValueError, TypeError):
#                  print(f"Warning: Invalid count '{count}' at cell ({r_idx},{c_idx}). Skipping.")
#                  continue

#     # --- Determine overall status based on cell counts ---
#     if critical_density_cells >= CRITICAL_DENSITY_CELL_COUNT_THRESHOLD:
#         overall_status = "CRITICAL RISK"
#     elif critical_density_cells > 0: # Even one critical cell might be worth escalating beyond just 'High'
#         overall_status = "Critical Density Cell Detected"
#     elif high_density_cells >= HIGH_DENSITY_CELL_COUNT_THRESHOLD:
#         overall_status = "High Density Warning"
#     elif high_density_cells > 0:
#         overall_status = "High Density Cell Detected"
#     # else: status remains "Normal"

#     # print(f"Analyzed Grid: Status='{overall_status}', Risky Cells={len(risky_cell_details)}") # Debug print
#     return overall_status, risky_cell_details

# # --- Status Priority Helper ---
# # Define a hierarchy for statuses to determine the overall video status
# STATUS_HIERARCHY = {
#     "Normal": 0,
#     "High Density Cell Detected": 1,
#     "High Density Warning": 2,
#     "Critical Density Cell Detected": 3,
#     "CRITICAL RISK": 4
# }

# def get_higher_priority_status(status1, status2):
#     """Returns the status with the higher priority based on STATUS_HIERARCHY."""
#     priority1 = STATUS_HIERARCHY.get(status1, -1)
#     priority2 = STATUS_HIERARCHY.get(status2, -1)
#     return status1 if priority1 >= priority2 else status2

# # --- Flask Routes ---
# @app.route('/', methods=['GET'])
# def index():
#     """Renders the main upload page (index.html)."""
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_video():
#     """
#     Handles video upload.
#     For each frame: detects persons, calculates density grid, analyzes density,
#     draws boxes and overlays, sends grid data to Fluvio.
#     Saves processed frames and renders the results page with overall status.
#     """
#     print("\n--- Request received for /upload ---")
#     sys.stdout.flush() # Ensure prints appear immediately

#     if 'video' not in request.files:
#         print("Upload Error: 'video' part not found in the request files.")
#         return 'No video file part in the request', 400
#     video_file = request.files['video']
#     if video_file.filename == '':
#         print("Upload Error: No file selected in the form.")
#         return 'No selected video file', 400

#     print(f"Received video file: {video_file.filename}")

#     # --- Initialize Fluvio Producer ---
#     fluvio_producer = None
#     fluvio_client = None
#     try:
#         print("Attempting to connect to Fluvio...")
#         fluvio_client = Fluvio.connect() # Adjust connection args if needed
#         print("Fluvio client connected.")
#         fluvio_producer = fluvio_client.topic_producer(FLUVIO_CROWD_TOPIC)
#         print(f"Fluvio producer ready for topic '{FLUVIO_CROWD_TOPIC}'.")
#     except Exception as e:
#         print(f"!!! FLUVIO WARNING: Could not connect or get producer: {e}")
#         print("!!! Video processing will continue, but crowd data will NOT be sent to Fluvio.")

#     # --- Process the video file ---
#     filename = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
#     try:
#         video_file.save(filename)
#         print(f"Video temporarily saved to: {filename}")
#     except Exception as e:
#         print(f"Error saving video file '{filename}': {e}")
#         if fluvio_client: del fluvio_client
#         return f"Error saving video file: {e}", 500

#     video_path = filename
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print(f"Error opening video file with OpenCV: {video_path}")
#         if fluvio_client: del fluvio_client
#         try: os.remove(filename) # Clean up saved file on error
#         except OSError: pass
#         return f"Error opening video file: {video_path}", 500
#     else:
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         print(f"Video opened: {video_path} ({width}x{height} @ {fps:.2f} FPS)")

#     processed_frame_filenames = []
#     frame_count = 0
#     overall_video_status = "Normal" # Track the highest risk status encountered

#     # Pre-calculate cell dimensions
#     cell_height = height // GRID_ROWS
#     cell_width = width // GRID_COLS
#     if cell_height == 0 or cell_width == 0:
#          print(f"ERROR: Video dimensions ({width}x{height}) too small for grid ({GRID_ROWS}x{GRID_COLS}). Cannot process.")
#          cap.release()
#          if fluvio_client: del fluvio_client
#          try: os.remove(filename)
#          except OSError: pass
#          return "Video dimensions too small for analysis grid.", 500

#     print("Starting frame processing loop...")
#     while True:
#         try:
#             ret, frame = cap.read()
#         except Exception as e:
#             print(f"Error reading frame {frame_count} from video: {e}")
#             break

#         if not ret:
#             print(f"End of video or cannot read frame {frame_count}. Total frames processed: {frame_count}")
#             break

#         # --- Prepare frame for the ML model ---
#         try:
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image_tensor = tf.convert_to_tensor(rgb_frame, dtype=tf.uint8) # Use uint8 directly if model supports it
#             image_tensor = tf.expand_dims(image_tensor, axis=0)
#         except Exception as e:
#             print(f"Error converting frame {frame_count} for TF: {e}. Skipping frame.")
#             frame_count += 1
#             continue

#         # --- Run Person Detection ---
#         try:
#             # start_time = time.time() # Optional timing
#             detections = detector(image_tensor) # Use the globally loaded detector
#             # end_time = time.time()
#             # print(f"Frame {frame_count} detection time: {end_time - start_time:.4f}s")
#         except Exception as e:
#             print(f"Error running ML model detection on frame {frame_count}: {e}. Skipping frame.")
#             frame_count += 1
#             continue

#         # --- Extract detection results ---
#         try:
#             boxes = detections['detection_boxes'][0].numpy()
#             classes = detections['detection_classes'][0].numpy().astype(int)
#             scores = detections['detection_scores'][0].numpy()
#         except Exception as e:
#             print(f"Error processing detection results for frame {frame_count}: {e}. Skipping frame.")
#             frame_count += 1
#             continue

#         # --- Calculate Density Grid ---
#         density_grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
#         person_bboxes_in_frame = [] # Store boxes for drawing later

#         for i in range(boxes.shape[0]):
#             if scores[i] >= DETECTION_THRESHOLD and classes[i] == PERSON_CLASS_INDEX:
#                 ymin, xmin, ymax, xmax = boxes[i]
#                 # Calculate absolute coordinates for bounding box
#                 xmin_abs = int(xmin * width)
#                 xmax_abs = int(xmax * width)
#                 ymin_abs = int(ymin * height)
#                 ymax_abs = int(ymax * height)

#                 person_bboxes_in_frame.append((xmin_abs, ymin_abs, xmax_abs, ymax_abs))

#                 # Calculate center for density grid assignment
#                 center_x = (xmin_abs + xmax_abs) // 2
#                 center_y = (ymin_abs + ymax_abs) // 2

#                 # Assign person to grid cell based on center point
#                 row = min(center_y // cell_height, GRID_ROWS - 1) # Clamp to grid bounds
#                 col = min(center_x // cell_width, GRID_COLS - 1)  # Clamp to grid bounds
#                 density_grid[row][col] += 1

#         # --- Send density grid to Fluvio ---
#         if fluvio_producer:
#             current_timestamp = int(time.time()) # Or use frame number / video time if preferred
#             fluvio_data = json.dumps({"timestamp": current_timestamp, "frame": frame_count, "density_grid": density_grid})
#             try:
#                 # Use a simple key or derive one if needed
#                 fluvio_producer.send(f"frame-{frame_count}".encode('utf-8'), fluvio_data.encode('utf-8'))
#             except Exception as e:
#                 print(f"!!! FLUVIO WARNING: Could not send density grid for frame {frame_count}: {e}")

#         # --- Analyze density grid for this frame ---
#         frame_status, risky_cells = analyze_density_grid_for_frame(density_grid)

#         # --- Update Overall Video Status ---
#         overall_video_status = get_higher_priority_status(overall_video_status, frame_status)

#         # --- Draw Red/Orange Overlays for High-Risk Cells ---
#         overlay_frame = frame.copy() # Work on a copy for blending
#         alpha = 0.4 # Transparency factor
#         overlay_color_critical = (0, 0, 255) # Red (BGR)
#         overlay_color_high = (0, 165, 255) # Orange (BGR)

#         for r, c, risk_level in risky_cells:
#             cell_y_start = r * cell_height
#             cell_y_end = (r + 1) * cell_height
#             cell_x_start = c * cell_width
#             cell_x_end = (c + 1) * cell_width
#             color = overlay_color_critical if risk_level == "critical" else overlay_color_high
#             # Draw filled rectangle on the overlay copy
#             cv2.rectangle(overlay_frame, (cell_x_start, cell_y_start), (cell_x_end, cell_y_end), color, -1)

#         # Blend the overlay with the original frame
#         cv2.addWeighted(overlay_frame, alpha, frame, 1 - alpha, 0, frame)
#         # 'frame' now contains the blended overlay

#         # --- Draw Person Bounding Boxes (on top of overlay) ---
#         box_color = (0, 255, 0) # Green (BGR)
#         thickness = 2
#         for x1, y1, x2, y2 in person_bboxes_in_frame:
#              cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

#         # --- Save the processed frame (with overlays and boxes) ---
#         output_filename_base = f"frame_{frame_count:05d}.jpg" # Use padding for sorting
#         output_filepath = os.path.join(PROCESSED_FRAMES_FOLDER, output_filename_base)
#         try:
#             cv2.imwrite(output_filepath, frame)
#             processed_frame_filenames.append(output_filename_base)
#         except Exception as e:
#             print(f"Error saving processed frame {frame_count} to {output_filepath}: {e}")

#         frame_count += 1
#         # Optional: Add a small delay if processing is too fast for Fluvio or slows down browser display
#         # time.sleep(0.01)

#     # --- End of video processing loop ---
#     print("Finished processing all frames.")

#     # --- Clean up resources ---
#     cap.release()
#     print("Video capture released.")
#     if fluvio_client:
#         # Proper cleanup might involve specific methods if available in future client versions
#         del fluvio_producer # Delete producer first
#         del fluvio_client
#         print("Fluvio client reference deleted.")

#     # Remove the original uploaded file
#     try:
#         os.remove(filename)
#         print(f"Removed temporary uploaded file: {filename}")
#     except OSError as e:
#         print(f"Warning: Error removing uploaded file {filename}: {e}")

#     # --- Render the results page ---
#     print(f"Rendering results page. Overall video status: {overall_video_status}")
#     return render_template('results.html',
#                            processed_frames=processed_frame_filenames,
#                            prediction_status=overall_video_status) # Pass the final overall status

# # --- Run the Flask Application ---
# if __name__ == '__main__':
#     # Set debug=False for production/hackathon submission if needed
#     # use_reloader=False can sometimes help with TF model loading issues during debugging
#     print("--- Starting Flask Development Server ---")
#     print("Access the application via http://127.0.0.1:5000 or http://<your-ip>:5000")
#     app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False) # Added use_reloader=False


'''
# Required Imports
from flask import Flask, render_template, request
import os
import cv2 # OpenCV for video/image processing
import tensorflow as tf # TensorFlow for the ML model
import tensorflow_hub as hub # For loading models from TensorFlow Hub
import sys # For system-level operations like printing/exiting
import time # To get timestamps for Fluvio data
from fluvio import Fluvio # Fluvio Python client
import json # For handling JSON data

# --- Flask Application Setup ---
app = Flask(__name__)

# --- Configuration for Folders ---
UPLOAD_FOLDER = 'uploads' # Folder to temporarily store uploaded videos
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

STATIC_FOLDER = 'static' # Standard folder for static files (CSS, JS, images)
PROCESSED_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'processed_frames') # Subfolder for output frames
if not os.path.exists(PROCESSED_FRAMES_FOLDER):
    os.makedirs(PROCESSED_FRAMES_FOLDER)

# --- Load Machine Learning Model ---
# Load the object detection model from TensorFlow Hub
# Using SSD MobileNet V2 FPNLite 320x320 - choose a model suitable for your needs
DETECTOR_HANDLE = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
# Note: If using a Kaggle model handle, ensure TF Hub compatibility or adjust loading.
# DETECTOR_HANDLE = "https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/TensorFlow2/fpnlite-320x320/1"

try:
    print(f"Loading detection model from: {DETECTOR_HANDLE}...")
    detector = hub.load(DETECTOR_HANDLE)
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load the detection model from {DETECTOR_HANDLE}")
    print(f"Error details: {e}")
    print("Ensure TensorFlow Hub can access the URL and the model format is compatible.")
    sys.exit("Model loading failed. Exiting.")


# --- Model Specific Settings ---
# IMPORTANT: Verify the class index for 'person' in your chosen model's dataset (e.g., COCO).
# It's often 1, but double-check the model documentation.
PERSON_CLASS_INDEX = 1
# Minimum confidence score for a detection to be considered valid
DETECTION_THRESHOLD = 0.3


# --- Fluvio Settings ---
FLUVIO_CROWD_TOPIC = "crowd-data" # Topic to send person counts to


# --- Flask Routes ---

# Route for the main page (upload form)
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page (index.html)."""
    return render_template('index.html')

# Route to handle video upload and processing
@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Handles video upload.
    For each frame: detects persons, draws boxes, sends density grid to Fluvio.
    Saves processed frames and renders the results page.
    """
    print("--- Request received for /upload ---")
    sys.stdout.flush() # Ensure prints appear immediately

    # --- Check if video file exists in the request ---
    if 'video' not in request.files:
        print("Upload Error: 'video' part not found in the request files.")
        return 'No video file part in the request', 400 # Bad Request

    video_file = request.files['video']

    # --- Check if a filename exists (means a file was selected) ---
    if video_file.filename == '':
        print("Upload Error: No file selected in the form.")
        return 'No selected video file', 400 # Bad Request

    print(f"Received video file: {video_file.filename}")

    # --- Initialize Fluvio Producer ---
    fluvio_producer = None # Start with no producer
    fluvio_client = None   # Initialize client variable
    try:
        print("Attempting to connect to Fluvio...")
        # Connect synchronously (adjust if your Fluvio setup needs specific options)
        fluvio_client = Fluvio.connect()
        print("Fluvio client connected.")
        # Get a producer for the specified topic
        fluvio_producer = fluvio_client.topic_producer(FLUVIO_CROWD_TOPIC)
        print(f"Fluvio producer ready for topic '{FLUVIO_CROWD_TOPIC}'.")
    except Exception as e:
        print(f"!!! FLUVIO WARNING: Could not connect or get producer: {e}")
        print("!!! Video processing will continue, but crowd data will NOT be sent to Fluvio.")
        # fluvio_producer remains None

    # --- Process the video file ---
    # Ensure the file object is valid before proceeding
    if video_file:
        # Construct full path to save the video temporarily
        filename = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        try:
            video_file.save(filename)
            print(f"Video temporarily saved to: {filename}")
        except Exception as e:
            print(f"Error saving video file '{filename}': {e}")
            # Clean up Fluvio connection if it exists before returning error
            if fluvio_client:
                del fluvio_client # Basic cleanup attempt
            return f"Error saving video file: {e}", 500 # Internal Server Error

        video_path = filename
        # --- Open the video file using OpenCV ---
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error opening video file with OpenCV: {video_path}")
            # Clean up Fluvio connection if it exists
            if fluvio_client:
                del fluvio_client
            return f"Error opening video file: {video_path}", 500 # Internal Server Error
        else:
            # Get video properties (optional, but good for info)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Video opened: {video_path} ({width}x{height} @ {fps:.2f} FPS)")


        processed_frame_filenames = [] # Store names of saved frames for the results page
        frame_count = 0

        # --- Process video frame by frame ---
        print("Starting frame processing loop...")
        while True:
            try:
                ret, frame = cap.read()
            except Exception as e:
                print(f"Error reading frame {frame_count} from video: {e}")
                break # Exit loop if reading fails

            # If 'ret' is False, we've reached the end of the video or had an error reading
            if not ret:
                if frame_count == 0:
                    print("Error: Could not read the first frame. Video might be corrupted or empty.")
                else:
                    print(f"End of video reached or cannot read next frame. Total frames processed: {frame_count}")
                break

            # --- Prepare frame for the ML model ---
            try:
                # Convert the frame from BGR (OpenCV default) to RGB (TF model format)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except cv2.error as e:
                print(f"Error converting frame {frame_count} to RGB: {e}. Skipping frame.")
                frame_count += 1
                continue # Skip this frame

            # Add a batch dimension (model expects a batch of images, even if it's just one)
            image_tensor = tf.expand_dims(rgb_frame, axis=0)

            # --- Run Person Detection ---
            try:
                start_time = time.time() # Optional: time the detection
                detections = detector(image_tensor)
                end_time = time.time()
                # print(f"Frame {frame_count} detection time: {end_time - start_time:.4f}s") # Uncomment for performance info
            except Exception as e:
                print(f"Error running ML model detection on frame {frame_count}: {e}. Skipping frame.")
                frame_count += 1
                continue # Skip this frame

            # --- Extract detection results (convert tensors to numpy arrays) ---
            try:
                # Using [0] index because our batch size is 1
                boxes = detections['detection_boxes'][0].numpy()
                classes = detections['detection_classes'][0].numpy().astype(int)
                scores = detections['detection_scores'][0].numpy()
                # num_detections = int(detections['num_detections'][0].numpy()) # May not exist in all models
            except KeyError as e:
                print(f"Error accessing detection results (KeyError: {e}). Check model output keys are correct. Skipping frame.")
                frame_count += 1
                continue # Skip frame if results format is unexpected
            except Exception as e:
                print(f"Error processing detection results for frame {frame_count}: {e}. Skipping frame.")
                frame_count += 1
                continue

            # --- Get current frame dimensions ---
            frame_height, frame_width, _ = frame.shape
            grid_rows = 10  # You can adjust the number of rows
            grid_cols = 10  # You can adjust the number of columns
            cell_height = frame_height // grid_rows
            cell_width = frame_width // grid_cols
            density_grid = [[0 for _ in range(grid_cols)] for _ in range(grid_rows)]

            # --- Populate the density grid ---
            for i in range(boxes.shape[0]):
                if scores[i] >= DETECTION_THRESHOLD and classes[i] == PERSON_CLASS_INDEX:
                    ymin, xmin, ymax, xmax = boxes[i]
                    xmin_abs = int(xmin * frame_width)
                    xmax_abs = int(xmax * frame_width)
                    ymin_abs = int(ymin * frame_height)
                    ymax_abs = int(ymax * frame_height)
                    center_x = (xmin_abs + xmax_abs) // 2
                    center_y = (ymin_abs + ymax_abs) // 2
                    row = center_y // cell_height
                    col = center_x // cell_width
                    if 0 <= row < grid_rows and 0 <= col < grid_cols:
                        density_grid[row][col] += 1

            # --- Send density grid to Fluvio ---
            if fluvio_producer:
                current_timestamp = int(time.time())
                fluvio_data = json.dumps({"timestamp": current_timestamp, "density_grid": density_grid})
                try:
                    fluvio_producer.send(b"crowd_key", fluvio_data.encode('utf-8'))
                    # print(f"Frame {frame_count}: Sent density grid to Fluvio: {fluvio_data}")
                except Exception as e:
                    print(f"!!! FLUVIO WARNING: Could not send density grid for frame {frame_count}: {e}")

            # --- Get current frame dimensions (needed for scaling boxes) ---
            # Do this *after* successfully reading the frame
            frame_height, frame_width, _ = frame.shape

            person_count_in_frame = 0 # Reset count for this frame

            # --- Iterate through all detections found in this frame ---
            # Use boxes.shape[0] as it represents the number of detections provided
            for i in range(boxes.shape[0]):
                score = scores[i]
                class_id = classes[i]

                # Check 1: Is the confidence score high enough?
                if score < DETECTION_THRESHOLD:
                    continue # Skip this detection if confidence is too low

                # Check 2: Is the detected object a person?
                if class_id == PERSON_CLASS_INDEX:
                    person_count_in_frame += 1 # Increment person count

                    # Get normalized box coordinates [ymin, xmin, ymax, xmax]
                    ymin, xmin, ymax, xmax = boxes[i]

                    # --- Scale coordinates to original frame size (absolute pixel values) ---
                    # Important: Multiply first, then convert to integer
                    xmin_abs = int(xmin * frame_width)
                    xmax_abs = int(xmax * frame_width)
                    ymin_abs = int(ymin * frame_height)
                    ymax_abs = int(ymax * frame_height)

                    # --- Draw the rectangle on the original BGR frame ---
                    color = (0, 255, 0) # Green color for the box (BGR format)
                    thickness = 2     # Thickness of the box lines
                    cv2.rectangle(frame, (xmin_abs, ymin_abs), (xmax_abs, ymax_abs), color, thickness)

                    # Optional: Add label with score
                    # label = f"Person: {score:.2f}"
                    # cv2.putText(frame, label, (xmin_abs, ymin_abs - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
######
            # # Draw red overlays for high-density areas
            # threshold_density = 3  # You can tweak this number
            # for row in range(grid_rows):
            #     for col in range(grid_cols):
            #         if density_grid[row][col] >= threshold_density:
            #             top_left = (col * cell_width, row * cell_height)
            #             bottom_right = ((col + 1) * cell_width, (row + 1) * cell_height)
            #             overlay_color = (0, 0, 255)  # Red in BGR
            #             alpha = 0.4  # Transparency of overlay
            #             sub_img = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            #             red_overlay = sub_img.copy()
            #             red_overlay[:] = overlay_color
            #             cv2.addWeighted(red_overlay, alpha, sub_img, 1 - alpha, 0, sub_img)



######
            # --- Save the processed frame (image with boxes drawn) ---
            output_filename_base = f"frame_{frame_count}.jpg"
            output_filepath = os.path.join(PROCESSED_FRAMES_FOLDER, output_filename_base)
            try:
                # Save the frame (which now has rectangles drawn on it)
                cv2.imwrite(output_filepath, frame)
                processed_frame_filenames.append(output_filename_base) # Add filename for display on results page
            except Exception as e:
                print(f"Error saving processed frame {frame_count} to {output_filepath}: {e}")

            frame_count += 1 # Increment frame counter

        # --- End of video processing loop ---
        print("Finished processing all frames.")

        # --- Clean up resources ---
        cap.release() # Release the video file resource
        print("Video capture released.")
        if fluvio_client:
            # If Fluvio client has a specific close/disconnect method, call it here
            # For now, just deleting the reference as basic cleanup
            del fluvio_producer # Delete producer first if it exists
            del fluvio_client
            print("Fluvio client reference deleted.")

        # Optionally remove the original uploaded file after processing
        try:
            os.remove(filename)
            print(f"Removed temporary uploaded file: {filename}")
        except OSError as e:
            print(f"Error removing uploaded file {filename}: {e}")

        # --- Render the results page, passing the list of processed frame filenames ---
        print(f"Rendering results page with {len(processed_frame_filenames)} processed frames.")
        return render_template('results.html', processed_frames=processed_frame_filenames)

    # --- Fallback case (should ideally not be reached if checks above work) ---
    print("Upload Error: Reached end of function unexpectedly (video_file condition failed?).")
    # Clean up Fluvio connection if it somehow still exists
    if fluvio_client:
        del fluvio_client
    return 'An unexpected error occurred processing the video file.', 500

# --- Run the Flask Application ---
if __name__ == '__main__':
    # debug=True enables auto-reloading on code changes and provides detailed error pages
    # host='0.0.0.0' makes the server accessible from other devices on your network
    #           (use '127.0.0.1' or remove host parameter for local access only)
    print("--- Starting Flask Development Server ---")
    print("Access the application via http://127.0.0.1:5000 or http://<your-ip>:5000")
    # Set use_reloader=False if you encounter issues with model loading on reload
    app.run(debug=True, host='0.0.0.0', port=5000)


'''
























# # Required Imports
# from flask import Flask, render_template, request
# import os
# import cv2 # OpenCV for video/image processing
# import tensorflow as tf # TensorFlow for the ML model
# import tensorflow_hub as hub # For loading models from TensorFlow Hub
# import sys # For system-level operations like printing/exiting
# import time # To get timestamps for Fluvio data
# from fluvio import Fluvio # Fluvio Python client

# # --- Flask Application Setup ---
# app = Flask(__name__)

# # --- Configuration for Folders ---
# UPLOAD_FOLDER = 'uploads' # Folder to temporarily store uploaded videos
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# STATIC_FOLDER = 'static' # Standard folder for static files (CSS, JS, images)
# PROCESSED_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'processed_frames') # Subfolder for output frames
# if not os.path.exists(PROCESSED_FRAMES_FOLDER):
#     os.makedirs(PROCESSED_FRAMES_FOLDER)

# # --- Load Machine Learning Model ---
# # Load the object detection model from TensorFlow Hub
# # Using SSD MobileNet V2 FPNLite 320x320 - choose a model suitable for your needs
# DETECTOR_HANDLE = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
# # Note: If using a Kaggle model handle, ensure TF Hub compatibility or adjust loading.
# # DETECTOR_HANDLE = "https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/TensorFlow2/fpnlite-320x320/1"

# try:
#     print(f"Loading detection model from: {DETECTOR_HANDLE}...")
#     detector = hub.load(DETECTOR_HANDLE)
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"FATAL ERROR: Could not load the detection model from {DETECTOR_HANDLE}")
#     print(f"Error details: {e}")
#     print("Ensure TensorFlow Hub can access the URL and the model format is compatible.")
#     sys.exit("Model loading failed. Exiting.")


# # --- Model Specific Settings ---
# # IMPORTANT: Verify the class index for 'person' in your chosen model's dataset (e.g., COCO).
# # It's often 1, but double-check the model documentation.
# PERSON_CLASS_INDEX = 1
# # Minimum confidence score for a detection to be considered valid
# DETECTION_THRESHOLD = 0.3


# # --- Fluvio Settings ---
# FLUVIO_CROWD_TOPIC = "crowd-data" # Topic to send person counts to


# # --- Flask Routes ---

# # Route for the main page (upload form)
# @app.route('/', methods=['GET'])
# def index():
#     """Renders the main upload page (index.html)."""
#     return render_template('index.html')

# # Route to handle video upload and processing
# @app.route('/upload', methods=['POST'])
# def upload_video():
#     """
#     Handles video upload.
#     For each frame: detects persons, draws boxes, sends count to Fluvio.
#     Saves processed frames and renders the results page.
#     """
#     print("--- Request received for /upload ---")
#     sys.stdout.flush() # Ensure prints appear immediately

#     # --- Check if video file exists in the request ---
#     if 'video' not in request.files:
#         print("Upload Error: 'video' part not found in the request files.")
#         return 'No video file part in the request', 400 # Bad Request

#     video_file = request.files['video']

#     # --- Check if a filename exists (means a file was selected) ---
#     if video_file.filename == '':
#         print("Upload Error: No file selected in the form.")
#         return 'No selected video file', 400 # Bad Request

#     print(f"Received video file: {video_file.filename}")

#     # --- Initialize Fluvio Producer ---
#     fluvio_producer = None # Start with no producer
#     fluvio_client = None   # Initialize client variable
#     try:
#         print("Attempting to connect to Fluvio...")
#         # Connect synchronously (adjust if your Fluvio setup needs specific options)
#         fluvio_client = Fluvio.connect()
#         print("Fluvio client connected.")
#         # Get a producer for the specified topic
#         fluvio_producer = fluvio_client.topic_producer(FLUVIO_CROWD_TOPIC)
#         print(f"Fluvio producer ready for topic '{FLUVIO_CROWD_TOPIC}'.")
#     except Exception as e:
#         print(f"!!! FLUVIO WARNING: Could not connect or get producer: {e}")
#         print("!!! Video processing will continue, but crowd data will NOT be sent to Fluvio.")
#         # fluvio_producer remains None

#     # --- Process the video file ---
#     # Ensure the file object is valid before proceeding
#     if video_file:
#         # Construct full path to save the video temporarily
#         filename = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
#         try:
#             video_file.save(filename)
#             print(f"Video temporarily saved to: {filename}")
#         except Exception as e:
#             print(f"Error saving video file '{filename}': {e}")
#             # Clean up Fluvio connection if it exists before returning error
#             if fluvio_client:
#                  del fluvio_client # Basic cleanup attempt
#             return f"Error saving video file: {e}", 500 # Internal Server Error

#         video_path = filename
#         # --- Open the video file using OpenCV ---
#         cap = cv2.VideoCapture(video_path)

#         if not cap.isOpened():
#             print(f"Error opening video file with OpenCV: {video_path}")
#             # Clean up Fluvio connection if it exists
#             if fluvio_client:
#                  del fluvio_client
#             return f"Error opening video file: {video_path}", 500 # Internal Server Error
#         else:
#             # Get video properties (optional, but good for info)
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             print(f"Video opened: {video_path} ({width}x{height} @ {fps:.2f} FPS)")


#         processed_frame_filenames = [] # Store names of saved frames for the results page
#         frame_count = 0

#         # --- Process video frame by frame ---
#         print("Starting frame processing loop...")
#         while True:
#             try:
#                 ret, frame = cap.read()
#             except Exception as e:
#                 print(f"Error reading frame {frame_count} from video: {e}")
#                 break # Exit loop if reading fails

#             # If 'ret' is False, we've reached the end of the video or had an error reading
#             if not ret:
#                 if frame_count == 0:
#                     print("Error: Could not read the first frame. Video might be corrupted or empty.")
#                 else:
#                     print(f"End of video reached or cannot read next frame. Total frames processed: {frame_count}")
#                 break

#             # --- Prepare frame for the ML model ---
#             try:
#                 # Convert the frame from BGR (OpenCV default) to RGB (TF model format)
#                 rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             except cv2.error as e:
#                 print(f"Error converting frame {frame_count} to RGB: {e}. Skipping frame.")
#                 frame_count += 1
#                 continue # Skip this frame

#             # Add a batch dimension (model expects a batch of images, even if it's just one)
#             image_tensor = tf.expand_dims(rgb_frame, axis=0)

#             # --- Run Person Detection ---
#             try:
#                 start_time = time.time() # Optional: time the detection
#                 detections = detector(image_tensor)
#                 end_time = time.time()
#                 # print(f"Frame {frame_count} detection time: {end_time - start_time:.4f}s") # Uncomment for performance info
#             except Exception as e:
#                 print(f"Error running ML model detection on frame {frame_count}: {e}. Skipping frame.")
#                 frame_count += 1
#                 continue # Skip this frame

#             # --- Extract detection results (convert tensors to numpy arrays) ---
#             try:
#                 # Using [0] index because our batch size is 1
#                 boxes = detections['detection_boxes'][0].numpy()
#                 classes = detections['detection_classes'][0].numpy().astype(int)
#                 scores = detections['detection_scores'][0].numpy()
#                 # num_detections = int(detections['num_detections'][0].numpy()) # May not exist in all models
#             except KeyError as e:
#                 print(f"Error accessing detection results (KeyError: {e}). Check model output keys are correct. Skipping frame.")
#                 frame_count += 1
#                 continue # Skip frame if results format is unexpected
#             except Exception as e:
#                 print(f"Error processing detection results for frame {frame_count}: {e}. Skipping frame.")
#                 frame_count += 1
#                 continue

#             # --- Get current frame dimensions (needed for scaling boxes) ---
#             # Do this *after* successfully reading the frame
#             frame_height, frame_width, _ = frame.shape

#             person_count_in_frame = 0 # Reset count for this frame

#             # --- Iterate through all detections found in this frame ---
#             # Use boxes.shape[0] as it represents the number of detections provided
#             for i in range(boxes.shape[0]):
#                 score = scores[i]
#                 class_id = classes[i]

#                 # Check 1: Is the confidence score high enough?
#                 if score < DETECTION_THRESHOLD:
#                     continue # Skip this detection if confidence is too low

#                 # Check 2: Is the detected object a person?
#                 if class_id == PERSON_CLASS_INDEX:
#                     person_count_in_frame += 1 # Increment person count

#                     # Get normalized box coordinates [ymin, xmin, ymax, xmax]
#                     ymin, xmin, ymax, xmax = boxes[i]

#                     # --- Scale coordinates to original frame size (absolute pixel values) ---
#                     # Important: Multiply first, then convert to integer
#                     xmin_abs = int(xmin * frame_width)
#                     xmax_abs = int(xmax * frame_width)
#                     ymin_abs = int(ymin * frame_height)
#                     ymax_abs = int(ymax * frame_height)

#                     # --- Draw the rectangle on the original BGR frame ---
#                     color = (0, 255, 0) # Green color for the box (BGR format)
#                     thickness = 2       # Thickness of the box lines
#                     cv2.rectangle(frame, (xmin_abs, ymin_abs), (xmax_abs, ymax_abs), color, thickness)

#                     # Optional: Add label with score
#                     # label = f"Person: {score:.2f}"
#                     # cv2.putText(frame, label, (xmin_abs, ymin_abs - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

#             # --- Send data to Fluvio ---
#             if fluvio_producer: # Only attempt if producer was successfully created
#                 try:
#                     current_timestamp = int(time.time()) # Get current epoch timestamp
#                     # Format data as: timestamp,count
#                     fluvio_data = f"{current_timestamp},{person_count_in_frame}"
#                     # Send data using a simple key 'crowd_key' (can be anything) and the formatted data string
#                     # Data must be encoded to bytes
#                     fluvio_producer.send(b"crowd_key", fluvio_data.encode('utf-8'))
#                     # print(f"Frame {frame_count}: Found {person_count_in_frame} persons. Sent to Fluvio: {fluvio_data}") # Verbose log
#                 except Exception as e:
#                     print(f"!!! FLUVIO WARNING: Could not send data for frame {frame_count}: {e}")
#                     # Optional: Implement logic to handle repeated send failures (e.g., disable producer)
#                     # fluvio_producer = None # Example: Stop trying if send fails

#             else: # If Fluvio producer is None (connection failed earlier)
#                 print(f"Frame {frame_count}: Found {person_count_in_frame} persons. (Fluvio producer not available)")

#             # --- Save the processed frame (image with boxes drawn) ---
#             output_filename_base = f"frame_{frame_count}.jpg"
#             output_filepath = os.path.join(PROCESSED_FRAMES_FOLDER, output_filename_base)
#             try:
#                 # Save the frame (which now has rectangles drawn on it)
#                 cv2.imwrite(output_filepath, frame)
#                 processed_frame_filenames.append(output_filename_base) # Add filename for display on results page
#             except Exception as e:
#                 print(f"Error saving processed frame {frame_count} to {output_filepath}: {e}")

#             frame_count += 1 # Increment frame counter

#         # --- End of video processing loop ---
#         print("Finished processing all frames.")

#         # --- Clean up resources ---
#         cap.release() # Release the video file resource
#         print("Video capture released.")
#         if fluvio_client:
#             # If Fluvio client has a specific close/disconnect method, call it here
#             # For now, just deleting the reference as basic cleanup
#              del fluvio_producer # Delete producer first if it exists
#              del fluvio_client
#              print("Fluvio client reference deleted.")

#         # Optionally remove the original uploaded file after processing
#         try:
#             os.remove(filename)
#             print(f"Removed temporary uploaded file: {filename}")
#         except OSError as e:
#             print(f"Error removing uploaded file {filename}: {e}")

#         # --- Render the results page, passing the list of processed frame filenames ---
#         print(f"Rendering results page with {len(processed_frame_filenames)} processed frames.")
#         return render_template('results.html', processed_frames=processed_frame_filenames)

#     # --- Fallback case (should ideally not be reached if checks above work) ---
#     print("Upload Error: Reached end of function unexpectedly (video_file condition failed?).")
#     # Clean up Fluvio connection if it somehow still exists
#     if fluvio_client:
#          del fluvio_client
#     return 'An unexpected error occurred processing the video file.', 500

# # --- Run the Flask Application ---
# if __name__ == '__main__':
#     # debug=True enables auto-reloading on code changes and provides detailed error pages
#     # host='0.0.0.0' makes the server accessible from other devices on your network
#     #          (use '127.0.0.1' or remove host parameter for local access only)
#     print("--- Starting Flask Development Server ---")
#     print("Access the application via http://127.0.0.1:5000 or http://<your-ip>:5000")
#     # Set use_reloader=False if you encounter issues with model loading on reload
#     app.run(debug=True, host='0.0.0.0', port=5000)