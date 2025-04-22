import json
from fluvio import Fluvio, Offset # Import Offset class
import time
import sys # For flushing output

# --- Configuration ---
FLUVIO_CROWD_TOPIC = "crowd-data"
FLUVIO_PARTITION = 0 # Default partition is usually 0

# --- Density Analysis Settings (Copied from app.py for reference/potential future use) ---
# You could re-implement analysis here if desired, but for now, we just print.
# GRID_ROWS = 8 # Match app.py if needed
# GRID_COLS = 8 # Match app.py if needed

# --- Placeholder Analysis Function (Currently just prints received data) ---
def analyze_received_data(data):
    """
    Analyzes the data dictionary received from Fluvio.
    (Currently just prints, but could be expanded later)
    """
    try:
        timestamp = data.get("timestamp", "N/A")
        frame_num = data.get("frame", "N/A")
        density_grid = data.get("density_grid", None)
        confirmed_persons = data.get("confirmed_persons", "N/A")
        frame_status = data.get("frame_status", "N/A")
        high_cells = data.get("high_density_cells", "N/A")
        critical_cells = data.get("critical_density_cells", "N/A")

        print(f"--- Received Frame {frame_num} (TS: {timestamp}) ---")
        print(f"  Status: {frame_status}, Persons: {confirmed_persons}, High Cells: {high_cells}, Crit Cells: {critical_cells}")
        if density_grid is not None:
            total_people = sum(sum(row) for row in density_grid if isinstance(row, list))
            print(f"  Total people in grid (calculated): {total_people}")
            # print(f"  Density Grid: {density_grid}") # Uncomment for full grid details
        else:
             print(f"  Density Grid: Not provided")
        print("-" * 20)

    except Exception as e:
        print(f"Error analyzing received data for frame {frame_num}: {e}")
        print(f"Raw data: {data}")
    finally:
        sys.stdout.flush() # Ensure prints appear immediately

def main():
    fluvio_client = None
    while True: # Keep trying to connect
        try:
            print("Attempting to connect to Fluvio...")
            sys.stdout.flush()
            # Ensure your Fluvio cluster is running locally or specify connection details
            fluvio_client = Fluvio.connect() # Assumes local default connection
            print("Fluvio client connected.")
            sys.stdout.flush()

            consumer = fluvio_client.partition_consumer(FLUVIO_CROWD_TOPIC, partition=FLUVIO_PARTITION)
            print(f"Fluvio consumer ready for topic '{FLUVIO_CROWD_TOPIC}' on partition {FLUVIO_PARTITION}.")
            sys.stdout.flush()

            # --- Start reading from the beginning of the topic partition ---
            # Use Offset.from_end(0) to get only new messages after consumer starts.
            # Use Offset.absolute(N) or Offset.from_beginning(N) to start N records from start.
            stream = consumer.stream(Offset.from_end(0)) # Start from end (only new messages)
            # stream = consumer.stream(Offset.from_beginning(0)) # Use this to process all historical data on topic
            print(f"Listening for new data on topic '{FLUVIO_CROWD_TOPIC}'...")
            sys.stdout.flush()

            # --- Process Records ---
            for record in stream:
                try:
                    record_value_str = record.value_string()
                    data = json.loads(record_value_str)
                    analyze_received_data(data) # Call the analysis/printing function

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    try: raw_value = record.value_string()
                    except: raw_value = "<Could not decode raw value>"
                    print(f"Problematic record value: {raw_value}")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"Error processing record: {e}")
                    try: raw_value = record.value_string()
                    except: raw_value = "<Could not decode raw value>"
                    print(f"Raw record value: {raw_value}")
                    sys.stdout.flush()

        except Exception as e:
            # This catches errors during connection, consumer creation, or stream iteration setup
            print(f"Error setting up Fluvio connection or stream: {e}")
            print("Check if the Fluvio cluster is running and the topic exists.")
            print("Retrying in 10 seconds...")
            sys.stdout.flush()
            if fluvio_client:
               try:
                   # Attempt cleanup, although Fluvio client cleanup might not be explicitly needed
                   del fluvio_client
                   fluvio_client = None
               except: pass
            time.sleep(10) # Wait longer before retrying connection

    # Cleanup (unlikely to be reached in this loop)
    print("Exiting main loop (unexpected).")
    if fluvio_client:
        del fluvio_client
        print("Fluvio client reference deleted.")

if __name__ == "__main__":
    main()
