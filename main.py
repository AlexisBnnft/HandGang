import cv2
import numpy as np
import time
import random
import string

# Custom modules
from hand_tracker import HandTracker
from gesture_store import GestureStore
from gesture_processor import extract_distance_features, calculate_similarity

# --- Constants ---
WEBCAM_INDEX = 0
SIMILARITY_THRESHOLD = 1.5  # Adjust this based on testing (lower = stricter match)
DISPLAY_MSG_TIME = 2  # Seconds to display status messages

# --- Initialization ---
hand_tracker = HandTracker(max_hands=1)  # Track only one hand
gesture_store = GestureStore(filepath="gestures.json")
cap = cv2.VideoCapture(WEBCAM_INDEX)

if not cap.isOpened():
    print(f"Error: Could not open webcam index {WEBCAM_INDEX}.")
    exit()

print("Starting main application...")
print("Press 'R' when hand pose is stable to record a new gesture.")
print("Press 'Q' to quit.")

# --- State Variables ---
prev_frame_time = 0
status_message = ""
message_display_end_time = 0
current_features = (
    None  # Store the normalized features of the hand in the current frame
)

# --- Main Loop ---
try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # --- FPS Calculation ---
        new_frame_time = time.time()
        fps = (
            1 / (new_frame_time - prev_frame_time)
            if (new_frame_time - prev_frame_time) > 0
            else 0
        )
        prev_frame_time = new_frame_time

        # --- Hand Tracking & Processing ---
        frame_processed, results = hand_tracker.process_frame(
            frame
        )  # Gets flipped frame and results

        best_match_name = "No Hand Detected"
        current_features = None  # Reset features for this frame

        if results.multi_hand_landmarks:
            # Get landmarks for the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks on the frame
            hand_tracker.draw_landmarks(frame_processed, hand_landmarks)

            # Use the new function for feature extraction
            current_features = extract_distance_features(hand_landmarks)

            if current_features is not None:
                # --- Classification ---
                saved_gestures = gesture_store.get_gestures()
                min_distance = SIMILARITY_THRESHOLD
                found_match = False

                for name, saved_feature_list in saved_gestures.items():
                    # Convert saved list back to numpy array for comparison
                    saved_feature_vector = np.array(saved_feature_list)

                    # Calculate similarity
                    distance = calculate_similarity(
                        current_features, saved_feature_vector
                    )

                    if distance < min_distance:
                        min_distance = distance
                        best_match_name = name
                        found_match = True

                if not found_match:
                    best_match_name = "Unknown Gesture"

            else:
                best_match_name = "Feature Extraction Failed"

        # --- Handle User Input ---
        key = cv2.waitKey(5) & 0xFF

        if key == ord("q"):
            print("Quit key pressed.")
            break

        elif key == ord("r"):
            if current_features is not None:
                # Generate a simple random name
                random_name = f"Sign_{random.randint(100, 999)}"
                while (
                    random_name in gesture_store.get_gesture_names()
                ):  # Ensure unique name
                    random_name = f"Sign_{random.randint(100, 999)}"

                gesture_store.add_gesture(random_name, current_features)
                status_message = f"Saved: {random_name}"
                message_display_end_time = time.time() + DISPLAY_MSG_TIME
                print(f"Recorded gesture as '{random_name}'")
            else:
                status_message = "Cannot record: No hand detected or features invalid."
                message_display_end_time = time.time() + DISPLAY_MSG_TIME
                print("Recording failed: Hand not detected clearly.")

        # --- Display Information ---
        # Display FPS
        cv2.putText(
            frame_processed,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Display Classification Result
        label_y = 60
        cv2.putText(
            frame_processed,
            f"Detected: {best_match_name}",
            (10, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        # Display Status Message (if any)
        if status_message and time.time() < message_display_end_time:
            cv2.putText(
                frame_processed,
                status_message,
                (10, frame_processed.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        elif time.time() >= message_display_end_time:
            status_message = ""  # Clear expired message

        # Display Instructions
        cv2.putText(
            frame_processed,
            "R: Record Gesture | Q: Quit",
            (10, frame_processed.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        # --- Show Frame ---
        cv2.imshow("Gang Sign Detector", frame_processed)


except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    # --- Cleanup ---
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    hand_tracker.close()
    print("Application closed.")
