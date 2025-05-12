import cv2
import numpy as np
import time
import random
import string
#import pygame
import os
import subprocess
from scipy.io import wavfile

# Custom modules
from hand_tracker import HandTracker
from gesture_store import GestureStore
from gesture_processor import extract_distance_features, calculate_similarity
from actions import ActionManager

print("Starting application...")

# --- Constants ---
WEBCAM_INDEX = 0
SIMILARITY_THRESHOLD = 2  # Adjust this based on testing (lower = stricter match)
DISPLAY_MSG_TIME = 2  # Seconds to display status messages

# --- Initialization ---
action_manager = ActionManager()
hand_tracker = HandTracker(max_hands=2)  # Track both hands
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
current_features = {
    "Left": None,
    "Right": None
}
last_validated_gestures = {
    "Left": None,
    "Right": None
}

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
        frame_processed, results = hand_tracker.process_frame(frame)

        best_matches = {
            "Left": "No Hand Detected",
            "Right": "No Hand Detected"
        }
        current_features = {"Left": None, "Right": None}  # Reset features for this frame

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_type in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                hand_type = hand_type.classification[0].label
                
                # Draw landmarks on the frame with appropriate style
                hand_tracker.draw_landmarks(frame_processed, hand_landmarks, hand_type)

                # Extract features for this hand
                current_features[hand_type] = extract_distance_features(hand_landmarks)

                if current_features[hand_type] is not None:
                    # --- Classification ---
                    saved_gestures = gesture_store.get_gestures()
                    min_distance = SIMILARITY_THRESHOLD
                    found_match = False

                    for name, saved_feature_list in saved_gestures.items():
                        saved_feature_vector = np.array(saved_feature_list)
                        distance = calculate_similarity(
                            current_features[hand_type], saved_feature_vector
                        )

                        if distance < min_distance:
                            min_distance = distance
                            best_matches[hand_type] = name
                            found_match = True

                    if not found_match:
                        best_matches[hand_type] = "Unknown Gesture"
                else:
                    best_matches[hand_type] = "Feature Extraction Failed"

        # --- Execute Actions ---
        right_gesture = best_matches["Right"]
        left_gesture = best_matches["Left"]
        action_manager.execute_action(right_gesture, left_gesture)

        # --- Handle User Input ---
        key = cv2.waitKey(5) & 0xFF

        if key == ord("q"):
            print("Quit key pressed.")
            break

        elif key == ord("r"):
            # Record gesture for the hand that is detected
            for hand_type in ["Left", "Right"]:
                if current_features[hand_type] is not None:
                    random_name = f"{hand_type}_Sign_{random.randint(100, 999)}"
                    while random_name in gesture_store.get_gesture_names():
                        random_name = f"{hand_type}_Sign_{random.randint(100, 999)}"

                    gesture_store.add_gesture(random_name, current_features[hand_type])
                    status_message = f"Saved {hand_type} hand: {random_name}"
                    message_display_end_time = time.time() + DISPLAY_MSG_TIME
                    print(f"Recorded {hand_type} hand gesture as '{random_name}'")

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

        # Display Classification Results for both hands
        label_y = 60
        for hand_type in ["Left", "Right"]:
            color = (0, 0, 255) if hand_type == "Left" else (0, 255, 0)
            cv2.putText(
                frame_processed,
                f"{hand_type} Hand: {best_matches[hand_type]}",
                (10, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
            label_y += 40

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
    action_manager.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    hand_tracker.close()
    print("Application closed.")
