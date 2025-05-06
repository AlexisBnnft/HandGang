import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        print("HandTracker initialized.")

    def process_frame(self, frame_bgr):
        """Processes a BGR frame to find hand landmarks."""
        # Flip the frame horizontally for a later selfie-view display
        frame_flipped = cv2.flip(frame_bgr, 1)
        # Convert the BGR image to RGB.
        image_rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True  # Make writeable again if needed later

        # Return the flipped frame (for drawing) and the results
        return frame_flipped, results

    def draw_landmarks(self, frame, hand_landmarks):
        """Draws landmarks and connections on the frame."""
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style(),
        )

    def close(self):
        """Releases MediaPipe resources."""
        self.hands.close()
        print("HandTracker closed.")
