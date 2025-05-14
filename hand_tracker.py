import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Styles personnalisés pour les mains gauche et droite
        self.left_hand_style = self.mp_drawing_styles.DrawingSpec(
            color=(0, 0, 255),  # Rouge pour la main gauche
            thickness=2,
            circle_radius=2
        )
        self.right_hand_style = self.mp_drawing_styles.DrawingSpec(
            color=(0, 255, 0),  # Vert pour la main droite
            thickness=2,
            circle_radius=2
        )
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

    def draw_landmarks(self, frame, hand_landmarks, hand_type):
        """Draws landmarks and connections on the frame with different styles for each hand."""
        style = self.left_hand_style if hand_type == "Left" else self.right_hand_style
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            style,
            self.mp_drawing_styles.get_default_hand_connections_style(),
        )

    def close(self):
        """Releases MediaPipe resources."""
        self.hands.close()
        print("HandTracker closed.")
