import json
import os
import numpy as np


class GestureStore:
    def __init__(self, filepath="gestures.json"):
        self.filepath = filepath
        self.gestures = {}  # Dictionary to store {gesture_name: feature_vector_list}
        self.load_gestures()
        print(
            f"GestureStore initialized. Loaded {len(self.gestures)} gestures from {self.filepath}"
        )

    def load_gestures(self):
        """Loads gestures from the JSON file."""
        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, "r") as f:
                    self.gestures = json.load(f)
                    # Optional: Basic validation could go here
            else:
                print(
                    f"Gesture file '{self.filepath}' not found. Starting with empty store."
                )
                self.gestures = {}
        except (json.JSONDecodeError, IOError) as e:
            print(
                f"Error loading gestures from {self.filepath}: {e}. Starting with empty store."
            )
            self.gestures = {}  # Reset to empty on error

    def save_gestures(self):
        """Saves the current gestures to the JSON file."""
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.gestures, f, indent=4)
        except IOError as e:
            print(f"Error saving gestures to {self.filepath}: {e}")

    def add_gesture(self, name, feature_vector):
        """Adds a new gesture or updates an existing one and saves."""
        if name and feature_vector is not None:
            self.gestures[name] = feature_vector.tolist()  # Store as list
            self.save_gestures()
            print(f"Gesture '{name}' saved.")
        else:
            print(
                "Error: Cannot save gesture with empty name or invalid feature vector."
            )

    def get_gestures(self):
        """Returns the dictionary of loaded gestures."""
        return self.gestures

    def get_gesture_names(self):
        """Returns a list of saved gesture names."""
        return list(self.gestures.keys())
