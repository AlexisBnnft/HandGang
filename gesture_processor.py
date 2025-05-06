import numpy as np
from itertools import combinations  # For calculating pairwise combinations

NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_FINGER_MCP = 9

# Original normalize_landmarks function is commented out
"""
def normalize_landmarks(hand_landmarks_object):
    Normalizes hand landmarks to be invariant to position and scale.
    Returns a 1D feature vector (flattened normalized coordinates) or None.
    
    if not hand_landmarks_object:
        return None

    landmarks = []
    for lm in hand_landmarks_object.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    landmarks = np.array(landmarks)  # Shape (21, 3)

    if landmarks.shape != (NUM_LANDMARKS, 3):
        return None  # Should not happen with mediapipe results

    # 1. Translate so wrist (landmark 0) is at the origin
    wrist_coords = landmarks[WRIST]
    translated_landmarks = landmarks - wrist_coords

    # 2. Calculate scale factor (e.g., distance between wrist and middle finger MCP)
    # Ensure wrist is at origin for this calculation
    middle_mcp_coords_relative = translated_landmarks[MIDDLE_FINGER_MCP]
    scale_factor = np.linalg.norm(middle_mcp_coords_relative)

    # Alternative: Use max distance from origin
    # scale_factor = np.max(np.linalg.norm(translated_landmarks, axis=1))

    if scale_factor < 1e-6:  # Avoid division by zero or near-zero
        # Fallback or indicate degenerate case. Using L2 norm of all translated points?
        scale_factor = np.linalg.norm(translated_landmarks)
        if scale_factor < 1e-6:
            print("Warning: Could not determine valid scale factor for normalization.")
            return None  # Cannot normalize

    # 3. Normalize by scale factor
    normalized_landmarks = translated_landmarks / scale_factor

    # 4. Flatten into a 1D feature vector
    feature_vector = normalized_landmarks.flatten()  # Shape (63,)

    return feature_vector
"""


# --- Feature Extraction using Pairwise Distances ---
def extract_distance_features(hand_landmarks_object):
    """
    Extracts features based on normalized pairwise distances between landmarks.
    Returns a 1D feature vector (210 distances) or None.
    """
    if not hand_landmarks_object:
        return None

    # 1. Extract absolute 3D coordinates
    landmarks_abs = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks_object.landmark]
    )

    if landmarks_abs.shape != (NUM_LANDMARKS, 3):
        return None  # Invalid input

    # 2. Calculate scale factor (distance between wrist and middle finger MCP)
    scale_distance = np.linalg.norm(
        landmarks_abs[MIDDLE_FINGER_MCP] - landmarks_abs[WRIST]
    )

    if scale_distance < 1e-6:
        print("Warning: Scale distance is too small for normalization.")
        return None  # Avoid division by zero

    # 3. Calculate all pairwise distances (21 * 20 / 2 = 210 distances)
    pairwise_distances = []
    # Iterate through all unique pairs of landmark indices
    for i, j in combinations(range(NUM_LANDMARKS), 2):
        distance = np.linalg.norm(landmarks_abs[i] - landmarks_abs[j])
        pairwise_distances.append(distance)

    pairwise_distances = np.array(pairwise_distances)

    # 4. Normalize distances by the scale factor
    normalized_distances = pairwise_distances / scale_distance

    return normalized_distances  # Feature vector of shape (210,)


def calculate_similarity(feature_vector1, feature_vector2):
    """
    Calculates similarity based on Euclidean distance between normalized feature vectors.
    Lower distance means higher similarity. Returns float distance or infinity if inputs invalid.
    """
    if (
        feature_vector1 is None
        or feature_vector2 is None
        or feature_vector1.shape != feature_vector2.shape
    ):
        return float("inf")  # Return infinity for invalid comparison

    distance = np.linalg.norm(feature_vector1 - feature_vector2)
    return distance
