# Hand Gesture Recognition System

A Python application that recognizes hand gestures using webcam input. The system uses MediaPipe for hand tracking and a custom distance-based feature extraction method to identify and classify gestures.

## Features

- Real-time hand tracking and gesture recognition
- Distance-based feature extraction for reliable gesture classification
- Ability to record and save new gestures
- Visualization of hand landmarks and detection results

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- MediaPipe

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd hand-gesture-recognition
```

2. Install required packages:
```
pip install -r requirements.txt
```

## Usage

Run the main application:
```
python main.py
```

### Controls
- **R**: Record current hand pose as a new gesture
- **Q**: Quit the application

## Project Structure

- `main.py`: Main application logic and UI
- `hand_tracker.py`: Hand tracking module using MediaPipe
- `gesture_processor.py`: Feature extraction and similarity calculation
- `gesture_store.py`: Storage and retrieval of saved gestures

## License

[MIT License](LICENSE) 