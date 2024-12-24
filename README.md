# Archery-Target-Recognition

This project is designed to recognize an archery target and trigger a "Fire" command once the target is within the designated target zone. It uses computer vision techniques to detect specific colors in a live camera feed, processes the image to highlight the target, and checks if the target is within the predefined target zone.

## Overview
- **Functionality:** The program captures video from the webcam, applies color filtering to identify specific colors (such as the target zone), and displays the target detection in real-time. It also triggers the "Fire" command once the target is in the predefined zone.
- **Objective:** Recognize a target and determine if it enters a predefined target zone, signaling when to "Fire."

## Key Learning Areas

### 1. Video Capture with OpenCV
- **Webcam Integration:** Captures live video from the webcam using OpenCV's cv2.VideoCapture() method.
- **Error Handling:** Ensures the webcam is accessible and handles any errors that may occur when capturing frames.

### 2. Color Filtering with HSV
- **HSV Color Space:** Converts video frames from BGR to RGB and then to HSV (Hue, Saturation, Value) for easier color-based filtering.
- **Masking:** Creates binary masks that isolate colors within the specified HSV range, used for target detection.

### 3. Object Detection and Centroid Calculation
- **Contour Detection:** Identifies contours of the target area using the Canny edge detection algorithm.
- **Centroid Calculation:** Calculates the centroid of the detected target and checks if it's within the predefined target zone.

### 4. Real-Time Visualization
- **Displaying Results:** Displays the live webcam feed along with the visual feedback of detected colors and the target zone using OpenCV's cv2.imshow().

## Languages and Tools Used

### Python
- **OpenCV:** For video capture, image processing, and real-time object detection.
- **NumPy:** For handling arrays and performing numerical operations.
- **Tkinter:** For fetching screen resolution to organize windows in a grid layout.

## Installation and Usage

### Requirements
- **Python 3.x** to run the script.
- **OpenCV library**: To install OpenCV, run the following:
```bash
pip install opencv-python
```

## Instructions
1. Clone the repository
  ```bash
  git clone https://github.com/carlosagalicia/Archery-Target-Recognition.git
  ```

2. Run the script
  ```bash
  python archery.py
  ```
3. Make sure your webcam is connected and functional.

## Operation
- The program captures video from your webcam and processes each frame.
- It applies color-based masking to isolate the target zone.
- The system calculates the contours of the detected areas and identifies the centroid of the target.
- Once the centroid of the detected target enters the defined target zone, the message "Fire" is displayed on the screen.
- The program shows the live webcam feed along with visual feedback for each detected color mask.

## Usage
- Press 'q' to quit the program.
- You can adjust the parameters for color masking if needed (e.g., for different lighting conditions or target color variations).

## Visual Representation
<table>
<tr>
  <td width="50%">
    <h3 align="center">Color recognition</h3>
    <div align="center">
      <img src="https://github.com/user-attachments/assets/d08e9462-ce7e-4e39-8d27-9d3827ba5b7d" >
    </div>
  </td>
</tr>
</table>

<table>
<tr>
  <td width="50%">
    <h3 align="center">Object Detection</h3>
    <div align="center">
      <img src="https://github.com/user-attachments/assets/44fdcf00-ac36-4c7e-b8e0-5784da9fb0c9">
    </div>
  </td>
</tr>
</table>
