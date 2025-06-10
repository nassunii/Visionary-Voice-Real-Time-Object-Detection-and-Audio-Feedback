# Visionary Voice: Real-Time Object Detection and Audio Feedback
This project is an assignment for the **Embedded AI** course, designed to run a deep learning model using the Jetson Nano.

## Overview

The application operates as follows:
1. **Circle Detection**: Recognizes the number of circles using the camera.
2. **Audio Feedback**: Returns the detected count of circles via audio output.

## Model Architecture

- The deep learning model is composed of:
  - **Two Encoders**
  - **Two Decoder**
- The training data is custom-created and saved in `.pt` format.
- ![image](https://github.com/user-attachments/assets/d9e0f3b0-8a50-41ae-9ca1-edaa38783924)


## How to Use

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   
2. Test Camera (in Terminal)
   ```bash
    gst-launch-1.0 nvarguscamerasrc sensor_id=0 ! nvoverlaysink
3. SCI Camera (in Jupyter notebook)
   ```bash
      cap = cv2.VideoCapture("nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM),width=640, height=480, format=(string)NV12, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)

