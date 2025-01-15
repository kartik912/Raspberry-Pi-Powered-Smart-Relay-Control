# Person Detection with Raspberry Pi for Smart Relay Control

This project demonstrates how to leverage computer vision using Raspberry Pi to control a relay in real time. The system uses object detection to identify a person from a live camera feed and toggles a relay accordingly. This application showcases a practical example of IoT and automation for energy efficiency.

## Features

- **Real-time Person Detection**: Uses a pre-trained YOLO-based SSD MobileNet v3 model.
- **Relay Control**: Turns the relay ON when a person is detected and OFF after 5 seconds of no detection.
- **Hardware Integration**: Combines Raspberry Pi with a relay module for seamless operation.
- **Optimized for Low Power**: Processes video at reduced resolution for efficient computation on Raspberry Pi.

---

## System Overview

### Hardware Components

1. **Raspberry Pi** (any model with GPIO support, e.g., Raspberry Pi 4)
2. **Raspberry Pi Camera Module**
3. **Relay Module** (connected to GPIO pin 18 in this project)
4. **Power Supply** for Raspberry Pi and relay
5. **Wiring and Breadboard** (if necessary for connections)

### Software Requirements

1. **Python 3.x**
2. **OpenCV** (for computer vision tasks)
3. **Picamera2** (to interface with Raspberry Pi Camera)
4. **RPi.GPIO** (for GPIO control on Raspberry Pi)
5. **Linux Environment** (tested on Raspberry Pi OS)

---

## Installation and Setup

### Hardware Setup

1. Connect the relay module to the Raspberry Pi GPIO pin 18.
2. Attach the Raspberry Pi Camera Module and ensure it is enabled in the Raspberry Pi settings.
3. Power up the Raspberry Pi.

### Software Installation

1. Install the required libraries:
   ```bash
   sudo apt update
   sudo apt install python3-opencv python3-picamera2
   pip3 install numpy RPi.GPIO
   ```
2. Clone this repository and navigate to the project folder.
   ```bash
   git clone <repository-link>
   cd person-detection-relay
   ```

---

## Running the Project

1. Run the Python script:
   ```bash
   python3 main.py
   ```
2. Observe the live feed and relay control in action. Press `q` to exit.

---

## How It Works

1. The **Picamera2** library captures real-time video frames from the Raspberry Pi camera.
2. Each frame is processed using OpenCV's DNN module, which uses the YOLO-based SSD MobileNet v3 model to detect objects.
3. If a person is detected:
   - The relay is turned ON immediately.
   - A 5-second timer resets every time a person is detected.
4. If no person is detected for 5 seconds:
   - The relay is turned OFF.
5. A live feed with bounding boxes and detection confidence is displayed.

---

## Example Use Cases

- **Smart Lighting**: Automatically turn on lights when someone enters a room and turn them off when they leave.
- **Energy Efficiency**: Control appliances based on occupancy to save energy.
- **Security Systems**: Monitor areas and trigger alarms or other systems when a person is detected.

---

## Key Code Highlights

- **Object Detection**: The YOLO-based SSD MobileNet v3 model detects objects with high accuracy and efficiency.
- **Relay Control Logic**: Utilizes GPIO pins for hardware control, ensuring real-time response.
- **Optimized Processing**: Reduced input size (320x320) and selective detection (person class only) for better performance on Raspberry Pi.

---

## Future Improvements

1. Integrate with a cloud platform (e.g., AWS or Google Cloud) for remote monitoring and control.
2. Add support for multiple relays and configurable GPIO pins.
3. Enhance detection accuracy by fine-tuning the YOLO model for specific environments.

---

## License

This project is open-source and available under the MIT License. Contributions are welcome!


