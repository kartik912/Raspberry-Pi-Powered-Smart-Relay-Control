import cv2
import numpy as np
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time

# Relay setup
RELAY_PIN = 18  # Change this to the GPIO pin connected to your relay
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)  # Initially, the relay is OFF

# Load class names
classNames = []
classFile = "/home/kartik/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Paths for model configuration and weights
configPath = "/home/kartik/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/kartik/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

# Load the DNN model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)  # Reduce input size to reduce computation
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def setup_camera():
    """Initialize and configure the Raspberry Pi camera"""
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (320, 320)})  # Reduce resolution
    picam2.configure(config)
    picam2.start()
    return picam2

def getObjects(img, thres, nms, draw=True):
    """
    Detect objects in an image using the pre-trained DNN model.
    Only detect and display bounding boxes around humans/persons.
    """
    # Convert BGRA (4 channels) to BGR (3 channels) if necessary
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    person_detected = False
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className == "person":  # Only process detections for "person"
                person_detected = True
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(),
                                (box[0] + 10, box[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)),
                                (box[0] + 200, box[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img, person_detected

def main():
    try:
        # Set up the Raspberry Pi camera
        camera = setup_camera()
        last_detection_time = time.time()
        relay_status = False

        while True:
            # Capture a frame from the camera
            frame = camera.capture_array()

            # Perform object detection
            result, person_detected = getObjects(frame, 0.45, 0.2)

            # Control relay based on detection
            current_time = time.time()
            if person_detected:
                last_detection_time = current_time
                if not relay_status:
                    GPIO.output(RELAY_PIN, GPIO.HIGH)  # Turn relay ON
                    relay_status = True
                    print("Relay ON - Person Detected")
            elif current_time - last_detection_time > 5:
                if relay_status:
                    GPIO.output(RELAY_PIN, GPIO.LOW)  # Turn relay OFF
                    relay_status = False
                    print("Relay OFF - No Person Detected for 5 Seconds")

            # Display the frame with detected objects
            cv2.imshow("Live Feed with Human Detection", result)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Add a small delay to reduce CPU usage
            time.sleep(0.05)  # Add a small delay (50ms)

    except KeyboardInterrupt:
        print("Program stopped by user")

    finally:
        # Cleanup GPIO and close all OpenCV windows
        GPIO.output(RELAY_PIN, GPIO.LOW)  # Ensure the relay is OFF on exit
        GPIO.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
