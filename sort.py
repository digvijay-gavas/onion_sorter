import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use a custom model if available

# Open the laptop camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

onion_sizes = []  # List to store detected onion sizes

# COCO class labels (YOLOv8 default)
COCO_CLASSES = model.names  # Load class names
ONION_CLASS_NAME = "onion"  # Change this if using a custom model

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Run YOLO detection
    results = model(frame)

    # Draw bounding boxes & extract sizes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            width = x2 - x1  # Approximate onion size
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class index
            class_name = COCO_CLASSES.get(cls, "Unknown")  # Get class name

            # Check if it's an onion
            if class_name.lower() == ONION_CLASS_NAME:
                color = (0, 255, 0)  # Green for onions
                onion_sizes.append(width)
                label = f"Onion {conf:.2f} | Size: {width}px"
            else:
                color = (0, 0, 255)  # Red for non-onion objects
                label = f"{class_name} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the video feed
    cv2.imshow("Onion Detection", frame)

    # Press 'q' to exit & plot size distribution
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Plot the onion size distribution
if onion_sizes:
    plt.figure(figsize=(8, 5))
    plt.hist(onion_sizes, bins=10, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel("Onion Size (Bounding Box Width in Pixels)")
    plt.ylabel("Frequency")
    plt.title("Onion Size Distribution")
    plt.grid(True)
    plt.show()
else:
    print("No onions detected for size analysis.")
