import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import time

# Set up directories
IMAGE_DIR = "captured_images"
LABEL_DIR = "labels"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# Function to draw rectangle
def draw_rect(event, x, y, flags, param):
    global ix, iy, drawing, img_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
        # Save bounding box coordinates
        global x1, y1, x2, y2
        x1, y1, x2, y2 = ix, iy, x, y

# Streamlit UI setup
st.title("Onion Labeling and Training UI")

# Video stream preview
camera_preview = st.empty()  # This is for displaying the live camera feed

cap = cv2.VideoCapture(0)

# Check if the camera is open
if not cap.isOpened():
    st.error("Error: Unable to access the camera.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture frame.")
            break
        
        # Display the live feed
        camera_preview.image(frame, channels="BGR", use_container_width=True)
        
        # Create a unique key for the capture button
        capture_button_key = f"capture_button_{time.time()}"
        
        # Wait for the user to click on the "Capture Image" button
        camera = st.button("Capture Image", key=capture_button_key)
        if camera:
            img_path = os.path.join(IMAGE_DIR, f"onion_{len(os.listdir(IMAGE_DIR))}.jpg")
            cv2.imwrite(img_path, frame)
            st.image(frame, caption="Captured Image", channels="BGR")
            st.session_state["last_image"] = img_path
            break

# Load last captured image for labeling
if "last_image" in st.session_state:
    img = cv2.imread(st.session_state["last_image"])
    img_copy = img.copy()
    
    # Initialize drawing state
    ix, iy = -1, -1
    drawing = False
    x1, y1, x2, y2 = 0, 0, 0, 0

    # Set mouse callback to draw bounding box
    cv2.namedWindow("Label Image")
    cv2.setMouseCallback("Label Image", draw_rect)

    while True:
        cv2.imshow("Label Image", img_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit drawing
            break

    cv2.destroyAllWindows()

    # Display the image and allow user to save the label
    st.image(img, caption="Label the Onion", channels="BGR", use_container_width=True)
    
    st.write(f"Bounding Box Coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    
    # Create a unique key for the "Save Label" button
    save_button_key = f"save_button_{time.time()}"
    
    if st.button("Save Label", key=save_button_key):  # Unique key for Save Label button
        label_path = st.session_state["last_image"].replace(".jpg", ".txt").replace(IMAGE_DIR, LABEL_DIR)
        with open(label_path, "w") as f:
            # Normalize coordinates
            f.write(f"0 {(x1 + x2) / 2 / img.shape[1]} {(y1 + y2) / 2 / img.shape[0]} {(x2 - x1) / img.shape[1]} {(y2 - y1) / img.shape[0]}\n")
        st.success(f"Label saved: {label_path}")

st.write("After labeling multiple images, you can proceed with training.")
