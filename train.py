import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Set up directories
IMAGE_DIR = "captured_images"
LABEL_DIR = "labels"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

st.title("Onion Labeling and Training UI")

# Open camera and capture image
camera = st.button("Capture Image")
if camera:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        img_path = os.path.join(IMAGE_DIR, f"onion_{len(os.listdir(IMAGE_DIR))}.jpg")
        cv2.imwrite(img_path, frame)
        st.image(frame, caption="Captured Image", channels="BGR")
        st.session_state["last_image"] = img_path

# Load last captured image for labeling
if "last_image" in st.session_state:
    img = Image.open(st.session_state["last_image"])
    st.image(img, caption="Label the Onion", use_container_width=True)
    
    x1 = st.slider("X1", 0, img.width, 0)
    y1 = st.slider("Y1", 0, img.height, 0)
    x2 = st.slider("X2", 0, img.width, img.width)
    y2 = st.slider("Y2", 0, img.height, img.height)
    
    if st.button("Save Label"):
        label_path = st.session_state["last_image"].replace(".jpg", ".txt").replace(IMAGE_DIR, LABEL_DIR)
        with open(label_path, "w") as f:
            f.write(f"0 {(x1+x2)/2/img.width} {(y1+y2)/2/img.height} {(x2-x1)/img.width} {(y2-y1)/img.height}\n")
        st.success(f"Label saved: {label_path}")

st.write("After labeling multiple images, you can proceed with training.")
