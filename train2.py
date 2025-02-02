from flask import Flask, render_template, request, jsonify
from flask import send_from_directory
import os
import time
import cv2
from flask import Flask, Response

cap=None
# Initialize Flask app
app = Flask(__name__)

# Set up directories for saving images and labels
IMAGE_DIR = "captured_images"
LABEL_DIR = "labels"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# Route to serve the main UI
@app.route('/')
def index():
    # Get all captured images and labels
    image_files = os.listdir(IMAGE_DIR)
    image_files.sort(reverse=True)
    
    images_data = []
    for image_file in image_files:
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(LABEL_DIR, label_file)
        label = None
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = f.read()
        images_data.append({
            "image_filename": image_file,
            "label": label
        })
    
    return render_template('index.html', images_data=images_data)

# Route to capture an image from the webcam
@app.route('/capture', methods=['POST'])
def capture():
    #cap = cv2.VideoCapture(0)  # Initialize the webcam

    if not cap.isOpened():
        return jsonify({"error": "Could not open webcam"}), 500
    
    ret, frame = cap.read()
    #cap.release()

    if not ret:
        return jsonify({"error": "Failed to capture image"}), 500

    # Save the captured image
    img_filename = f"onion_{int(time.time())}.jpg"
    img_path = os.path.join(IMAGE_DIR, img_filename)
    cv2.imwrite(img_path, frame)

    # Encode the image to send it to the frontend
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_data = img_encoded.tobytes()

    # Return image path and base64 image data to frontend
    return jsonify({
        "image_path": img_filename,
        "image_data": img_data.hex()  # Send as hex string (base64 could be another approach)
    })

# Route to save the bounding box label for an image
@app.route('/save_label', methods=['POST'])
def save_label():
    data = request.json
    image_filename = data['image_filename']
    x1, y1, x2, y2 = data['bbox']

    label_filename = image_filename.replace('.jpg', '.txt')
    label_path = os.path.join(LABEL_DIR, label_filename)

    # Save label in YOLO format
    with open(label_path, 'w') as f:
        f.write(f"0 {(x1 + x2) / 2} {(y1 + y2) / 2} {(x2 - x1)} {(y2 - y1)}\n")

    return jsonify({"message": "Label saved successfully"})

# Route to delete a captured image and its label
@app.route('/delete_image', methods=['POST'])
def delete_image():
    data = request.json
    image_filename = data['image_filename']

    # Delete image and label
    image_path = os.path.join(IMAGE_DIR, image_filename)
    label_path = os.path.join(LABEL_DIR, image_filename.replace('.jpg', '.txt'))

    if os.path.exists(image_path):
        os.remove(image_path)
    if os.path.exists(label_path):
        os.remove(label_path)

    return jsonify({"message": "Image and label deleted successfully"})

@app.route('/captured_images/<filename>')
def send_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



# Create a route to stream the webcam feed
def generate_frames():
    global cap
    if(cap==None):
        cap = cv2.VideoCapture(0)  # Initialize webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as part of the MJPEG stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/get_image_gallery', methods=['GET'])
def get_image_gallery():
    """Get a list of captured images."""
    images = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)]
    return jsonify({"images": images})

if __name__ == "__main__":
    app.run(debug=True)
