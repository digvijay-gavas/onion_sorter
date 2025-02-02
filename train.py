from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # or a custom model if you have one

# Train the model on your dataset
model.train(data='./dataset.yaml', epochs=50, imgsz=640)  # You can adjust epochs and image size



#--------------------------------
# Evaluate the trained model
metrics = model.val()

# Perform inference
results = model('onion_1738482574.jpg')

# Show results for each image in the list of results
for result in results:
    result.show()  # Show the detected objects for this result

#--------------------------------------------------
# Save the model
model.save('path_to_save_model')

# Export the model to other formats if needed (e.g., TensorRT, ONNX)
model.export(format='onnx')  # or 'tflite', 'saved_model', etc.

