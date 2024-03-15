import onnxruntime
import numpy as np
import time
from PIL import Image

# Step 1: Load the ONNX model
onnx_model_path = 'yolov7/runs/train/to_save/weights/best.onnx'
session = onnxruntime.InferenceSession(onnx_model_path)

# Step 2: Prepare input data
# Example: Load and preprocess an image
image_path = 'yolov7/split_dataset_new/test/images/5.jpg'
image = Image.open(image_path).convert('RGB')
image = image.resize((640, 640))  # Resize the image to match input size
image_data = np.array(image, dtype=np.float32)  # Convert image to NumPy array
image_data /= 255.0  # Normalize pixel values
input_data = np.transpose(image_data, (2, 0, 1))  # Transpose dimensions to match ONNX model format
input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

# Step 3: Perform inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

start_time = time.time()
result = session.run([output_name], {input_name: input_data})
end_time = time.time()
inference_time = end_time - start_time
print("inference time:")
print(inference_time)

