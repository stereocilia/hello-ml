# python3 -m pip install opencv-python
import cv2
import time
# the yolo v5 model can't be used with numpy 2.x, so make sure you have 1.x
# python3 -m pip install "numpy<2.0"
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse

#
# ACCEPT COMMAND LINE ARGUMENTS
#
# Initialize the parser
parser = argparse.ArgumentParser(description="Identify objects in images using YOLOv5")

# Add a string argument
parser.add_argument('--image', type=str, required=True, help="Path to image file relative to this script")

# Add an optional argument for TFLITE_FILE_PATH
parser.add_argument(
    '--tflite_path',
    type=str,
    default='models/yolo-v5-tflite-tflite-tflite-model-v1.tflite',
    help="Path to the TFLite file (default: models/yolo-v5-tflite-tflite-tflite-model-v1.tflite)"
)

# Parse the arguments from the command line
args = parser.parse_args()

# Use the argument in the script
MAIN_IMAGE = args.image

# Use a model file that is already trained to detect objects
# https://www.kaggle.com/models/kaggle/yolo-v5
TFLITE_FILE_PATH = args.tflite_path

# Load the LiteRT model and allocate tensors.
#   python3 -m pip install ai-edge-litert
from ai_edge_litert.interpreter import Interpreter
interpreter = Interpreter(TFLITE_FILE_PATH)

print("Interpreter loaded:", interpreter)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
print(input_details)

output_details = interpreter.get_output_details()

CLASS_NAMES = [             # Class names from COCO
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair dryer", "toothbrush"
]

# Load and process the image
original_image = cv2.imread(MAIN_IMAGE)
original_height, original_width = original_image.shape[:2]

image = Image.open(MAIN_IMAGE).resize((320, 320))
input_data = np.array(image, dtype=np.float32) / 255.0
input_data = np.expand_dims(input_data, axis=0)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Process the output data (YOLOv5 specific)
# Assuming output_data has the format [batch, detections, (x, y, w, h, confidence, class)]
detections = output_data[0]

# Filter detections by confidence
confidence_threshold = 0.7
filtered_detections = detections[detections[:, 4] > confidence_threshold]

# Draw bounding boxes on the original image
for detection in filtered_detections:
    x, y, w, h  = detection[0:4]
    scores = detection[5:]  # Class confidence scores start after [x, y, w, h, obj_conf]
    class_id = np.argmax(scores)
    confidence = scores[class_id]

    # Scale bounding box coordinates back to the original image size
    x_min = int((x - w / 2) * original_width)
    y_min = int((y - h / 2) * original_height)
    x_max = int((x + w / 2) * original_width)
    y_max = int((y + h / 2) * original_height)

    #label = f"{CLASS_NAMES[class_id]}: {scores[class_id]:.2f}"
    color = (0, 255, 0)

    # Draw the bounding box
    cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), color, 2)

    # Add class label and confidence
    label = f"Class: {CLASS_NAMES[class_id]} - {confidence:.2f}"
    print(label)
    cv2.putText(original_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display or save the image with bounding boxes
# cv2.imshow("Detected Objects", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Generate a unique filename using a timestamp
unique_filename = f"output/detected_objects_{int(time.time())}.jpg"

# Optionally save the image
cv2.imwrite(unique_filename, original_image)