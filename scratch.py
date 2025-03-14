# python3 -m pip install opencv-python
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TFLITE_FILE_PATH = 'models/yolo-v5-tflite-tflite-tflite-model-v1.tflite'


# Load the LiteRT model and allocate tensors.
#   python3 -m pip install ai-edge-litert

from ai_edge_litert.interpreter import Interpreter
interpreter = Interpreter(TFLITE_FILE_PATH)

# sanity check that everything loaded correctly
print("Interpreter loaded:", interpreter)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()

# Output the input_details variable to the console
print(input_details)

output_details = interpreter.get_output_details()

MAIN_IMAGE = 'images/3426785596_cd8b093e31_z.jpg'

# Load and process the image
image = Image.open(MAIN_IMAGE).resize((320, 320))
input_data = np.array(image, dtype=np.float32) / 255.0  # Normalize
# Ensure the shape matches the input tensor (batch of 1)
input_data = np.expand_dims(input_data, axis=0)  # Shape: (1, 320, 320, 3)


# Test the model on random input data.
# input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)


interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

# Debugging output data
#print("Output Data Shape:", output_data.shape)
#print("Output Data Example:", output_data[0][:10])  # Print first 10 detections

# Constants
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for detection
NMS_THRESHOLD = 0.6         # Non-max suppression threshold
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

# Paths
OUTPUT_IMAGE = 'output/output_with_boxes.jpg'

# Load original image
original_image = Image.open(MAIN_IMAGE)
original_width, original_height = original_image.size

# Image size YOLO model was trained on
INPUT_SIZE = (320, 320)

# Post-processing YOLO output
def process_yolo_output(output_data, input_dims, original_dims, confidence_threshold):
    """
    Extract boxes, scores, and class predictions from YOLOv5 output.
    """
    boxes, confidences, class_ids = [], [], []
    grid_h, grid_w = input_dims
    orig_width, orig_height = original_dims
    print(input_dims)
    print(original_dims)

    for detection in output_data[0]:  # Assuming batch_size=1
        scores = detection[5:]  # Class confidence scores start after [x, y, w, h, obj_conf]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > confidence_threshold:
            # YOLO outputs are normalized to the input image size
            print(detection[0:4])
            center_x, center_y, width, height = detection[0:4]
            box_x1 = int((center_x - width / 2) * orig_width / grid_w)
            box_y1 = int((center_y - height / 2) * orig_height / grid_h)
            box_x2 = int((center_x + width / 2) * orig_width / grid_w)
            box_y2 = int((center_y + height / 2) * orig_height / grid_h)

            boxes.append([box_x1, box_y1, box_x2, box_y2])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # Apply non-max suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, NMS_THRESHOLD)
    filtered_boxes = []
    filtered_confidences = []
    filtered_class_ids = []
    for i in indices.flatten():
        filtered_boxes.append(boxes[i])
        filtered_confidences.append(confidences[i])
        filtered_class_ids.append(class_ids[i])

    return filtered_boxes, filtered_confidences, filtered_class_ids

# Get annotated outputs
boxes, scores, class_ids = process_yolo_output(output_data, INPUT_SIZE, (original_width, original_height), CONFIDENCE_THRESHOLD)

# Draw bounding boxes on the original image
draw_image = original_image.copy()
draw = ImageDraw.Draw(draw_image)
font = ImageFont.load_default()  # You can use custom fonts if needed

for box, score, class_id in zip(boxes, scores, class_ids):
    print(box)
    x1, y1, x2, y2 = box
    label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
    color = (255, 0, 0)  # Red bounding box

    # Draw rectangle
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    # Draw label background
    text_size = draw.textsize(label, font=font)
    text_background = [x1, y1 - text_size[1], x1 + text_size[0], y1]
    draw.rectangle(text_background, fill=color)

    # Draw the text label
    draw.text((x1, y1 - text_size[1]), label, fill=(255, 255, 255), font=font)

# Save output image
draw_image.save(OUTPUT_IMAGE)

print(f"Output saved to {OUTPUT_IMAGE}")