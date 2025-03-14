from tflite_runtime.interpreter import Interpreter

# Path to your TensorFlow Lite model
model_path = "yolo-v5-tflite-tflite-tflite-model-v1/1.tflite"

# Initialize the TFLite interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded successfully!")
print("Input details:", input_details)
print("Output details:", output_details)