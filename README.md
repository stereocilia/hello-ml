# hello-ml  

a place for my [LiteRT](https://ai.google.dev/edge/litert)  experiments, primarily focused on detecting objects.  

  - Install python3 on your development machine
  - [Download the model `.tflite` file](https://www.kaggle.com/models/kaggle/yolo-v5), add it to the `models` folder, and rename it to
`yolo-v5-tflite-tflite-tflite-model-v1.tflite`
  - Add the image you want to test for objects to the `images` folder
  - Run the script as: `python3 main.py --image "images/my-image.jpg"`, replacing the image file path with your image's
actual file path. Example: `python3 main.py --image 'images/3426785596_cd8b093e31_z.jpg`
  - Run `python3 main.py` for all options
  - Results are output to the 'output' folder as a uniquely named file. If the output file shows bounding boxes, a detection was made. If not, no detection was made. 
