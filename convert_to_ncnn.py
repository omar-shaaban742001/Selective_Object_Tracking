from ultralytics import YOLO
from configs import config
# Load the YOLO26 model
model = YOLO(config.MODEL_MIDIUM_PATH)

# Export the model to NCNN format
model.export(format="ncnn")  # creates '/yolo26n_ncnn_model'

# Load the exported NCNN model
# ncnn_model = YOLO("models/yolo26n_ncnn_model")
