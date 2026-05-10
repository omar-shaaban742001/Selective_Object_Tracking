from ultralytics import YOLO
from configs import config
from detection import detector

output = detector(r"E:\Selective_Object_Tracking\WhatsApp Image 2026-05-07 at 15.53.54.jpeg",
                  config.MODEL_PATH)

print(output)
