"""
Run Yolo model on frame to get detection 
"""


from ultralytics import YOLO

def detector(frame, model_path):

    model = YOLO(model_path)
    model_names = model.names
    results = model(frame)[0]

    detections = []

    for box in results.boxes:

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        conf = float(box.conf[0])

        cls = int(box.cls[0])

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class_id": cls
        })

    return detections, model_names
    
    