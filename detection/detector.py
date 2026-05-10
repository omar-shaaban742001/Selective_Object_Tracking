from ultralytics import YOLO

def detector(frame, model_path):

    model = YOLO(model_path)


    results = model(frame, save=True)

    return results
    
    