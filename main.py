from ultralytics import YOLO
from configs import config
from detection import detector
import cv2

cap = cv2.VideoCapture(r"E:\Selective_Object_Tracking\outputs\Drone Street Traffic, New York City.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    "outputs/output.mp4",
    fourcc,
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    

    detections, names = detector(frame,
                    config.MODEL_PATH)
    

    for det in detections:

        if names[det['class_id']] == 'car':
            x1, y1, x2, y2 = map(int, det["bbox"])
    
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )


    writer.write(frame)

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break


cap.release()
writer.release()
cv2.destroyAllWindows()
