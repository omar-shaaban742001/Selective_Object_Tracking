from ultralytics import YOLO
from configs import config
from detection import detector
import cv2
from tracker import tracking


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
    
    xyxy = []
    confidences = []
    class_ids = []

    detections, names = detector(frame,
                    config.MODEL_MIDIUM_PATH)



    for det in detections:

        if names[det['class_id']] == 'car':

            # x1, y1, x2, y2 = map(int, det["bbox"])
            xyxy.append(det['bbox'])
            confidences.append(det['confidence'])
            class_ids.append(det['class_id'])

    tracked_objects = tracking(xyxy=xyxy, confidence=confidences,class_id=class_ids)
    for track in tracked_objects:
        x1, y1, x2, y2 = map(int, track[0])

        track_id = int(track[4])
        if track_id == 4:
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )


    writer.write(frame)

    # cv2.imshow("Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break


cap.release()
writer.release()
cv2.destroyAllWindows()
