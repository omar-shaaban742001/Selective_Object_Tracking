import time
import cv2
import numpy as np
from configs import config
from detection import detector
from tracker import tracking


def process_video(input_path, skip_frames, start_frame, target_id, selected_class):

    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    writer = cv2.VideoWriter(
        "outputs/output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    start_time = time.time()

    while True:

        # ✅ proper skipping
        for _ in range(skip_frames - 1):
            cap.grab()

        ret, frame = cap.read()
        if not ret:
            break

        xyxy, confidences, class_ids = [], [], []

        detections, names = detector(frame, config.MODEL_MIDIUM_NCNN_PATH)

        for det in detections:

            if selected_class == "all" or names[det['class_id']] == selected_class:
                xyxy.append(det['bbox'])
                confidences.append(det['confidence'])
                class_ids.append(det['class_id'])

        # ✅ SAFE TRACKER INPUT
        if len(xyxy) == 0:
            tracked_objects = []
        else:
            tracked_objects = tracking(
                xyxy=np.array(xyxy),
                confidence=np.array(confidences),
                class_id=np.array(class_ids)
            )

        for track in tracked_objects:
            # print(track)
            x1, y1, x2, y2 = map(int, track[0])
            track_id = int(track[4])

            color = (0, 0, 255) if track_id == target_id else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            class_id = int(track[3])  # if tracker returns it
            class_name = names[class_id]

            cv2.putText(
                frame,
                f"ID: {track_id} | {class_name}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )


        writer.write(frame)

        elapsed = time.time() - start_time
        proc_fps = 1 / (elapsed + 1e-6)

        yield frame, {
            "fps": proc_fps,
            "frame_count": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            "tracks": len(tracked_objects)
        }

    writer.release()
    cap.release()