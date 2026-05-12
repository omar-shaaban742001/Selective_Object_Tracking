import time
import cv2
from configs import config
from detection import detector
from tracker import tracking


def process_video(input_path, skip_frames, start_frame, target_id, selected_class):

    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        "outputs/output.mp4",
        fourcc,
        fps,
        (width, height)
    )

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ✅ proper skipping
        if frame_count % skip_frames != 0:
            continue

        xyxy, confidences, class_ids = [], [], []

        detections, names = detector(frame, config.MODEL_MIDIUM_PATH)

        for det in detections:
            if names[det['class_id']] == selected_class:
                xyxy.append(det['bbox'])
                confidences.append(det['confidence'])
                class_ids.append(det['class_id'])

        tracked_objects = tracking(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_ids
        )

        # draw boxes
        for track in tracked_objects:

            x1, y1, x2, y2 = map(int, track[0])
            track_id = int(track[4])
            # 🎯 highlight selected ID
            if target_id is not None and track_id == target_id:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

        # ✅ ensure correct size
        frame = cv2.resize(frame, (width, height))

        writer.write(frame)

        # =========================
        # 📊 REAL-TIME METRICS
        # =========================
        elapsed = time.time() - start_time
        proc_fps = frame_count / (elapsed + 1e-6)

        yield frame, {
            "fps": proc_fps,
            "frame_count": frame_count,
            "tracks": len(tracked_objects)
        }

    writer.release()
    cap.release()