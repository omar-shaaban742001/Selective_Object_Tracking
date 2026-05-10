import supervision as sv
import numpy as np

tracker = sv.ByteTrack()
def tracking(xyxy, confidence, class_id):

    sv_detections = sv.Detections(
        xyxy = np.array(xyxy),
        confidence = np.array(confidence),
        class_id = np.array(class_id)
    )

    tracked_objects = tracker.update_with_detections(
    sv_detections
    )

    return tracked_objects