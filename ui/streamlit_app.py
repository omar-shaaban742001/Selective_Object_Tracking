import sys
import os
import streamlit as st
import cv2

# =========================
# PATHS
# =========================
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(ROOT_DIR)

import pipeline

INPUT_PATH = os.path.join(ROOT_DIR, "temp_input.mp4")
OUTPUT_PATH = os.path.join(ROOT_DIR, "outputs", "output.mp4")

# =========================
# SESSION STATES
# =========================
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = ""

if "is_tracking" not in st.session_state:
    st.session_state.is_tracking = False

# =========================
# UI
# =========================
st.title("Multi Object Tracking System")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("System Info")

source_type = st.sidebar.selectbox(
    "Video Source",
    ["Upload Video", "Webcam"]
)

target_id = st.sidebar.number_input(
    "Track ID to follow",
    min_value=0,
    value=0,
    step=1
)

classes = [
    'all',
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle"
]

selected_class = st.sidebar.selectbox(
    "Select Class",
    classes
)

# =========================
# RESET IF CLASS CHANGED
# =========================
if "previous_class" not in st.session_state:
    st.session_state.previous_class = selected_class

if selected_class != st.session_state.previous_class:

    st.session_state.frame_idx = 0

    st.session_state.previous_class = selected_class

# =========================
# VIDEO SOURCE
# =========================
input_source = None
total_frames = 1

# ---------- Upload Video ----------
if source_type == "Upload Video":

    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file is not None:

        if st.session_state.uploaded_name != uploaded_file.name:

            with open(INPUT_PATH, "wb") as f:
                f.write(uploaded_file.read())

            st.session_state.uploaded_name = uploaded_file.name
            st.session_state.frame_idx = 0

        input_source = INPUT_PATH

        st.success("Video Uploaded")

        cap = cv2.VideoCapture(INPUT_PATH)

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        cap.release()

        # reset if video ended
        if st.session_state.frame_idx >= total_frames:
            st.session_state.frame_idx = 0

# ---------- Webcam ----------
elif source_type == "Webcam":

    input_source = 0

# =========================
# UI BOXES
# =========================
fps_box = st.sidebar.empty()
track_box = st.sidebar.empty()
progress_bar = st.sidebar.progress(0)

video_box = st.empty()

# =========================
# START / STOP BUTTONS
# =========================
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Tracking"):
        st.session_state.is_tracking = True

with col2:
    if st.button("⏹ Stop Tracking"):
        st.session_state.is_tracking = False

# =========================
# START PROCESSING
# =========================
if st.session_state.is_tracking and input_source is not None:

    for frame, info in pipeline.process_video(
        input_source,
        skip_frames=1,
        target_id=target_id,
        start_frame=(
            0 if source_type == "Webcam"
            else st.session_state.frame_idx
        ),
        selected_class=selected_class
    ):

        # stop immediately
        if not st.session_state.is_tracking:
            break

        st.session_state.frame_idx = info["frame_count"]

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        video_box.image(
            frame,
            channels="RGB"
        )

        fps_box.metric(
            "FPS",
            f"{info['fps']:.2f}"
        )

        track_box.metric(
            "Tracks",
            info["tracks"]
        )

        # only for uploaded videos
        if source_type == "Upload Video":

            progress_bar.progress(
                min(
                    info["frame_count"] / total_frames,
                    1.0
                )
            )

# =========================
# FINAL VIDEO
# =========================
if (
    source_type == "Upload Video"
    and os.path.exists(OUTPUT_PATH)
):
    st.video(OUTPUT_PATH)