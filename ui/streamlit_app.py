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
# UI
# =========================
st.title("Multi Object Tracking System")

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    # =========================
    # SESSION STATES
    # =========================
    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0

    if "uploaded_name" not in st.session_state:
        st.session_state.uploaded_name = ""

    # =========================
    # SAVE VIDEO ONLY IF NEW
    # =========================
    if st.session_state.uploaded_name != uploaded_file.name:

        with open(INPUT_PATH, "wb") as f:
            f.write(uploaded_file.read())

        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.frame_idx = 0

    st.success("Video Uploaded")

    # =========================
    # VIDEO INFO
    # =========================
    cap = cv2.VideoCapture(INPUT_PATH)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    # =========================
    # RESET IF REACHED END
    # =========================
    if st.session_state.frame_idx >= total_frames:
        st.session_state.frame_idx = 0

    # =========================
    # SIDEBAR
    # =========================
    st.sidebar.title("System Info")

    fps_box = st.sidebar.empty()
    track_box = st.sidebar.empty()

    target_id = st.sidebar.number_input(
        "Track ID to follow",
        min_value=0,
        value=0,
        step=1
    )

    classes = [
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

    # initialize selected class state
    if "previous_class" not in st.session_state:
        st.session_state.previous_class = selected_class

    # reset if class changed
    if selected_class != st.session_state.previous_class:

        st.session_state.frame_idx = 0



        st.session_state.previous_class = selected_class
    # st.write("Selected class:", selected_class)
    progress_bar = st.sidebar.progress(0)

    video_box = st.empty()

    # =========================
    # START PROCESSING
    # =========================
    if st.button("Start Tracking"):

        for frame, info in pipeline.process_video(
            INPUT_PATH,
            skip_frames=1,
            target_id=target_id,
            start_frame=st.session_state.frame_idx,
            selected_class=selected_class
        ):

            st.session_state.frame_idx = info["frame_count"]

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            video_box.image(frame, channels="RGB")

            fps_box.metric(
                "FPS",
                f"{info['fps']:.2f}"
            )

            track_box.metric(
                "Tracks",
                info["tracks"]
            )

            progress_bar.progress(
                min(
                    info["frame_count"] / total_frames,
                    1.0
                )
            )

# =========================
# FINAL VIDEO
# =========================
if os.path.exists(OUTPUT_PATH):
    st.video(OUTPUT_PATH)