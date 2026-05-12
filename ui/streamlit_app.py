import sys 
import os 
sys.path.append( os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) )
ROOT_DIR = os.path.abspath( os.path.join(os.path.dirname(__file__), "..") )


import streamlit as st
import cv2
import os
import pipeline

st.title("Multi Object Tracking System")

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    input_path = "temp_input.mp4"
    output_path = "outputs/output.mp4"

    # save upload
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("Video Uploaded")

    # sidebar
    st.sidebar.title("System Info")
    fps_box = st.sidebar.empty()
    track_box = st.sidebar.empty()

    video_box = st.empty()

    if st.button("Start Tracking"):

        # 🚀 REAL-TIME PIPELINE STREAM
        for frame, info in pipeline.process_video(input_path, 2):

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            video_box.image(frame, channels="RGB")

            fps_box.metric("FPS", f"{info['fps']:.2f}")
            track_box.metric("Tracks", info["tracks"])

    # show final video AFTER processing exists
    if os.path.exists(output_path):
        st.video(output_path)