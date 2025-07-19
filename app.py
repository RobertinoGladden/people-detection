import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import youtube_dl
import pafy
import os
import tempfile
import yt_dlp

# ========== Title Section ==========
st.title("Smart Surveillance: Real-Time Human Detection")
st.caption("Monitor presence using YOLOv8-powered object detection from camera, video, or RTSP stream.")

# ========== Load Model ==========
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# ========== Session State ==========
if 'running' not in st.session_state:
    st.session_state.running = False
if 'notification_shown' not in st.session_state:
    st.session_state.notification_shown = False

# ========== Input Selection ==========
input_option = st.selectbox(
    "Select Input Source",
    ["Realtime Camera", "Video File/YouTube", "RTSP Stream"],
    help="Choose the source for human detection: camera, video file/YouTube link, or RTSP stream."
)

# ========== Input Handling ==========
video_source = None
# Ganti bagian input YouTube di kode sebelumnya
if input_option == "Video File/YouTube":
    video_input = st.text_input("Enter YouTube URL or upload a video file", "")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_source = tfile.name
    elif video_input.startswith("http"):
        try:
            ydl_opts = {'format': 'best[ext=mp4]'}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_input, download=False)
                video_source = info['url']
        except Exception as e:
            st.error(f"Error loading YouTube video: {e}")
elif input_option == "RTSP Stream":
    rtsp_url = st.text_input("Enter RTSP URL", "rtsp://your_rtsp_stream_url")
    if rtsp_url:
        video_source = rtsp_url
else:
    video_source = 0  # Default to webcam

# ========== Button Row ==========
col1, col2, _ = st.columns([1, 1, 5])
with col1:
    if st.button("▶ Start Detection", type="primary"):
        st.session_state.running = True
        st.session_state.notification_shown = False
with col2:
    if st.button("⏹ Stop Detection"):
        st.session_state.running = False

# ========== Detect Function ==========
def detect_people(frame):
    results = model(frame)
    people_detected = False
    annotated_frame = frame.copy()

    for result in results:
        for box in result.boxes:
            if result.names[int(box.cls)] == 'person':
                people_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'Person {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return annotated_frame, people_detected

# ========== Video Stream Processing ==========
if st.session_state.running and video_source:
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Failed to open video source.")
        st.session_state.running = False
    else:
        stframe = st.empty()
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame from video source.")
                break

            annotated_frame, people_detected = detect_people(frame)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)

            if people_detected and not st.session_state.notification_shown:
                st.success("Person detected.")
                st.session_state.notification_shown = True

            time.sleep(0.05)  # Control frame rate

        cap.release()
        if input_option == "Video File/YouTube" and uploaded_file:
            os.remove(video_source)  # Clean up temporary file

# ========== Footer Info ==========
st.info("Select an input source, then click 'Start Detection' to begin human detection. Click 'Stop Detection' to halt.")