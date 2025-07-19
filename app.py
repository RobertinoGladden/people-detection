import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import tempfile
import yt_dlp
from PIL import Image

# Import WebRTC components
from streamlit_webrtc import webrtc_stream, VideoProcessorBase, WebRtcMode, RTCConfiguration

# ========== Title Section ==========
st.title("Smart Surveillance: Real-Time Human Detection")
st.caption("Monitor presence using YOLOv8-powered object detection from camera, video, or RTSP stream.")

# ========== Load Model ==========
@st.cache_resource
def load_model():
    # Load a pre-trained YOLOv8n model
    return YOLO('yolov8n.pt')

model = load_model()

# ========== Session State ==========
# Initialize session state variables if they don't exist
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
uploaded_file = None
rtsp_url = None
youtube_url = None

if input_option == "Video File/YouTube":
    youtube_url = st.text_input("Enter YouTube URL (optional)", "")
    uploaded_file = st.file_uploader("Upload a video file (optional)", type=["mp4", "avi", "mov"])
    if uploaded_file:
        # Create a temporary file to save the uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_source = tfile.name
        st.info(f"Video file uploaded: {uploaded_file.name}")
    elif youtube_url.startswith("http"):
        try:
            # Use yt-dlp to get the direct URL of the best quality MP4 stream
            ydl_opts = {'format': 'best[ext=mp4]', 'noplaylist': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                # Get the direct URL of the video stream
                video_source = info['url']
            st.success("YouTube video loaded successfully.")
        except Exception as e:
            st.error(f"Error loading YouTube video: {e}. Please check the URL or try another source.")
elif input_option == "RTSP Stream":
    rtsp_url = st.text_input("Enter RTSP URL", "rtsp://your_rtsp_stream_url")
    if rtsp_url and rtsp_url != "rtsp://your_rtsp_stream_url": # Ensure URL is not default placeholder
        video_source = rtsp_url
    else:
        st.warning("Please enter a valid RTSP URL to start detection.")

# Placeholder for detection notifications
notification_placeholder = st.empty()

# ========== Detect Function (now a class for WebRTC) ==========
class HumanDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()
        self.notification_shown = False # Local state for this processor instance

    def recv(self, frame):
        # Convert the WebRTC frame (av.VideoFrame) to a numpy array (OpenCV format)
        img = frame.to_ndarray(format="bgr24")

        results = self.model(img)
        people_detected_in_frame = False
        annotated_frame = img.copy()

        for result in results:
            for box in result.boxes:
                if self.model.names[int(box.cls)] == 'person':
                    people_detected_in_frame = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf.item()
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f'Person {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Handle notifications within the Streamlit context, not directly in recv()
        # This part will be handled in the main Streamlit loop after webrtc_stream returns
        st.session_state.people_detected_current_frame = people_detected_in_frame

        # Convert back to av.VideoFrame for output
        return_frame = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        return frame.from_ndarray(cv2.cvtColor(np.array(return_frame), cv2.COLOR_RGB2BGR), format="bgr24")


# WebRTC configuration (optional, but good for specifying STUN/TURN servers for NAT traversal)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ========== Button Row ==========
col1, col2, _ = st.columns([1, 1, 5])
with col1:
    if st.button("▶ Start Detection", type="primary"):
        st.session_state.running = True
        st.session_state.notification_shown = False
        st.session_state.people_detected_current_frame = False # Reset detection status
with col2:
    if st.button("⏹ Stop Detection"):
        st.session_state.running = False
        notification_placeholder.empty() # Clear notification on stop

# ========== Video Stream Processing ==========
if st.session_state.running:
    if input_option == "Realtime Camera":
        st.info("Click 'Start' below and allow browser camera access to begin real-time detection.")
        
        # Use webrtc_stream for real-time camera feed
        ctx = webrtc_stream(
            key="human_detection_webrtc",
            mode=WebRtcMode.SENDRECV, # Send video from browser, receive processed video
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=HumanDetectionProcessor,
            media_stream_constraints={"video": True, "audio": False}, # Only video
            async_processing=True, # Process frames asynchronously
        )

        if ctx.state.playing:
            # Check detection status from the processor via session state
            if st.session_state.people_detected_current_frame and not st.session_state.notification_shown:
                notification_placeholder.success("Person detected!")
                st.session_state.notification_shown = True
            elif not st.session_state.people_detected_current_frame and st.session_state.notification_shown:
                notification_placeholder.empty()
                st.session_state.notification_shown = False
        else:
            st.warning("Waiting for camera to start...")

    elif video_source: # Handles Video File/YouTube and RTSP Stream
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error(f"Failed to open video source: {video_source}. Please check the URL/file.")
            st.session_state.running = False
        else:
            stframe = st.empty() # Placeholder for video frames
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read frame from video source or end of stream reached.")
                    break # Exit loop if frame cannot be read

                # Perform detection using the model directly
                results = model(frame)
                people_detected_in_frame = False
                annotated_frame = frame.copy()

                for result in results:
                    for box in result.boxes:
                        if model.names[int(box.cls)] == 'person':
                            people_detected_in_frame = True
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf.item()
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f'Person {conf:.2f}', (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_column_width=True)

                if people_detected_in_frame and not st.session_state.notification_shown:
                    notification_placeholder.success("Person detected!")
                    st.session_state.notification_shown = True
                elif not people_detected_in_frame and st.session_state.notification_shown:
                    notification_placeholder.empty()
                    st.session_state.notification_shown = False

                time.sleep(0.05)  # Control frame rate to avoid excessive CPU usage

            cap.release()
            # Clean up temporary file if it was an uploaded video
            if input_option == "Video File/YouTube" and uploaded_file and os.path.exists(video_source):
                os.remove(video_source)
                st.info("Temporary video file cleaned up.")
    else:
        st.warning("Please select a valid input source and click 'Start Detection'.")
        st.session_state.running = False # Stop running if no valid source

# ========== Footer Info ==========
st.info("Select an input source, then click 'Start Detection' to begin human detection. Click 'Stop Detection' to halt.")
