import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# ========== Custom CSS ==========
st.markdown(
    """
    <style>
    body {
        background-color: #0f1117;
        color: #ffffff;
    }
    .main-container {
        background-color: #1e2128;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .title {
        font-size: 2rem;
        font-weight: 600;
        text-align: left;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    .subtitle {
        font-size: 1rem;
        color: #aaaaaa;
        margin-bottom: 1.5rem;
    }
    .status {
        font-size: 0.95rem;
        margin-top: 1rem;
        color: #cccccc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== Title Section ==========
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="title">Smart Surveillance: Real-Time Human Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Monitor presence in real-time using YOLOv8-powered object detection.</div>', unsafe_allow_html=True)

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

# ========== Button Row (One Line, Left Aligned) ==========
left_col, middle_col, _ = st.columns([1, 1, 5])
with left_col:
    if st.button("▶ Start Camera"):
        st.session_state.running = True
        st.session_state.notification_shown = False
with middle_col:
    if st.button("⏹ Stop Camera"):
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

# ========== Kamera Stream ==========
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal membuka kamera.")
            break

        annotated_frame, people_detected = detect_people(frame)
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        if people_detected and not st.session_state.notification_shown:
            st.success("Person detected.")
            st.session_state.notification_shown = True

        time.sleep(0.1)

    cap.release()

# ========== Footer Info ==========
st.markdown('<div class="status">Klik "Start Camera" untuk memulai streaming dan deteksi orang. Klik "Stop Camera" untuk berhenti.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
