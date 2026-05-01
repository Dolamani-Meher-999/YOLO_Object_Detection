import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import time

# ---------------- CONFIG ----------------

MODEL_PATH = "yolov8n.pt"  # change to your trained model if needed

# ----------------------------------------

st.set_page_config(page_title="YOLO Object Detection", layout="wide")

st.title("Real-Time Object Detection using YOLO")
st.write("Upload an image, video, or use webcam for detection.")

# Load model (cached)
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# Sidebar controls
st.sidebar.header("Settings")
confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

option = st.sidebar.selectbox(
    "Select Input Type",
    ("Image", "Video", "Webcam")
)

# ---------------- IMAGE DETECTION ----------------

if option == "Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        results = model(image, conf=confidence)
        annotated_frame = results[0].plot()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, channels="BGR")

        with col2:
            st.subheader("Detected Objects")
            st.image(annotated_frame, channels="BGR")

# ---------------- VIDEO DETECTION ----------------

elif option == "Video":

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        prev_time = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            current_time = time.time()

            results = model(frame, conf=confidence)
            annotated_frame = results[0].plot()

            fps = 1 / (current_time - prev_time) if prev_time else 0
            prev_time = current_time

            cv2.putText(
                annotated_frame,
                f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            stframe.image(annotated_frame, channels="BGR")

        cap.release()

# ---------------- WEBCAM DETECTION ----------------

elif option == "Webcam":

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    prev_time = 0

    while run:

        ret, frame = cap.read()

        if not ret:
            st.write("Failed to access webcam")
            break

        current_time = time.time()

        results = model(frame, conf=confidence)
        annotated_frame = results[0].plot()

        # Count persons
        person_count = 0

        if results[0].boxes is not None:
            classes = results[0].boxes.cls
            for cls in classes:
                if int(cls) == 0:
                    person_count += 1

        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        cv2.putText(
            annotated_frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Persons: {person_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        FRAME_WINDOW.image(annotated_frame, channels="BGR")

    cap.release()

st.markdown("---")
st.caption("YOLO Object Detection Project | Built with Streamlit")
