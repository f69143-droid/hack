"""Streamlit app for SDG 5 gender-based participant counting."""

from __future__ import annotations

import time
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from detection.person_detector import FaceDetector
from model.gender_model import GenderClassifier, batch_predict
from utils.preprocessing import crop_face, safe_resize, to_rgb
from utils.visualization import draw_detections, summarize_predictions


st.set_page_config(page_title="SDG 5 Gender-Based Participant Counting", layout="wide")

st.title("Gender-Based Participant Counting System for SDG 5")
st.markdown(
    "**Goal:** Detect participants, classify gender presentation, and visualize counts to "
    "support Gender Equality (SDG 5) monitoring."
)

with st.sidebar:
    st.header("Controls")
    input_mode = st.radio("Input Mode", ["Upload Image", "Webcam Snapshot"], index=0)
    show_boxes = st.toggle("Show Bounding Boxes", value=True)
    min_confidence = st.slider("Detection Confidence Threshold", 0.3, 0.9, 0.5, 0.05)

st.subheader("Input")
image_bgr: np.ndarray | None = None

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload a group photo", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
else:
    camera_file = st.camera_input("Capture a snapshot")
    if camera_file is not None:
        image_bgr = cv2.imdecode(np.frombuffer(camera_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)

if image_bgr is None:
    st.info("Upload an image or capture a snapshot to begin.")
    st.stop()

start_time = time.time()
image_bgr = safe_resize(image_bgr)

face_detector = FaceDetector(min_confidence=min_confidence)
classifier = GenderClassifier()

detections = face_detector.detect(image_bgr)
faces_rgb = [crop_face(image_bgr, det.bbox) for det in detections]
predictions = batch_predict(classifier, faces_rgb) if faces_rgb else []

summary = summarize_predictions(predictions)

annotated = draw_detections(image_bgr, detections, predictions, show_boxes=show_boxes)

elapsed = time.time() - start_time
fps = 1.0 / elapsed if elapsed > 0 else 0.0

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Detections")
    st.image(to_rgb(annotated), channels="RGB", use_container_width=True)

with col2:
    st.subheader("Participant Counts")
    st.metric("Total", summary.get("Total", 0))
    st.metric("Male", summary.get("Male", 0))
    st.metric("Female", summary.get("Female", 0))
    st.metric("FPS (approx)", f"{fps:.2f}")

if not detections:
    st.warning("No faces detected. Try a clearer image or better lighting.")

chart_data = pd.DataFrame(
    {
        "Gender": ["Male", "Female"],
        "Count": [summary.get("Male", 0), summary.get("Female", 0)],
    }
)

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Gender Distribution (Bar)")
    bar_fig = px.bar(chart_data, x="Gender", y="Count", color="Gender", text="Count")
    st.plotly_chart(bar_fig, use_container_width=True)

with chart_col2:
    st.subheader("Gender Distribution (Pie)")
    pie_fig = px.pie(chart_data, names="Gender", values="Count", hole=0.4)
    st.plotly_chart(pie_fig, use_container_width=True)

st.subheader("Downloadable Report")
report_df = pd.DataFrame(
    [
        {
            "id": idx + 1,
            "gender": pred.label,
            "confidence": round(pred.confidence, 3),
        }
        for idx, pred in enumerate(predictions)
    ]
)
if report_df.empty:
    st.info("No detections to export yet.")
else:
    csv_bytes = report_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="gender_report.csv")

st.subheader("Ethical & SDG 5 Awareness")
st.warning(
    "This model infers gender visually and may not reflect self-identified gender. "
    "Use results for aggregate insights, not individual decision-making."
)
st.markdown(
    "**Bias & limitations:** Training data can over-represent certain demographics, "
    "and face-based inference fails for occlusions or low-quality images. "
    "Consider inclusive data collection and human oversight."
)
st.markdown(
    "**SDG 5 relevance:** Accurate, privacy-aware participation counts can help "
    "track representation in events, training programs, and public services."
)
