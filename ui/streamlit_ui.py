from __future__ import annotations

import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.pipeline = TrackingPipeline(
            config=DetectionConfig(
                model_name="yolo11n.pt",
                confidence=0.35,
                iou=0.55,
                imgsz=640,
            ),
            source_name="webcam",
            enable_line=True,
            enable_zone=True,
            enable_voice=False,
            enable_screenshots=False,
            save_output_video=False,
        )

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        result = self.pipeline.process_frame(img, source_fps=30)

        return result.annotated_frame

from config import AVAILABLE_MODELS, DetectionConfig, PATHS
from tracker.tracking_pipeline import TrackingPipeline
from ui.styles import STREAMLIT_CSS
from utils.coco import COCO_CLASSES, class_ids_from_names
from utils.video import open_video_capture, safe_video_fps


def _source_to_capture(source_type: str, uploaded_file, camera_index: int, network_url: str):
    if source_type == "Webcam":
        return camera_index, f"webcam_{camera_index}"
    if source_type == "IP/RTSP camera":
        return network_url, network_url
    if uploaded_file is None:
        return None, "uploaded_video"

    suffix = Path(uploaded_file.name).suffix or ".mp4"
    target = PATHS.uploads_dir / f"streamlit_upload_{int(time.time())}{suffix}"
    with target.open("wb") as file:
        file.write(uploaded_file.getbuffer())
    return str(target), uploaded_file.name


def _build_config(settings: dict[str, object]) -> DetectionConfig:
    selected_classes = settings["classes"]
    class_ids = class_ids_from_names(selected_classes) if selected_classes else None
    return DetectionConfig(
        model_name=str(settings["model"]),
        confidence=float(settings["confidence"]),
        iou=float(settings["iou"]),
        imgsz=int(settings["imgsz"]),
        device=str(settings["device"]) if settings["device"] != "auto" else None,
        half=bool(settings["half"]),
        augment=bool(settings["augment"]),
        agnostic_nms=bool(settings["agnostic_nms"]),
        classes=class_ids,
        low_light=bool(settings["low_light"]),
    )


def _render_sidebar() -> dict[str, object]:
    st.sidebar.title("Vision Control")
    source_type = st.sidebar.radio("Source", ["Webcam", "Video file", "IP/RTSP camera"], horizontal=False)
    uploaded_file = None
    network_url = ""
    camera_index = 0
    if source_type == "Webcam":
        camera_index = st.sidebar.number_input("Camera index", min_value=0, max_value=8, value=0, step=1)
    elif source_type == "Video file":
        uploaded_file = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv", "webm"])
    else:
        network_url = st.sidebar.text_input("Stream URL", placeholder="rtsp:// or http://")

    model_label = st.sidebar.selectbox("Model", list(AVAILABLE_MODELS.keys()), index=0)
    confidence = st.sidebar.slider("Confidence", 0.10, 0.85, 0.35, 0.01)
    iou = st.sidebar.slider("NMS IoU", 0.30, 0.90, 0.55, 0.01)
    imgsz = st.sidebar.select_slider("Image size", options=[640, 768, 960, 1280], value=960)
    device = st.sidebar.selectbox("Device", ["auto", "cpu", "0", "cuda:0", "mps"], index=0)

    with st.sidebar.expander("Accuracy and speed", expanded=True):
        half = st.checkbox("FP16 on CUDA", value=True)
        low_light = st.checkbox("Low-light enhancement", value=False)
        augment = st.checkbox("Test-time augmentation", value=False)
        agnostic_nms = st.checkbox("Class-agnostic NMS", value=False)
        classes = st.multiselect("Class filter", COCO_CLASSES, default=[])

    with st.sidebar.expander("Events and output", expanded=True):
        enable_line = st.checkbox("Line crossing", value=True)
        enable_zone = st.checkbox("Restricted zone", value=True)
        enable_voice = st.checkbox("Voice alerts", value=False)
        enable_screenshots = st.checkbox("Screenshots on detection", value=True)
        save_output_video = st.checkbox("Save output video", value=True)
        max_frames = st.number_input("Frame limit", min_value=0, max_value=100000, value=0, step=100)

    run = st.sidebar.toggle("Run detection", value=False)
    return {
        "source_type": source_type,
        "uploaded_file": uploaded_file,
        "camera_index": int(camera_index),
        "network_url": network_url,
        "model": AVAILABLE_MODELS[model_label],
        "confidence": confidence,
        "iou": iou,
        "imgsz": imgsz,
        "device": device,
        "half": half,
        "low_light": low_light,
        "augment": augment,
        "agnostic_nms": agnostic_nms,
        "classes": classes,
        "enable_line": enable_line,
        "enable_zone": enable_zone,
        "enable_voice": enable_voice,
        "enable_screenshots": enable_screenshots,
        "save_output_video": save_output_video,
        "max_frames": int(max_frames),
        "run": run,
    }


def _render_stats(snapshot: dict[str, object] | None) -> None:
    if not snapshot:
        st.markdown(
            '<div class="status-card"><strong>Idle</strong><br>Choose a source and start detection.</div>',
            unsafe_allow_html=True,
        )
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("FPS", f"{snapshot['fps']:.1f}")
    col2.metric("Visible", int(snapshot["current_objects"]))
    col3.metric("Unique IDs", int(snapshot["unique_objects"]))
    col4.metric("Zone Active", int(snapshot["restricted_active"]))

    counts = snapshot.get("frame_counts", {})
    if counts:
        data = pd.DataFrame({"class": list(counts.keys()), "count": list(counts.values())})
        st.bar_chart(data, x="class", y="count", height=220)

    events = snapshot.get("events", [])
    if events:
        st.subheader("Live Events")
        for event in events[:6]:
            st.markdown(f'<div class="event-row">{event}</div>', unsafe_allow_html=True)

    if snapshot.get("output_video"):
        st.caption(f"Saving video: {snapshot['output_video']}")
    if snapshot.get("latest_screenshot"):
        st.caption(f"Latest screenshot: {snapshot['latest_screenshot']}")


def _run_stream(settings: dict[str, object], stats_slot) -> None:
    source, source_name = _source_to_capture(
        str(settings["source_type"]),
        settings["uploaded_file"],
        int(settings["camera_index"]),
        str(settings["network_url"]),
    )
    if source is None:
        st.info("Upload a video file to begin.")
        return

    try:
        capture = open_video_capture(source)
    except RuntimeError as exc:
        st.error(str(exc))
        return

    fps = safe_video_fps(capture)
    pipeline = TrackingPipeline(
        config=_build_config(settings),
        source_name=str(source_name),
        enable_line=bool(settings["enable_line"]),
        enable_zone=bool(settings["enable_zone"]),
        enable_voice=bool(settings["enable_voice"]),
        enable_screenshots=bool(settings["enable_screenshots"]),
        save_output_video=bool(settings["save_output_video"]),
    )

    frame_slot = st.empty()
    frame_count = 0

    try:
        while settings["run"] and capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            result = pipeline.process_frame(frame, source_fps=fps)
            rgb = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
            frame_slot.image(rgb, channels="RGB", use_container_width=True)
            with stats_slot.container():
                _render_stats(result.snapshot.as_dict())

            frame_count += 1
            max_frames = int(settings["max_frames"])
            if max_frames and frame_count >= max_frames:
                break
            time.sleep(0.001)
    finally:
        pipeline.release()
        capture.release()


def main() -> None:
    PATHS.ensure()
    st.set_page_config(page_title="Real-Time Object Detection and Tracking", layout="wide")
    st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)

    st.title("Real-Time Object Detection and Tracking")
    settings = _render_sidebar()

    left, right = st.columns([3.2, 1.2], gap="large")
    stats_slot = right.empty()
        with left:

        if settings["run"]:

            if settings["source_type"] == "Webcam":

                webrtc_streamer(
                    key="object-detection",
                    video_processor_factory=VideoProcessor,
                    media_stream_constraints={
                        "video": True,
                        "audio": False,
                    },
                    async_processing=True,
                )

            else:
                _run_stream(settings, stats_slot)

        else:
            st.markdown(
                '<div class="status-card"><strong>Ready</strong><br>YOLO11 COCO detection with ByteTrack IDs.</div>',
                unsafe_allow_html=True,
            )

            st.image(
                _placeholder_frame(),
                channels="RGB",
                use_container_width=True
            )

    if not settings["run"]:
        with stats_slot.container():
            _render_stats(None)

    
    import numpy as np

    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:] = (9, 13, 20)
    cv2.rectangle(image, (32, 32), (1248, 688), (42, 52, 70), 2)
    cv2.putText(
        image,
        "LIVE FEED",
        (520, 350),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (36, 215, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Start detection from the sidebar",
        (430, 400),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (210, 222, 234),
        1,
        cv2.LINE_AA,
    )
    return image