from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
from flask import Flask, Response, jsonify, redirect, render_template, request, stream_with_context, url_for
from werkzeug.datastructures import MultiDict

from config import AVAILABLE_MODELS, DetectionConfig, PATHS
from tracker.tracking_pipeline import TrackingPipeline
from utils.video import open_video_capture, safe_video_fps


app = Flask(__name__, template_folder="ui/templates", static_folder="ui/static")
PATHS.ensure()

_stats_lock = threading.Lock()
_latest_stats: dict[str, Any] = {
    "fps": 0.0,
    "current_objects": 0,
    "unique_objects": 0,
    "frame_counts": {},
    "unique_counts": {},
    "line_crossings": {},
    "restricted_active": 0,
    "output_video": None,
    "events": [],
}


def _update_stats(snapshot: dict[str, Any]) -> None:
    with _stats_lock:
        _latest_stats.update(snapshot)


def _bool_arg(args: MultiDict, name: str, default: bool = False) -> bool:
    raw = args.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _build_config(args: MultiDict) -> DetectionConfig:
    return DetectionConfig(
        model_name=args.get("model", "yolo11m.pt"),
        confidence=float(args.get("conf", 0.35)),
        iou=float(args.get("iou", 0.55)),
        imgsz=int(args.get("imgsz", 960)),
        device=args.get("device") or None,
        low_light=_bool_arg(args, "low_light", False),
    )


def _frame_generator(source: str, config: DetectionConfig, options: dict[str, bool]):
    capture_source = int(source) if str(source).isdigit() else source
    capture = open_video_capture(capture_source)
    fps = safe_video_fps(capture)
    pipeline = TrackingPipeline(
        config=config,
        source_name=str(source),
        enable_line=options["line"],
        enable_zone=options["zone"],
        enable_voice=options["voice"],
        enable_screenshots=options["screenshots"],
        save_output_video=options["save_video"],
    )

    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break

            result = pipeline.process_frame(frame, source_fps=fps)
            _update_stats(result.snapshot.as_dict())

            ok, jpg = cv2.imencode(".jpg", result.annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            )
    finally:
        pipeline.release()
        capture.release()


@app.route("/")
def index():
    uploaded_videos = sorted(PATHS.uploads_dir.glob("*"))
    return render_template(
        "flask_index.html",
        models=AVAILABLE_MODELS,
        uploads=[{"name": path.name, "path": str(path)} for path in uploaded_videos if path.is_file()],
    )


@app.route("/video_feed")
def video_feed():
    args = request.args.copy()
    source = args.get("source", "0")
    config = _build_config(args)
    options = {
        "line": _bool_arg(args, "line", True),
        "zone": _bool_arg(args, "zone", True),
        "voice": _bool_arg(args, "voice", False),
        "screenshots": _bool_arg(args, "screenshots", False),
        "save_video": _bool_arg(args, "save_video", True),
    }
    frames = stream_with_context(_frame_generator(source, config, options))
    return Response(frames, mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    with _stats_lock:
        return jsonify(_latest_stats)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if not file or not file.filename:
        return redirect(url_for("index"))

    target = PATHS.uploads_dir / Path(file.filename).name
    file.save(target)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
