from __future__ import annotations

import argparse
import time

import cv2

from config import DetectionConfig, PATHS
from tracker.tracking_pipeline import TrackingPipeline
from utils.video import open_video_capture, safe_video_fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time YOLO11 object detection and tracking")
    parser.add_argument("--source", default="0", help="Webcam index, video path, RTSP URL, or image/video source")
    parser.add_argument("--model", default="yolo11m.pt", help="YOLO model, for example yolo11m.pt or best.pt")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.55, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size")
    parser.add_argument("--device", default=None, help="Use auto, cpu, 0, cuda:0, or mps")
    parser.add_argument("--no-display", action="store_true", help="Run headless without cv2.imshow")
    parser.add_argument("--save-video", action="store_true", help="Save annotated output video")
    parser.add_argument("--screenshots", action="store_true", help="Capture screenshots when objects are detected")
    parser.add_argument("--voice", action="store_true", help="Enable pyttsx3 voice alerts")
    parser.add_argument("--low-light", action="store_true", help="Apply low-light contrast enhancement before inference")
    parser.add_argument("--disable-line", action="store_true", help="Disable line crossing detection")
    parser.add_argument("--disable-zone", action="store_true", help="Disable restricted zone detection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PATHS.ensure()

    config = DetectionConfig(
        model_name=args.model,
        confidence=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        low_light=args.low_light,
    )
    source = int(args.source) if str(args.source).isdigit() else args.source
    capture = open_video_capture(source)
    fps = safe_video_fps(capture)

    pipeline = TrackingPipeline(
        config=config,
        source_name=str(args.source),
        enable_line=not args.disable_line,
        enable_zone=not args.disable_zone,
        enable_voice=args.voice,
        enable_screenshots=args.screenshots,
        save_output_video=args.save_video,
    )

    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break

            result = pipeline.process_frame(frame, source_fps=fps)

            if not args.no_display:
                cv2.imshow("Professional Object Detection and Tracking", result.annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.001)
    finally:
        pipeline.release()
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

