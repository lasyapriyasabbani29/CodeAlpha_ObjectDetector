from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import cv2
import numpy as np

from models.detector import Detection
from utils.colors import color_for_class

if TYPE_CHECKING:
    from utils.events import LineCrossingDetector, RestrictedZoneDetector


def _put_text_with_bg(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x, y - th - baseline - 7), (x + tw + 9, y + 4), (12, 16, 24), -1)
    cv2.rectangle(frame, (x, y - th - baseline - 7), (x + tw + 9, y + 4), color, 1)
    cv2.putText(frame, text, (x + 5, y - 5), font, scale, (240, 245, 250), thickness, cv2.LINE_AA)


def _draw_zone(frame: np.ndarray, zone_detector: "RestrictedZoneDetector") -> None:
    if not zone_detector.enabled:
        return
    points = zone_detector.scaled_polygon(frame.shape)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], (35, 38, 80))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [points], isClosed=True, color=(92, 82, 255), thickness=2)
    cv2.putText(
        frame,
        "RESTRICTED",
        tuple(points[0]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_line(frame: np.ndarray, line_detector: "LineCrossingDetector") -> None:
    if not line_detector.enabled:
        return
    start, end = line_detector.scaled_line(frame.shape)
    cv2.line(frame, start, end, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, start, 5, (0, 220, 255), -1)
    cv2.circle(frame, end, 5, (0, 220, 255), -1)


def draw_frame(
    frame: np.ndarray,
    detections: list[Detection],
    fps: float,
    frame_counts: Counter[str],
    line_detector: "LineCrossingDetector",
    zone_detector: "RestrictedZoneDetector",
) -> np.ndarray:
    annotated = frame.copy()
    _draw_zone(annotated, zone_detector)
    _draw_line(annotated, line_detector)

    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        color = color_for_class(det.class_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, det.center, 3, color, -1)
        track_label = f"ID {det.track_id}" if det.track_id >= 0 else "ID --"
        label = f"{track_label}  {det.class_name}  {det.confidence * 100:.0f}%"
        _put_text_with_bg(annotated, label, (x1, max(y1 - 6, 20)), color)

    panel_h = 86
    overlay = annotated.copy()
    cv2.rectangle(overlay, (12, 12), (360, panel_h), (8, 12, 18), -1)
    cv2.addWeighted(overlay, 0.72, annotated, 0.28, 0, annotated)
    cv2.rectangle(annotated, (12, 12), (360, panel_h), (55, 65, 85), 1)
    cv2.putText(annotated, f"FPS {fps:.1f}", (28, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 230, 170), 2, cv2.LINE_AA)
    cv2.putText(
        annotated,
        f"Visible objects {sum(frame_counts.values())}",
        (28, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 240, 246),
        1,
        cv2.LINE_AA,
    )
    return annotated

