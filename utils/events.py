from __future__ import annotations

import csv
import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from models.detector import Detection


@dataclass(slots=True)
class DetectionEvent:
    timestamp: str
    event_type: str
    track_id: int
    class_name: str
    confidence: float
    message: str


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _scale_point(point: tuple[float, float], shape: tuple[int, ...]) -> tuple[int, int]:
    height, width = shape[:2]
    x, y = point
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return int(x * width), int(y * height)
    return int(x), int(y)


class LineCrossingDetector:
    def __init__(
        self,
        enabled: bool,
        start: tuple[float, float],
        end: tuple[float, float],
        cooldown_frames: int = 12,
    ):
        self.enabled = enabled
        self.start = start
        self.end = end
        self.cooldown_frames = cooldown_frames
        self.last_side: dict[int, int] = {}
        self.last_cross_frame: dict[int, int] = {}
        self.counts: Counter[str] = Counter()
        self.frame_index = 0

    def scaled_line(self, shape: tuple[int, ...]) -> tuple[tuple[int, int], tuple[int, int]]:
        return _scale_point(self.start, shape), _scale_point(self.end, shape)

    def _side(self, center: tuple[int, int], shape: tuple[int, ...]) -> int:
        start, end = self.scaled_line(shape)
        value = (end[0] - start[0]) * (center[1] - start[1]) - (end[1] - start[1]) * (center[0] - start[0])
        if abs(value) < 1e-6:
            return 0
        return 1 if value > 0 else -1

    def update(self, detections: list[Detection], shape: tuple[int, ...]) -> list[DetectionEvent]:
        self.frame_index += 1
        if not self.enabled:
            return []

        events: list[DetectionEvent] = []
        active_ids = {det.track_id for det in detections if det.track_id >= 0}
        for stale_id in set(self.last_side) - active_ids:
            if self.frame_index - self.last_cross_frame.get(stale_id, 0) > self.cooldown_frames * 8:
                self.last_side.pop(stale_id, None)

        for det in detections:
            if det.track_id < 0:
                continue
            current_side = self._side(det.center, shape)
            previous_side = self.last_side.get(det.track_id)
            self.last_side[det.track_id] = current_side

            if previous_side is None or previous_side == 0 or current_side == 0 or previous_side == current_side:
                continue

            last_cross = self.last_cross_frame.get(det.track_id, -10_000)
            if self.frame_index - last_cross < self.cooldown_frames:
                continue

            self.last_cross_frame[det.track_id] = self.frame_index
            self.counts[det.class_name] += 1
            message = f"{det.class_name} ID {det.track_id} crossed line"
            events.append(
                DetectionEvent(
                    timestamp=_now_iso(),
                    event_type="line_crossing",
                    track_id=det.track_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    message=message,
                )
            )
        return events


class RestrictedZoneDetector:
    def __init__(self, enabled: bool, polygon: list[tuple[float, float]]):
        self.enabled = enabled
        self.polygon = polygon
        self.active_ids: set[int] = set()
        self.counts: Counter[str] = Counter()

    def scaled_polygon(self, shape: tuple[int, ...]) -> np.ndarray:
        points = [_scale_point(point, shape) for point in self.polygon]
        return np.array(points, dtype=np.int32)

    def update(self, detections: list[Detection], shape: tuple[int, ...]) -> list[DetectionEvent]:
        if not self.enabled:
            self.active_ids.clear()
            return []

        polygon = self.scaled_polygon(shape)
        events: list[DetectionEvent] = []
        current_inside: set[int] = set()
        for det in detections:
            inside = cv2.pointPolygonTest(polygon, det.center, False) >= 0
            if not inside or det.track_id < 0:
                continue

            current_inside.add(det.track_id)
            if det.track_id in self.active_ids:
                continue

            self.counts[det.class_name] += 1
            message = f"{det.class_name} ID {det.track_id} entered restricted zone"
            events.append(
                DetectionEvent(
                    timestamp=_now_iso(),
                    event_type="restricted_zone",
                    track_id=det.track_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    message=message,
                )
            )

        self.active_ids = current_inside
        return events


class VoiceAlertManager:
    def __init__(self, enabled: bool = False, cooldown_seconds: float = 4.0):
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.last_spoken: dict[str, float] = {}
        self.messages: queue.Queue[str | None] = queue.Queue()
        self.worker: threading.Thread | None = None
        if enabled:
            self.worker = threading.Thread(target=self._run, daemon=True)
            self.worker.start()

    def _run(self) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 168)
        except Exception:
            self.enabled = False
            return

        while True:
            message = self.messages.get()
            if message is None:
                break
            try:
                engine.say(message)
                engine.runAndWait()
            except Exception:
                self.enabled = False
                break

    def speak(self, message: str, key: str) -> None:
        if not self.enabled:
            return
        now = time.time()
        if now - self.last_spoken.get(key, 0.0) < self.cooldown_seconds:
            return
        self.last_spoken[key] = now
        self.messages.put(message)

    def close(self) -> None:
        if self.worker and self.worker.is_alive():
            self.messages.put(None)


class ScreenshotManager:
    def __init__(
        self,
        enabled: bool,
        output_dir: Path,
        cooldown_seconds: float = 2.0,
        min_confidence: float = 0.40,
    ):
        self.enabled = enabled
        self.output_dir = output_dir
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence
        self.last_saved = 0.0
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, frame: np.ndarray, detections: list[Detection], reason: str) -> Path | None:
        if not self.enabled or not detections:
            return None
        if max(det.confidence for det in detections) < self.min_confidence:
            return None
        now = time.time()
        if now - self.last_saved < self.cooldown_seconds:
            return None
        self.last_saved = now
        path = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{reason}.jpg"
        cv2.imwrite(str(path), frame)
        return path


class DetectionHistoryLogger:
    def __init__(self, path: Path, enabled: bool = True, min_interval_seconds: float = 0.5):
        self.path = path
        self.enabled = enabled
        self.min_interval_seconds = min_interval_seconds
        self.last_log_time = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if enabled and not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "timestamp",
                        "source",
                        "frame",
                        "fps",
                        "event_type",
                        "track_id",
                        "class_name",
                        "confidence",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "message",
                    ]
                )

    def log_detections(self, source: str, frame_index: int, fps: float, detections: list[Detection]) -> None:
        if not self.enabled or not detections:
            return
        now = time.time()
        if now - self.last_log_time < self.min_interval_seconds:
            return
        self.last_log_time = now
        with self.path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for det in detections:
                x1, y1, x2, y2 = det.xyxy
                writer.writerow(
                    [
                        _now_iso(),
                        source,
                        frame_index,
                        round(fps, 2),
                        "detection",
                        det.track_id,
                        det.class_name,
                        round(det.confidence, 4),
                        x1,
                        y1,
                        x2,
                        y2,
                        "",
                    ]
                )

    def log_events(self, source: str, frame_index: int, events: list[DetectionEvent]) -> None:
        if not self.enabled or not events:
            return
        with self.path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for event in events:
                writer.writerow(
                    [
                        event.timestamp,
                        source,
                        frame_index,
                        "",
                        event.event_type,
                        event.track_id,
                        event.class_name,
                        round(event.confidence, 4),
                        "",
                        "",
                        "",
                        "",
                        event.message,
                    ]
                )

