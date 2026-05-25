from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def open_video_capture(source: Any) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return capture


def safe_video_fps(capture: cv2.VideoCapture, fallback: float = 30.0) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1.0 or fps > 240.0:
        return fallback
    return fps


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    mean_luma = float(np.mean(enhanced_l))
    if mean_luma < 90.0:
        gamma = 0.72
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        enhanced_bgr = cv2.LUT(enhanced_bgr, table)
    return enhanced_bgr


def _safe_stem(source_name: str) -> str:
    stem = Path(str(source_name)).stem or "stream"
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", stem)[:80]


class OutputVideoWriter:
    def __init__(self, output_dir: Path, enabled: bool = True):
        self.output_dir = output_dir
        self.enabled = enabled
        self.writer: cv2.VideoWriter | None = None
        self.path: Path | None = None
        self.frame_size: tuple[int, int] | None = None
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, frame: np.ndarray, fps: float, source_name: str) -> None:
        if not self.enabled:
            return
        if self.writer is None:
            height, width = frame.shape[:2]
            self.frame_size = (width, height)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path = self.output_dir / f"{_safe_stem(source_name)}_{timestamp}_tracked.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc, max(float(fps), 1.0), self.frame_size)

        self.writer.write(frame)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

