from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ultralytics import YOLO

from config import DetectionConfig
from utils.video import enhance_low_light

try:
    import torch
except Exception:  # pragma: no cover - torch is installed through requirements in normal use.
    torch = None


@dataclass(slots=True)
class Detection:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[int, int, int, int]
    center: tuple[int, int]


def resolve_device(requested: str | None = None) -> str:
    if requested and requested.lower() != "auto":
        return requested
    if torch is not None and torch.cuda.is_available():
        return "0"
    if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_cuda_device(device: str) -> bool:
    normalized = str(device).lower()
    return normalized.isdigit() or normalized.startswith("cuda")


class ObjectDetector:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.use_half = bool(config.half and _is_cuda_device(self.device))
        self.model = YOLO(config.model_name)
        self.names = self.model.names

    def track_frame(self, frame: np.ndarray) -> tuple[list[Detection], Any]:
        frame_for_model = enhance_low_light(frame) if self.config.low_light else frame
        kwargs: dict[str, Any] = {
            "source": frame_for_model,
            "persist": True,
            "tracker": self.config.tracker_config,
            "conf": self.config.confidence,
            "iou": self.config.iou,
            "imgsz": self.config.imgsz,
            "max_det": self.config.max_det,
            "device": self.device,
            "half": self.use_half,
            "augment": self.config.augment,
            "agnostic_nms": self.config.agnostic_nms,
            "verbose": False,
        }
        if self.config.classes:
            kwargs["classes"] = self.config.classes

        results = self.model.track(**kwargs)
        result = results[0] if isinstance(results, list) else results
        return self._parse_result(result), result

    def _class_name(self, class_id: int) -> str:
        if isinstance(self.names, dict):
            return str(self.names.get(class_id, class_id))
        if 0 <= class_id < len(self.names):
            return str(self.names[class_id])
        return str(class_id)

    def _parse_result(self, result: Any) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy_values = boxes.xyxy.cpu().numpy().astype(int)
        conf_values = boxes.conf.cpu().numpy()
        class_values = boxes.cls.cpu().numpy().astype(int)
        if boxes.id is not None:
            track_values = boxes.id.cpu().numpy().astype(int)
        else:
            track_values = np.full(len(xyxy_values), -1, dtype=int)

        detections: list[Detection] = []
        for xyxy, conf, cls_id, track_id in zip(xyxy_values, conf_values, class_values, track_values):
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            detections.append(
                Detection(
                    track_id=int(track_id),
                    class_id=int(cls_id),
                    class_name=self._class_name(int(cls_id)),
                    confidence=float(conf),
                    xyxy=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                )
            )
        return detections

