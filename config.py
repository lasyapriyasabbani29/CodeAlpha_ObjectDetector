from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RuntimePaths:
    base_dir: Path = BASE_DIR
    logs_dir: Path = BASE_DIR / "logs"
    outputs_dir: Path = BASE_DIR / "outputs"
    screenshots_dir: Path = BASE_DIR / "outputs" / "screenshots"
    uploads_dir: Path = BASE_DIR / "outputs" / "uploads"

    def ensure(self) -> None:
        for path in (
            self.logs_dir,
            self.outputs_dir,
            self.screenshots_dir,
            self.uploads_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


PATHS = RuntimePaths()


AVAILABLE_MODELS = {
    "Balanced accuracy - YOLO11m": "yolo11m.pt",
    "High accuracy - YOLO11l": "yolo11l.pt",
    "Maximum accuracy - YOLO11x": "yolo11x.pt",
    "Fast - YOLO11s": "yolo11s.pt",
    "Very fast - YOLO11n": "yolo11n.pt",
    "YOLOv8 medium fallback": "yolov8m.pt",
}


@dataclass
class DetectionConfig:
    model_name: str = "yolo11m.pt"
    tracker_config: str = str(BASE_DIR / "tracker" / "custom_bytetrack.yaml")
    confidence: float = 0.35
    iou: float = 0.55
    imgsz: int = 960
    max_det: int = 300
    device: Optional[str] = None
    half: bool = True
    augment: bool = False
    agnostic_nms: bool = False
    classes: Optional[list[int]] = None
    frame_stride: int = 1
    low_light: bool = False
    line_start: tuple[float, float] = (0.15, 0.62)
    line_end: tuple[float, float] = (0.85, 0.62)
    restricted_zone: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.62, 0.15), (0.95, 0.15), (0.95, 0.92), (0.62, 0.92)]
    )

