from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from models.detector import Detection


@dataclass(slots=True)
class AnalyticsCounts:
    frame_counts: Counter[str]
    unique_counts: Counter[str]
    total_unique: int


@dataclass(slots=True)
class AnalyticsSnapshot:
    fps: float
    current_objects: int
    unique_objects: int
    frame_counts: dict[str, int]
    unique_counts: dict[str, int]
    line_crossings: dict[str, int]
    restricted_active: int
    events: list[str]
    output_video: str | None
    latest_screenshot: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "fps": round(self.fps, 2),
            "current_objects": self.current_objects,
            "unique_objects": self.unique_objects,
            "frame_counts": self.frame_counts,
            "unique_counts": self.unique_counts,
            "line_crossings": self.line_crossings,
            "restricted_active": self.restricted_active,
            "events": self.events,
            "output_video": self.output_video,
            "latest_screenshot": self.latest_screenshot,
        }


class ObjectAnalytics:
    def __init__(self):
        self.seen_tracks: defaultdict[str, set[int]] = defaultdict(set)
        self.synthetic_seen: Counter[str] = Counter()

    def update(self, detections: list[Detection]) -> AnalyticsCounts:
        frame_counts: Counter[str] = Counter(det.class_name for det in detections)
        for det in detections:
            if det.track_id >= 0:
                self.seen_tracks[det.class_name].add(det.track_id)
            else:
                self.synthetic_seen[det.class_name] = max(self.synthetic_seen[det.class_name], frame_counts[det.class_name])

        unique_counts = Counter(
            {
                class_name: max(len(track_ids), self.synthetic_seen[class_name])
                for class_name, track_ids in self.seen_tracks.items()
            }
        )
        for class_name, count in self.synthetic_seen.items():
            unique_counts[class_name] = max(unique_counts[class_name], count)

        return AnalyticsCounts(
            frame_counts=frame_counts,
            unique_counts=unique_counts,
            total_unique=sum(unique_counts.values()),
        )

