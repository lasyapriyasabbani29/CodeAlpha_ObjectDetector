from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import DetectionConfig, PATHS
from models.detector import Detection, ObjectDetector
from utils.analytics import AnalyticsSnapshot, ObjectAnalytics
from utils.drawing import draw_frame
from utils.events import (
    DetectionEvent,
    DetectionHistoryLogger,
    LineCrossingDetector,
    RestrictedZoneDetector,
    ScreenshotManager,
    VoiceAlertManager,
)
from utils.fps import FPSMeter
from utils.video import OutputVideoWriter


@dataclass(slots=True)
class PipelineResult:
    annotated_frame: np.ndarray
    detections: list[Detection]
    events: list[DetectionEvent]
    snapshot: AnalyticsSnapshot


class TrackingPipeline:
    def __init__(
        self,
        config: DetectionConfig,
        source_name: str = "stream",
        enable_line: bool = True,
        enable_zone: bool = True,
        enable_voice: bool = False,
        enable_screenshots: bool = True,
        save_history: bool = True,
        save_output_video: bool = True,
    ):
        PATHS.ensure()
        self.config = config
        self.source_name = source_name
        self.detector = ObjectDetector(config)
        self.fps = FPSMeter()
        self.analytics = ObjectAnalytics()
        self.line_detector = LineCrossingDetector(
            enabled=enable_line,
            start=config.line_start,
            end=config.line_end,
        )
        self.zone_detector = RestrictedZoneDetector(
            enabled=enable_zone,
            polygon=config.restricted_zone,
        )
        self.voice = VoiceAlertManager(enabled=enable_voice)
        self.screenshots = ScreenshotManager(enabled=enable_screenshots, output_dir=PATHS.screenshots_dir)
        self.history = DetectionHistoryLogger(PATHS.logs_dir / "detection_history.csv", enabled=save_history)
        self.video_writer = OutputVideoWriter(PATHS.outputs_dir, enabled=save_output_video)
        self.recent_events: list[DetectionEvent] = []
        self.frame_index = 0

    def process_frame(self, frame: np.ndarray, source_fps: float = 30.0) -> PipelineResult:
        self.frame_index += 1
        detections, _ = self.detector.track_frame(frame)
        fps_value = self.fps.tick()

        analytics_counts = self.analytics.update(detections)
        line_events = self.line_detector.update(detections, frame.shape)
        zone_events = self.zone_detector.update(detections, frame.shape)
        events = line_events + zone_events
        if events:
            self.recent_events = (events + self.recent_events)[:8]
            self._alert(events)

        annotated = draw_frame(
            frame=frame,
            detections=detections,
            fps=fps_value,
            frame_counts=analytics_counts.frame_counts,
            line_detector=self.line_detector,
            zone_detector=self.zone_detector,
        )

        screenshot_path = self.screenshots.capture(annotated, detections, reason="detection")
        self.video_writer.write(annotated, fps=source_fps, source_name=self.source_name)
        self.history.log_detections(
            source=self.source_name,
            frame_index=self.frame_index,
            fps=fps_value,
            detections=detections,
        )
        self.history.log_events(source=self.source_name, frame_index=self.frame_index, events=events)

        snapshot = AnalyticsSnapshot(
            fps=fps_value,
            current_objects=sum(analytics_counts.frame_counts.values()),
            unique_objects=analytics_counts.total_unique,
            frame_counts=dict(analytics_counts.frame_counts),
            unique_counts=dict(analytics_counts.unique_counts),
            line_crossings=dict(self.line_detector.counts),
            restricted_active=len(self.zone_detector.active_ids),
            events=[event.message for event in self.recent_events],
            output_video=str(self.video_writer.path) if self.video_writer.path else None,
            latest_screenshot=str(screenshot_path) if screenshot_path else None,
        )
        return PipelineResult(annotated, detections, events, snapshot)

    def _alert(self, events: list[DetectionEvent]) -> None:
        for event in events:
            if event.event_type == "restricted_zone":
                self.voice.speak(f"Restricted zone alert. {event.class_name} detected.", key="zone")
            elif event.event_type == "line_crossing":
                self.voice.speak(f"Line crossing. {event.class_name}.", key="line")

    def release(self) -> None:
        self.video_writer.release()
        self.voice.close()

