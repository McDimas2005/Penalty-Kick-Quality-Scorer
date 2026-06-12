"""Distance and scoring strategies for ball-to-target evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence

from .config import DEFAULT_THRESHOLDS, TargetConfig


@dataclass
class FrameRecord:
    frame_index: int
    timestamp: float
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str
    center_x: float
    center_y: float
    scoring_x: float
    scoring_y: float
    area: float
    distance_px: float
    distance_cm: float
    grade: str
    smoothed_x: float | None = None
    smoothed_y: float | None = None
    smoothed_distance_cm: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "bbox": [round(v, 3) for v in self.bbox],
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "label": self.label,
            "center_x": round(self.center_x, 3),
            "center_y": round(self.center_y, 3),
            "scoring_x": round(self.scoring_x, 3),
            "scoring_y": round(self.scoring_y, 3),
            "area": round(self.area, 3),
            "distance_px": round(self.distance_px, 3),
            "distance_cm": round(self.distance_cm, 3),
            "grade": self.grade,
            "smoothed_x": None if self.smoothed_x is None else round(self.smoothed_x, 3),
            "smoothed_y": None if self.smoothed_y is None else round(self.smoothed_y, 3),
            "smoothed_distance_cm": (
                None if self.smoothed_distance_cm is None else round(self.smoothed_distance_cm, 3)
            ),
        }


def euclidean_distance_px(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    """Compute Euclidean distance in pixels between two 2D points."""

    ax, ay = point_a
    bx, by = point_b
    return sqrt((float(ax) - float(bx)) ** 2 + (float(ay) - float(by)) ** 2)


def pixel_to_cm(distance_px: float, calibration_factor: float) -> float:
    """Convert a pixel distance to centimeters."""

    if calibration_factor <= 0:
        raise ValueError("Calibration factor must be greater than zero.")
    return float(distance_px) * float(calibration_factor)


def grade_distance(distance_cm: float, thresholds: dict[str, float] | None = None) -> str:
    """Map a distance in centimeters to the notebook-style grade band."""

    t = thresholds or DEFAULT_THRESHOLDS
    if distance_cm <= t["SSS"]:
        return "SSS"
    if distance_cm <= t["A"]:
        return "A"
    if distance_cm <= t["B"]:
        return "B"
    if distance_cm <= t["C"]:
        return "C"
    if distance_cm <= t["D"]:
        return "D"
    return "F"


def select_detection(
    detections: Iterable[dict[str, object]],
    confidence_threshold: float,
    target: TargetConfig | None = None,
    prefer_area: bool = False,
) -> dict[str, object] | None:
    """Select one ball detection from a frame.

    By default this mirrors the notebook's "highest confidence only" behavior.
    The original area-aware mode can instead favor the detection whose bounding-box
    area is closest to the target rectangle area.
    """

    candidates = [d for d in detections if float(d.get("confidence", 0.0)) >= confidence_threshold]
    if not candidates:
        return None

    if prefer_area and target is not None:
        return min(
            candidates,
            key=lambda d: (
                abs(float(d.get("area", 0.0)) - target.area),
                -float(d.get("confidence", 0.0)),
            ),
        )
    return max(candidates, key=lambda d: float(d.get("confidence", 0.0)))


def score_original_area_aware(records: Sequence[FrameRecord], target: TargetConfig) -> FrameRecord | None:
    """Choose the record whose ball-box area is closest to the target area.

    This preserves the notebook's main scoring idea. The upgraded app uses the ball
    center as the scoring point by default, while documenting that the notebook used
    the top-left bounding-box coordinate.
    """

    if not records:
        return None
    return min(records, key=lambda r: (abs(r.area - target.area), r.distance_cm))


def score_final_visible_ball(
    records: Sequence[FrameRecord],
    min_confidence: float = 0.4,
    stable_tail: int = 5,
) -> FrameRecord | None:
    """Use the final stable/high-confidence visible ball detection."""

    if not records:
        return None
    candidates = [r for r in records if r.confidence >= min_confidence]
    if not candidates:
        return None
    return candidates[-stable_tail:][-1]


def score_tracked_ball(records: Sequence[FrameRecord]) -> FrameRecord | None:
    """Use temporally smoothed detections and select the best smoothed distance."""

    if not records:
        return None
    return min(records, key=lambda r: r.smoothed_distance_cm if r.smoothed_distance_cm is not None else r.distance_cm)


def summarize_score(
    record: FrameRecord | None,
    method: str,
    calibration_factor: float,
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    """Create a compact summary dictionary for UI tables and JSON output."""

    if record is None:
        return {
            "final_grade": "No detection",
            "best_distance_cm": None,
            "best_frame_index": None,
            "detection_confidence": None,
            "selected_scoring_method": method,
            "calibration_factor": calibration_factor,
        }

    distance_cm = record.smoothed_distance_cm if record.smoothed_distance_cm is not None else record.distance_cm
    return {
        "final_grade": grade_distance(distance_cm, thresholds),
        "best_distance_cm": round(distance_cm, 3),
        "best_frame_index": record.frame_index,
        "detection_confidence": round(record.confidence, 4),
        "selected_scoring_method": method,
        "calibration_factor": round(calibration_factor, 6),
    }
