"""Lightweight temporal smoothing for ball detections."""

from __future__ import annotations

from collections import deque
from copy import copy
from typing import Sequence

from .config import TargetConfig
from .scoring import FrameRecord, euclidean_distance_px, grade_distance, pixel_to_cm


def smooth_records(
    records: Sequence[FrameRecord],
    target: TargetConfig,
    calibration_factor: float,
    window: int = 5,
) -> list[FrameRecord]:
    """Apply a simple moving average over ball centers.

    This keeps the Space robust on CPU and avoids relying on tracker state that may
    behave differently across environments.
    """

    if window <= 1:
        return [copy(r) for r in records]

    xs: deque[float] = deque(maxlen=window)
    ys: deque[float] = deque(maxlen=window)
    smoothed: list[FrameRecord] = []

    for record in records:
        xs.append(record.scoring_x)
        ys.append(record.scoring_y)
        next_record = copy(record)
        next_record.smoothed_x = sum(xs) / len(xs)
        next_record.smoothed_y = sum(ys) / len(ys)
        distance_px = euclidean_distance_px((next_record.smoothed_x, next_record.smoothed_y), target.center)
        next_record.smoothed_distance_cm = pixel_to_cm(distance_px, calibration_factor)
        next_record.grade = grade_distance(next_record.smoothed_distance_cm)
        smoothed.append(next_record)

    return smoothed

