"""Shared configuration objects for the penalty-kick video pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DEFAULT_THRESHOLDS = {
    "SSS": 10.0,
    "A": 30.0,
    "B": 50.0,
    "C": 100.0,
    "D": 200.0,
}

SCORING_METHODS = (
    "Original area-aware scoring",
    "Final visible ball scoring",
    "Tracked ball scoring",
    "Perspective-calibrated scoring",
)

ScoringMethod = Literal[
    "Original area-aware scoring",
    "Final visible ball scoring",
    "Tracked ball scoring",
    "Perspective-calibrated scoring",
]


@dataclass(frozen=True)
class TargetConfig:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> float:
        return float(max(self.width, 0) * max(self.height, 0))

    @property
    def top_left(self) -> tuple[int, int]:
        return self.x - self.width // 2, self.y - self.height // 2

    @property
    def bottom_right(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    @property
    def center(self) -> tuple[int, int]:
        return self.x, self.y


@dataclass(frozen=True)
class ProcessingConfig:
    target: TargetConfig
    ball_diameter_cm: float = 22.0
    ball_pixel_diameter: float = 150.0
    confidence_threshold: float = 0.4
    scoring_method: ScoringMethod = "Tracked ball scoring"
    frame_stride: int = 1
    max_frames: int = 300
    save_debug_frames: bool = False
    use_center_point: bool = True
    thresholds: dict[str, float] = field(default_factory=lambda: DEFAULT_THRESHOLDS.copy())
    output_dir: Path | None = None

