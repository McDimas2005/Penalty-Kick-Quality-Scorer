"""Calibration helpers."""

from __future__ import annotations


class CalibrationError(ValueError):
    """Raised when pixel-to-real-world calibration values are invalid."""


def compute_calibration_factor(ball_diameter_cm: float, ball_pixel_diameter: float) -> float:
    """Return centimeters per pixel from a known ball size.

    The original notebook used `ball_diameter_cm / ball_pixel_diameter`, for example
    `22 / 150`. This single scale factor is approximate for perspective video.
    """

    if ball_diameter_cm <= 0:
        raise CalibrationError("Ball diameter in centimeters must be greater than zero.")
    if ball_pixel_diameter <= 0:
        raise CalibrationError("Ball pixel diameter must be greater than zero.")
    return float(ball_diameter_cm) / float(ball_pixel_diameter)


def perspective_calibration_placeholder() -> dict[str, str]:
    """Describe the optional homography path without claiming it is implemented."""

    return {
        "status": "not_implemented",
        "message": (
            "Perspective-calibrated scoring requires manually selected field reference "
            "points and a homography. This Space exposes the method as an experimental "
            "option, then safely falls back to calibrated pixel-distance scoring."
        ),
    }

