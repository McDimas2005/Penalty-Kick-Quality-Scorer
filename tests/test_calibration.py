import pytest

from penalty_kick_distancer.calibration import CalibrationError, compute_calibration_factor


def test_compute_calibration_factor():
    assert compute_calibration_factor(22, 150) == pytest.approx(0.1466666667)


def test_invalid_ball_diameter():
    with pytest.raises(CalibrationError):
        compute_calibration_factor(0, 150)


def test_invalid_pixel_diameter():
    with pytest.raises(CalibrationError):
        compute_calibration_factor(22, 0)

