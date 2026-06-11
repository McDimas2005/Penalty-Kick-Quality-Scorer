from penalty_kick_distancer.config import TargetConfig
from penalty_kick_distancer.scoring import (
    FrameRecord,
    euclidean_distance_px,
    grade_distance,
    pixel_to_cm,
    score_final_visible_ball,
    score_original_area_aware,
    score_tracked_ball,
)


def make_record(frame_index: int, distance_cm: float, area: float = 100.0) -> FrameRecord:
    return FrameRecord(
        frame_index=frame_index,
        timestamp=float(frame_index),
        bbox=(0.0, 0.0, 10.0, 10.0),
        confidence=0.9,
        class_id=0,
        label="Ball",
        center_x=5.0,
        center_y=5.0,
        scoring_x=5.0,
        scoring_y=5.0,
        area=area,
        distance_px=distance_cm,
        distance_cm=distance_cm,
        grade=grade_distance(distance_cm),
    )


def test_euclidean_distance_px():
    assert euclidean_distance_px((0, 0), (3, 4)) == 5


def test_pixel_to_cm():
    assert pixel_to_cm(150, 22 / 150) == 22


def test_grade_distance_bands():
    assert grade_distance(10) == "SSS"
    assert grade_distance(30) == "A"
    assert grade_distance(50) == "B"
    assert grade_distance(100) == "C"
    assert grade_distance(200) == "D"
    assert grade_distance(201) == "F"


def test_original_area_aware_selects_area_closest_to_target():
    target = TargetConfig(x=100, y=100, width=10, height=10)
    records = [make_record(1, 10, area=20), make_record(2, 80, area=95)]
    selected = score_original_area_aware(records, target)
    assert selected is not None
    assert selected.frame_index == 2


def test_final_visible_ball_uses_last_confident_record():
    records = [make_record(1, 40), make_record(2, 20)]
    selected = score_final_visible_ball(records)
    assert selected is not None
    assert selected.frame_index == 2


def test_tracked_ball_uses_smoothed_distance_when_available():
    a = make_record(1, 50)
    b = make_record(2, 60)
    a.smoothed_distance_cm = 15
    b.smoothed_distance_cm = 25
    selected = score_tracked_ball([a, b])
    assert selected is a

