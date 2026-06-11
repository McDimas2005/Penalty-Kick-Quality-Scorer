"""OpenCV drawing and plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import TargetConfig
from .scoring import FrameRecord


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for video visualization.") from exc
    return cv2


def draw_label_box(frame, text: str, origin: tuple[int, int], color: tuple[int, int, int]):
    cv2 = _import_cv2()
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(frame.shape[1], frame.shape[0]) / 1600)
    thickness = max(1, int(round(scale * 2)))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + 8, y)
    cv2.rectangle(frame, (x - 4, y - th - baseline - 6), (x + tw + 6, y + baseline + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_target(frame, target: TargetConfig):
    cv2 = _import_cv2()
    x1, y1 = target.top_left
    x2, y2 = target.bottom_right
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 80), 2)
    cv2.drawMarker(frame, target.center, (40, 220, 80), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
    draw_label_box(frame, "TARGET", (max(0, x1), max(20, y1 - 8)), (40, 220, 80))


def draw_ball(frame, record: FrameRecord | None, target: TargetConfig, is_best: bool = False):
    if record is None:
        return
    cv2 = _import_cv2()
    x1, y1, x2, y2 = [int(round(v)) for v in record.bbox]
    color = (45, 170, 255) if is_best else (255, 190, 60)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    center = (int(round(record.center_x)), int(round(record.center_y)))
    cv2.circle(frame, center, 4, color, -1)
    cv2.line(frame, center, target.center, (255, 255, 255), 1, cv2.LINE_AA)
    label = f"{record.label} {record.confidence:.2f}"
    draw_label_box(frame, label, (max(0, x1), max(20, y1 - 8)), color)


def annotate_frame(
    frame,
    target: TargetConfig,
    record: FrameRecord | None,
    selected_record: FrameRecord | None,
    method: str,
    fallback_demo_mode: bool = False,
):
    draw_target(frame, target)
    draw_ball(frame, record, target, selected_record is not None and record is selected_record)

    if selected_record is None:
        summary = "No ball detected"
        grade = "Grade: n/a"
    else:
        distance = (
            selected_record.smoothed_distance_cm
            if selected_record.smoothed_distance_cm is not None
            else selected_record.distance_cm
        )
        summary = f"Distance: {distance:.2f} cm"
        grade = f"Grade: {selected_record.grade}"

    draw_label_box(frame, summary, (12, 32), (255, 255, 255))
    draw_label_box(frame, grade, (12, 64), (255, 255, 255))
    draw_label_box(frame, f"Method: {method}", (12, 96), (220, 220, 220))
    if record is not None:
        draw_label_box(
            frame,
            f"Frame {record.frame_index} | ({record.center_x:.0f}, {record.center_y:.0f})",
            (12, 128),
            (220, 220, 220),
        )
    if fallback_demo_mode:
        draw_label_box(frame, "Fallback model mode", (12, frame.shape[0] - 18), (80, 220, 255))
    return frame


def save_best_frame(frame, output_path: str | Path) -> str:
    cv2 = _import_cv2()
    output_path = str(output_path)
    ok = cv2.imwrite(output_path, frame)
    if not ok:
        raise RuntimeError(f"Could not write best frame image to {output_path}.")
    return output_path


def create_distance_plot(records: Sequence[FrameRecord], output_path: str | Path) -> str | None:
    if not records:
        return None

    frames = [r.frame_index for r in records]
    distances = [
        r.smoothed_distance_cm if r.smoothed_distance_cm is not None else r.distance_cm
        for r in records
    ]

    plt.figure(figsize=(8, 4.2))
    plt.plot(frames, distances, color="#1f77b4", linewidth=2)
    plt.scatter(frames, distances, s=18, color="#ff7f0e")
    plt.xlabel("Frame index")
    plt.ylabel("Distance to target (cm)")
    plt.title("Ball distance over processed frames")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    output_path = str(output_path)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path

