"""OpenCV video processing pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .calibration import compute_calibration_factor, perspective_calibration_placeholder
from .config import ProcessingConfig
from .scoring import (
    FrameRecord,
    euclidean_distance_px,
    grade_distance,
    pixel_to_cm,
    score_final_visible_ball,
    score_original_area_aware,
    score_tracked_ball,
    select_detection,
    summarize_score,
)
from .tracking import smooth_records
from .utils import ensure_dir, make_run_dir
from .visualization import annotate_frame, create_distance_plot, save_best_frame


class VideoProcessingError(RuntimeError):
    """Raised for recoverable video processing failures."""


BALL_LABEL_TERMS = {
    "ball",
    "sports ball",
    "tennis ball",
    "football",
    "volleyball",
    "golf ball",
    "rugby ball",
    "cricket ball",
    "soccer ball",
}


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise VideoProcessingError("opencv-python-headless is required for video processing.") from exc
    return cv2


def _parse_yolo_results(results: Any, confidence_threshold: float) -> list[dict[str, object]]:
    """Convert Ultralytics results or test doubles to a plain detection list."""

    if results is None:
        return []
    result = results[0] if isinstance(results, (list, tuple)) else results
    names = getattr(result, "names", {0: "Ball"}) or {0: "Ball"}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    raw_rows = boxes
    if hasattr(boxes, "data"):
        raw_rows = boxes.data
    if hasattr(raw_rows, "cpu"):
        raw_rows = raw_rows.cpu()
    if hasattr(raw_rows, "numpy"):
        raw_rows = raw_rows.numpy()
    if hasattr(raw_rows, "tolist"):
        raw_rows = raw_rows.tolist()

    detections = []
    for row in raw_rows:
        if len(row) < 6:
            continue
        x1, y1, x2, y2, score, class_id = row[:6]
        score = float(score)
        if score < confidence_threshold:
            continue
        class_id = int(class_id)
        label = names.get(class_id, "Ball") if isinstance(names, dict) else "Ball"
        if isinstance(names, dict) and len(names) > 1 and str(label).strip().lower() not in BALL_LABEL_TERMS:
            continue
        area = max(float(x2) - float(x1), 0.0) * max(float(y2) - float(y1), 0.0)
        detections.append(
            {
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "confidence": score,
                "class_id": class_id,
                "label": str(label),
                "area": area,
            }
        )
    return detections


def _record_from_detection(
    detection: dict[str, object],
    frame_index: int,
    timestamp: float,
    config: ProcessingConfig,
    calibration_factor: float,
) -> FrameRecord:
    x1, y1, x2, y2 = detection["bbox"]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    scoring_x, scoring_y = (center_x, center_y) if config.use_center_point else (x1, y1)
    distance_px = euclidean_distance_px((scoring_x, scoring_y), config.target.center)
    distance_cm = pixel_to_cm(distance_px, calibration_factor)
    return FrameRecord(
        frame_index=frame_index,
        timestamp=timestamp,
        bbox=(x1, y1, x2, y2),
        confidence=float(detection["confidence"]),
        class_id=int(detection["class_id"]),
        label=str(detection["label"]),
        center_x=center_x,
        center_y=center_y,
        scoring_x=scoring_x,
        scoring_y=scoring_y,
        area=float(detection["area"]),
        distance_px=distance_px,
        distance_cm=distance_cm,
        grade=grade_distance(distance_cm, config.thresholds),
    )


def _select_final_record(records: list[FrameRecord], config: ProcessingConfig, calibration_factor: float):
    warnings: list[str] = []
    method = config.scoring_method

    if method == "Original area-aware scoring":
        selected = score_original_area_aware(records, config.target)
    elif method == "Final visible ball scoring":
        selected = score_final_visible_ball(records, config.confidence_threshold)
    elif method == "Tracked ball scoring":
        records = smooth_records(records, config.target, calibration_factor)
        selected = score_tracked_ball(records)
    elif method == "Perspective-calibrated scoring":
        warnings.append(perspective_calibration_placeholder()["message"])
        selected = score_final_visible_ball(records, config.confidence_threshold)
    else:
        warnings.append(f"Unknown scoring method '{method}'. Used final visible ball scoring.")
        selected = score_final_visible_ball(records, config.confidence_threshold)

    return records, selected, warnings


def process_video(
    video_path: str | Path,
    model: object,
    config: ProcessingConfig,
    fallback_demo_mode: bool = False,
) -> tuple[str, str | None, dict[str, object], list[dict[str, object]], dict[str, object], str | None]:
    """Process a video and return artifacts for the Gradio app."""

    cv2 = _import_cv2()
    video_path = Path(video_path)
    if not video_path.exists():
        raise VideoProcessingError(f"Video file does not exist: {video_path}")

    if config.frame_stride < 1:
        raise VideoProcessingError("Frame stride must be at least 1.")
    if config.max_frames < 1:
        raise VideoProcessingError("Max frames must be at least 1.")

    calibration_factor = compute_calibration_factor(config.ball_diameter_cm, config.ball_pixel_diameter)
    output_dir = ensure_dir(config.output_dir or make_run_dir())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoProcessingError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise VideoProcessingError("Could not read video dimensions.")

    output_video_path = output_dir / "annotated_output.mp4"
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise VideoProcessingError("Could not initialize MP4 video writer.")

    frame_index = 0
    processed_count = 0
    records: list[FrameRecord] = []
    frame_cache: dict[int, Any] = {}
    latest_record: FrameRecord | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            should_infer = frame_index % config.frame_stride == 0 and processed_count < config.max_frames
            current_record = latest_record
            if should_infer:
                result = model(frame, verbose=False) if callable(model) else model.predict(frame, verbose=False)
                detections = _parse_yolo_results(result, config.confidence_threshold)
                selected_detection = select_detection(
                    detections,
                    config.confidence_threshold,
                    target=config.target,
                    prefer_area=config.scoring_method == "Original area-aware scoring",
                )
                if selected_detection is not None:
                    timestamp = frame_index / float(fps)
                    current_record = _record_from_detection(
                        selected_detection,
                        frame_index,
                        timestamp,
                        config,
                        calibration_factor,
                    )
                    records.append(current_record)
                    latest_record = current_record
                    if config.save_debug_frames or len(frame_cache) < 30:
                        frame_cache[frame_index] = frame.copy()
                processed_count += 1

            annotated = annotate_frame(
                frame,
                config.target,
                current_record,
                latest_record,
                config.scoring_method,
                fallback_demo_mode=fallback_demo_mode,
            )
            writer.write(annotated)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    records, selected_record, warnings = _select_final_record(records, config, calibration_factor)

    best_frame_path = None
    if selected_record is not None:
        best_frame = frame_cache.get(selected_record.frame_index)
        if best_frame is None:
            best_frame = _read_frame(video_path, selected_record.frame_index)
        if best_frame is not None:
            annotated_best = annotate_frame(
                best_frame,
                config.target,
                selected_record,
                selected_record,
                config.scoring_method,
                fallback_demo_mode=fallback_demo_mode,
            )
            best_frame_path = save_best_frame(annotated_best, output_dir / "best_frame.jpg")

    plot_path = create_distance_plot(records, output_dir / "distance_over_time.png")

    metrics = summarize_score(selected_record, config.scoring_method, calibration_factor, config.thresholds)
    metrics.update(
        {
            "total_video_frames": total_frames,
            "total_read_frames": frame_index,
            "total_processed_frames": processed_count,
            "detections": len(records),
            "frame_stride": config.frame_stride,
            "fps": round(float(fps), 3),
            "video_width": width,
            "video_height": height,
            "fallback_demo_mode": fallback_demo_mode,
            "warnings": warnings,
        }
    )
    if not records:
        metrics["warnings"] = warnings + ["No ball was detected above the confidence threshold."]

    per_frame = [record.to_dict() for record in records]
    metadata = {
        "video_path": str(video_path),
        "output_video_path": str(output_video_path),
        "best_frame_path": best_frame_path,
        "plot_path": plot_path,
        "metrics": metrics,
        "target": {
            "x": config.target.x,
            "y": config.target.y,
            "width": config.target.width,
            "height": config.target.height,
        },
        "calibration": {
            "ball_diameter_cm": config.ball_diameter_cm,
            "ball_pixel_diameter": config.ball_pixel_diameter,
            "calibration_factor": calibration_factor,
        },
        "per_frame": per_frame,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return str(output_video_path), best_frame_path, metrics, per_frame, metadata, plot_path


def _read_frame(video_path: Path, frame_index: int):
    cv2 = _import_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None
