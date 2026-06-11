from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gradio as gr
import pandas as pd

from penalty_kick_distancer.config import SCORING_METHODS, ProcessingConfig, TargetConfig
from penalty_kick_distancer.model_loader import ModelLoadError, load_yolo_model
from penalty_kick_distancer.video_processor import VideoProcessingError, process_video


TITLE = "Penalty-Kick Distancer: YOLO Ball Detection + Target Distance Scoring"
DESCRIPTION = (
    "Upload a penalty kick video, set a target coordinate, and score the kick by measuring "
    "the detected ball's distance from the intended target."
)


def _empty_outputs(message: str):
    summary = pd.DataFrame([{"status": message}])
    return None, None, summary, None, {"status": "error", "message": message}, pd.DataFrame()


def run_detection(
    video_path,
    x_target,
    y_target,
    target_width,
    target_height,
    ball_diameter_cm,
    ball_pixel_diameter,
    confidence_threshold,
    scoring_method,
    frame_stride,
    max_frames,
    save_debug_frames,
):
    if not video_path:
        return _empty_outputs("No video uploaded.")

    try:
        loaded = load_yolo_model(None)
    except ModelLoadError as exc:
        return _empty_outputs(str(exc))

    try:
        config = ProcessingConfig(
            target=TargetConfig(
                x=int(x_target),
                y=int(y_target),
                width=int(target_width),
                height=int(target_height),
            ),
            ball_diameter_cm=float(ball_diameter_cm),
            ball_pixel_diameter=float(ball_pixel_diameter),
            confidence_threshold=float(confidence_threshold),
            scoring_method=scoring_method,
            frame_stride=max(1, int(frame_stride)),
            max_frames=max(1, int(max_frames)),
            save_debug_frames=bool(save_debug_frames),
        )
        output_video, best_frame, metrics, per_frame, metadata, plot_path = process_video(
            video_path,
            loaded.model,
            config,
            fallback_demo_mode=loaded.fallback_demo_mode,
        )
    except (VideoProcessingError, ValueError) as exc:
        return _empty_outputs(str(exc))
    except Exception as exc:
        return _empty_outputs(f"Unexpected processing failure: {exc}")

    metrics["model_source"] = loaded.source
    metrics["model_path"] = loaded.model_path
    metadata["metrics"] = metrics

    summary_fields = [
        "final_grade",
        "best_distance_cm",
        "best_frame_index",
        "detection_confidence",
        "selected_scoring_method",
        "calibration_factor",
        "total_processed_frames",
        "detections",
        "model_source",
        "fallback_demo_mode",
    ]
    summary_df = pd.DataFrame([{key: metrics.get(key) for key in summary_fields}])
    per_frame_df = pd.DataFrame(per_frame)
    return output_video, best_frame, summary_df, plot_path, metadata, per_frame_df


def build_app() -> gr.Blocks:
    css = """
    .method-note {font-size: 0.95rem; line-height: 1.55;}
    .warning-box {border-left: 4px solid #d97706; padding: 0.8rem 1rem; background: #fff7ed;}
    """
    with gr.Blocks(title="Penalty-Kick Distancer", css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {TITLE}\n\n{DESCRIPTION}")

        with gr.Tab("Demo"):
            gr.Markdown(
                "<div class='warning-box'>Short clips are recommended for CPU Spaces. "
                "Increase frame stride or lower max frames for long videos.</div>"
            )
            with gr.Row():
                with gr.Column(scale=4):
                    video_input = gr.Video(label="Penalty kick video")
                    with gr.Row():
                        x_target = gr.Slider(0, 3840, value=960, step=1, label="Target X coordinate")
                        y_target = gr.Slider(0, 2160, value=540, step=1, label="Target Y coordinate")
                    with gr.Row():
                        target_width = gr.Number(value=124, precision=0, label="Target box width (px)")
                        target_height = gr.Number(value=130, precision=0, label="Target box height (px)")
                    with gr.Row():
                        ball_diameter_cm = gr.Number(value=22, label="Ball diameter (cm)")
                        ball_pixel_diameter = gr.Number(value=150, label="Ball pixel diameter")
                    with gr.Row():
                        confidence_threshold = gr.Slider(0.05, 0.95, value=0.4, step=0.05, label="Confidence threshold")
                        scoring_method = gr.Dropdown(
                            choices=list(SCORING_METHODS),
                            value="Tracked ball scoring",
                            label="Scoring method",
                        )
                    with gr.Row():
                        frame_stride = gr.Slider(1, 30, value=2, step=1, label="Frame stride")
                        max_frames = gr.Slider(10, 1500, value=300, step=10, label="Max processed frames")
                    save_debug_frames = gr.Checkbox(value=False, label="Save debug frames")
                    run_button = gr.Button("Run ball detection", variant="primary")
                with gr.Column(scale=5):
                    output_video = gr.Video(label="Annotated result video")
                    best_frame = gr.Image(label="Best frame", type="filepath")

            with gr.Row():
                summary_table = gr.Dataframe(label="Result summary")
            with gr.Row():
                distance_plot = gr.Image(label="Distance-over-time plot", type="filepath")
                metadata_json = gr.JSON(label="JSON metadata")
            per_frame_table = gr.Dataframe(label="Per-frame detections")

            run_button.click(
                fn=run_detection,
                inputs=[
                    video_input,
                    x_target,
                    y_target,
                    target_width,
                    target_height,
                    ball_diameter_cm,
                    ball_pixel_diameter,
                    confidence_threshold,
                    scoring_method,
                    frame_stride,
                    max_frames,
                    save_debug_frames,
                ],
                outputs=[
                    output_video,
                    best_frame,
                    summary_table,
                    distance_plot,
                    metadata_json,
                    per_frame_table,
                ],
            )

        with gr.Tab("Method"):
            gr.Markdown(
                """
                ## Method Overview

                This prototype uses Open Images V7 through FiftyOne to prepare ball-related detection data. The original notebook explored labels including Tennis ball, Football, Volleyball (Ball), Golf ball, Rugby ball, Ball, and Cricket ball. The final dataset workflow remaps those visual sources into one YOLO class: `Ball`.

                A YOLOv8 Nano model is fine-tuned from `yolov8n.pt`, then applied frame-by-frame to uploaded penalty kick videos. For each detected ball, the app draws a bounding box, estimates the ball center, draws the user-defined target rectangle, and measures Euclidean distance from ball to target.

                Pixel distance is converted to centimeters using:

                `calibration_factor = ball_diameter_cm / ball_pixel_diameter`

                The upgraded implementation scores by the ball center by default because it better represents the physical ball location. The notebook's original top-left-coordinate behavior is documented and preserved conceptually in the original area-aware scoring option.

                Limitation: a single camera pixel-to-centimeter conversion is approximate unless the camera geometry is controlled or perspective calibration points are provided.
                """
            )

        with gr.Tab("Comparison"):
            comparison = pd.DataFrame(
                [
                    {
                        "Method": "Original area-aware scoring",
                        "Behavior": "Selects the detection whose box area is closest to the target area, preserving the notebook logic.",
                        "Best use": "Reproducing the original prototype.",
                    },
                    {
                        "Method": "Final visible ball scoring",
                        "Behavior": "Uses the last high-confidence ball detection after the kick.",
                        "Best use": "Simple outcome scoring for short clips.",
                    },
                    {
                        "Method": "Tracked ball scoring",
                        "Behavior": "Applies lightweight temporal smoothing and scores the best smoothed ball-to-target distance.",
                        "Best use": "Default portfolio demo because it reduces frame-to-frame jitter without fragile tracker dependencies.",
                    },
                    {
                        "Method": "Perspective-calibrated scoring",
                        "Behavior": "Experimental placeholder. Falls back safely unless field reference points and homography are provided.",
                        "Best use": "Future physically grounded distance scoring.",
                    },
                ]
            )
            gr.Dataframe(comparison, label="Scoring method comparison", interactive=False)
            gr.Markdown(
                "Recommended default: `Tracked ball scoring`. It keeps the original detection-plus-distance idea while improving video stability."
            )

        with gr.Tab("Deployment Guide"):
            gr.Markdown(
                """
                ## Hugging Face Spaces Deployment

                1. Create a Hugging Face account and a new Gradio Space.
                2. Clone the Space repository locally.
                3. Copy this project into the Space repo.
                4. Put your trained model at `models/best.pt`, or upload it to a separate model repository.
                5. If using a model repository, set `HF_MODEL_REPO_ID` and `HF_MODEL_FILENAME` in Space variables.
                6. Commit and push the files.
                7. Watch the Space build logs and test with a short video.

                ```bash
                pip install -U huggingface_hub
                hf auth login
                git clone https://huggingface.co/spaces/<username>/penalty-kick-distancer
                cd penalty-kick-distancer
                git add .
                git commit -m "Deploy Penalty-Kick Distancer Gradio Space"
                git push
                ```

                Preferred model repository option:

                ```bash
                hf upload <username>/penalty-kick-distancer-yolov8-ball models/best.pt best.pt --repo-type model
                ```
                """
            )

    return demo


if __name__ == "__main__":
    build_app().launch()
