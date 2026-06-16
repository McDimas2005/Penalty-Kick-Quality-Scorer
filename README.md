---
title: Penalty-Kick Distancer
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.45.0
python_version: 3.12
app_file: app.py
pinned: false
license: mit
suggested_hardware: cpu-basic
---

# Penalty-Kick Distancer

Penalty-Kick Distancer is a self-study sports computer vision prototype that detects a ball in penalty-kick video and scores the kick by measuring the detected ball's distance from a user-defined target coordinate.

[Live Demo](https://huggingface.co/spaces/TsukishimaAlan20/penalty-kick-distancer)

## Overview

This project turns a YOLOv8 ball detector into a small applied scoring system for penalty-kick analysis. The app accepts a video, lets the user define the intended target coordinate, runs ball detection frame by frame, estimates the ball-to-target distance, and returns an annotated video with scoring metadata.

The deployment is designed for Hugging Face Spaces using Gradio on free CPU Basic hardware. Python 3.12 and Gradio 5.45.0 are pinned in the Space metadata for deployment stability.

## Why This Project Matters

Raw object detection is only the first step in a useful sports-CV workflow. Penalty-Kick Distancer connects visual detection to measurable feedback: where the ball was detected relative to an intended target, how far away it was in calibrated units, and which quality grade that distance maps to.

The portfolio value is the full pipeline: dataset preparation, label fusion, YOLOv8 fine-tuning, video inference, domain-specific scoring, and web deployment.

## Key Features

- Hugging Face Spaces deployment with Gradio.
- Free CPU Basic-compatible runtime.
- Python 3.12 and Gradio 5.45.0 deployment configuration.
- YOLOv8-based ball detection.
- OpenCV video processing and annotation.
- User-defined target coordinate and configurable target box size.
- Calibration-aware pixel-to-centimeter distance conversion.
- Distance-based kick grading.
- Annotated MP4 output video.
- Best-frame and scoring summary display.
- Schema-safe JSON metadata rendered as text with `gr.Code`.
- Multiple scoring modes: original area-aware, final visible ball, tracked ball, and experimental perspective-calibrated fallback.
- Local model loading from `models/best.pt` or `models/last.pt`.
- Hugging Face model repository loading through `HF_MODEL_REPO_ID` and `HF_MODEL_FILENAME`.
- Fallback demo mode using `yolov8n.pt` when no fine-tuned checkpoint is available.

## Demo Workflow

1. Upload a short penalty-kick video.
2. Set the target X/Y coordinate and target box size.
3. Configure calibration, confidence, frame stride, max frames, and scoring mode.
4. Run YOLOv8 ball detection on the video.
5. Review the annotated video, best frame, distance plot, result table, and JSON metadata.

## Technical Architecture

```text
Video Upload
   ↓
OpenCV Frame Processing
   ↓
YOLOv8 Ball Detection
   ↓
Target Coordinate Distance Measurement
   ↓
Calibration-Aware Conversion
   ↓
Grade Assignment
   ↓
Annotated Video + Result Metadata
```

## Method

The dataset workflow uses FiftyOne to download and filter Open Images V7 detections for ball-related labels, including Tennis ball, Football, Volleyball (Ball), Golf ball, Rugby ball, Ball, and Cricket ball. The final dataset preparation remaps those labels into a single YOLO class: `Ball`.

The detector is fine-tuned from YOLOv8, originally using `yolov8n.pt` as the starting checkpoint. During inference, OpenCV reads the uploaded video, YOLOv8 detects the ball on processed frames, and the app measures Euclidean distance from the detected ball center to the selected target coordinate.

Calibration follows the notebook method:

```text
calibration_factor = ball_diameter_cm / ball_pixel_diameter
distance_cm = distance_px * calibration_factor
```

The deployed app uses the center of the detected bounding box as the upgraded scoring point because it better represents the ball location. The original notebook's coordinate behavior is documented in [docs/NOTEBOOK_MIGRATION.md](docs/NOTEBOOK_MIGRATION.md).

## Scoring Grades

| Grade | Distance Range        |
| ----- | --------------------- |
| SSS   | ≤ 10 cm               |
| A     | > 10 cm and ≤ 30 cm   |
| B     | > 30 cm and ≤ 50 cm   |
| C     | > 50 cm and ≤ 100 cm  |
| D     | > 100 cm and ≤ 200 cm |
| F     | > 200 cm              |

## Scoring Modes

- **Original area-aware scoring**: Preserves the notebook idea by selecting the detection whose bounding-box area is closest to the configured target area.
- **Final visible ball scoring**: Uses the last high-confidence visible ball detection.
- **Tracked ball scoring**: Applies lightweight temporal smoothing and scores the best smoothed ball-to-target distance. This is the recommended default for the demo.
- **Perspective-calibrated scoring**: Experimental placeholder. The current app explains that true perspective calibration requires field reference points and falls back safely rather than claiming full homography-based scoring.

## Model Strategy

### Local model

Place a fine-tuned YOLO checkpoint at:

```text
models/best.pt
```

The loader also checks:

```text
models/last.pt
```

### Hugging Face model repository

For a cleaner Space repository, store model weights in a separate Hugging Face model repo and configure:

```text
HF_MODEL_REPO_ID=TsukishimaAlan20/penalty-kick-distancer-yolov8-ball
HF_MODEL_FILENAME=best.pt
```

If no custom checkpoint is available, the app can load `yolov8n.pt` as fallback demo mode. This validates the app flow but is not the fine-tuned Penalty-Kick Distancer ball detector.

## Local Installation

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Optional test setup:

```bash
pip install -e ".[test]"
python -m pytest -q
```

## Hugging Face Deployment

This Space is configured for:

- Gradio SDK
- Python 3.12
- Gradio 5.45.0
- Free CPU Basic hardware
- `app.py` as the entry point

Detailed deployment notes are in [docs/DEPLOY_HUGGINGFACE.md](docs/DEPLOY_HUGGINGFACE.md).

```bash
pip install -U huggingface_hub
hf auth login
git clone https://huggingface.co/spaces/TsukishimaAlan20/penalty-kick-distancer
cd penalty-kick-distancer
git add .
git commit -m "Deploy Penalty-Kick Distancer Gradio Space"
git push
```

## Repository Structure

```text
.
├── app.py
├── README.md
├── requirements.txt
├── packages.txt
├── pyproject.toml
├── src/
│   └── penalty_kick_distancer/
├── scripts/
├── docs/
├── models/
├── examples/
└── tests/
```

## Notebooks And Migration

The original notebooks are preserved as project history. The productionized code was refactored from:

- `UsetheBall.ipynb`
- `Ball_Detection_Fiftyone.ipynb`
- `Copy_of_OpentImagesDataset_usingFiftyOne.ipynb`
- `OpentImagesDataset_usingFiftyOne_fuseONEClass.ipynb`

See [docs/NOTEBOOK_MIGRATION.md](docs/NOTEBOOK_MIGRATION.md) for the mapping from notebook workflow to reusable scripts and modules.

## Limitations

- Pixel-to-centimeter conversion is approximate unless camera geometry is controlled.
- Open Images ball labels include multiple ball-like classes, not only soccer balls.
- Detection can be affected by motion blur, lighting, occlusion, and camera angle.
- Perspective changes can distort physical distance estimates.
- Free CPU deployment is suitable for short demo clips, not heavy batch processing.
- This is a prototype, not validated professional sports analytics software.

## Future Improvements

- Homography-based field calibration.
- Stronger soccer-ball-specific dataset.
- Better temporal tracking.
- Automatic kick endpoint detection.
- More robust video examples.
- Optional ONNX inference optimization.

## Portfolio Value

This project demonstrates how to turn exploratory notebooks into a deployable computer vision demo. It covers dataset preparation, label fusion, YOLOv8 fine-tuning, frame-based video inference, applied scoring logic, Gradio UI design, Hugging Face Spaces deployment, and deployment troubleshooting.

## License

This project is released under the MIT License.

## Notes

No model accuracy claims are made in this README. A custom `models/best.pt` checkpoint is supported, but the repository should not be read as claiming that a fine-tuned checkpoint is present unless the model file is explicitly added or configured through a Hugging Face model repository.
