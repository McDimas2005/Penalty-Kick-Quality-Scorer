---
title: Penalty-Kick Distancer
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
python_version: 3.12
app_file: app.py
pinned: false
license: mit
---

# Penalty-Kick Distancer

**Penalty-Kick Distancer** is an independent self-study computer vision prototype for sports video analysis. It fine-tunes YOLOv8 to detect a ball, then scores a penalty kick by measuring the detected ball's distance from a user-defined target coordinate.

Live demo placeholder: `https://huggingface.co/spaces/<username>/penalty-kick-distancer`

## Key Features

- YOLO ball detection for penalty-kick videos.
- Target coordinate scoring with configurable target box size.
- OpenCV video annotation with ball box, target box, distance, grade, confidence, and frame data.
- Calibration-aware distance estimate using `ball_diameter_cm / ball_pixel_diameter`.
- Gradio Blocks app ready for Hugging Face Spaces.
- Original notebook scoring preserved alongside more stable video scoring methods.

## Method Overview

The dataset workflow uses FiftyOne to download/filter Open Images V7 detections for ball-like labels such as Tennis ball, Football, Volleyball (Ball), Golf ball, Rugby ball, Ball, and Cricket ball. The final training workflow remaps all selected labels to a single YOLO class: `Ball`.

Training starts from `yolov8n.pt` and produces a one-class detector. During inference, the app processes video frames with OpenCV, runs YOLO on a configurable frame stride, and records per-frame ball distance to the target coordinate. Distances are converted from pixels to centimeters with the notebook's calibration idea.

Grade thresholds:

- `SSS`: distance <= 10 cm
- `A`: 10 < distance <= 30 cm
- `B`: 30 < distance <= 50 cm
- `C`: 50 < distance <= 100 cm
- `D`: 100 < distance <= 200 cm
- `F`: distance > 200 cm

## Screenshots And Video

Add portfolio screenshots or a short demo GIF here after deploying the Space.

## Local Run

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

## Model Weights

Strategy 1, local Space model:

```text
models/best.pt
```

Strategy 2, preferred professional approach, separate Hugging Face model repository:

```text
HF_MODEL_REPO_ID=username/penalty-kick-distancer-yolov8-ball
HF_MODEL_FILENAME=best.pt
```

If neither a local model nor a model repository is available, the app attempts to load `yolov8n.pt` as fallback demo mode. This can run the UI, but it is not your fine-tuned one-class ball detector.

## Dataset And Training

Prepare the one-class Open Images dataset:

```bash
python scripts/prepare_openimages_ball_dataset.py --max-samples 1000 --split train
```

Train YOLOv8:

```bash
python scripts/train_yolov8_ball.py --data data/openimages_ball_yolo/dataset.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --patience 5
```

Run a video smoke test:

```bash
python scripts/smoke_test_video.py --video examples/your_clip.mp4 --model models/best.pt
```

## Hugging Face Deployment

See [docs/DEPLOY_HUGGINGFACE.md](docs/DEPLOY_HUGGINGFACE.md).

Short version:

```bash
pip install -U huggingface_hub
hf auth login
git clone https://huggingface.co/spaces/<username>/penalty-kick-distancer
cd penalty-kick-distancer
git add .
git commit -m "Deploy Penalty-Kick Distancer Gradio Space"
git push
```

Upload model weights to a separate model repo:

```bash
hf upload <username>/penalty-kick-distancer-yolov8-ball models/best.pt best.pt --repo-type model
```

## Limitations

- Pixel-to-centimeter conversion is approximate unless camera geometry is controlled.
- Open Images ball classes are not only soccer balls.
- Occlusion, motion blur, lighting, and camera resolution affect detection quality.
- Camera perspective changes how pixel distance maps to physical distance.
- This is a portfolio prototype, not production sports analytics software.

## Ethical And Use Notes

Use this project as an educational demo for computer vision and sports video analysis. Do not use it to make high-stakes athlete evaluations without validation, controlled calibration, and domain review.

## Portfolio Value

This project demonstrates dataset preparation with FiftyOne, label fusion, YOLO fine-tuning, video inference with OpenCV, application-specific scoring logic, Gradio productization, and Hugging Face Spaces deployment.

