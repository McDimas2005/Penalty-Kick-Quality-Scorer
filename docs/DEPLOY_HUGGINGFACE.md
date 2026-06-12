# Deploy To Hugging Face Spaces

This project is structured as a Gradio Space. The default deployment target is Hugging Face Spaces free/default CPU Basic hardware. GPU is optional for faster inference, but it is not required.

## Steps

1. Create a Hugging Face account.
2. Create a new Space.
3. Choose the Gradio SDK.
4. Select free/default CPU hardware.
5. Clone the Space repository.
6. Copy the project files into the Space repository.
7. Add model weights locally or upload them to a separate model repository.
8. Configure Space variables if using a separate model repository.
9. Commit and push.
10. Watch the Space build logs.
11. Test with a short video.

## Commands

```bash
pip install -U huggingface_hub
hf auth login
git clone https://huggingface.co/spaces/<username>/penalty-kick-distancer
cd penalty-kick-distancer
cp -a /path/to/this/repository/. .
git add .
git commit -m "Deploy Penalty-Kick Distancer Gradio Space"
git push
```

## Model Strategy 1: Local Space Model

Place your trained model here:

```text
models/best.pt
```

This is simple if the model file is small enough and works with your Hugging Face storage setup. The repository keeps `models/.gitkeep`, but ignores model binaries by default so large files are not accidentally committed.

## Model Strategy 2: Separate Model Repository

This is the preferred professional approach. Upload the model to a Hugging Face model repository:

```bash
hf upload <username>/penalty-kick-distancer-yolov8-ball models/best.pt best.pt --repo-type model
```

Then configure these Space variables:

```text
HF_MODEL_REPO_ID=username/penalty-kick-distancer-yolov8-ball
HF_MODEL_FILENAME=best.pt
```

The app checks `models/best.pt`, then `models/last.pt`, then the model repository variables. If no custom model is available, it attempts `yolov8n.pt` fallback demo mode.

## Build Notes

`packages.txt` includes:

```text
ffmpeg
libgl1
libglib2.0-0
```

These packages support video handling and OpenCV in the Space runtime. They are included because browser-playable MP4 output and headless OpenCV are central to the demo.

## Testing The Space

Use a short clip first, ideally under 10 seconds. If the Space is slow on CPU:

- Increase frame stride.
- Lower max processed frames.
- Raise the confidence threshold.
- Use a smaller video resolution.

## Troubleshooting

### Build cannot import OpenCV

Confirm `packages.txt` contains:

```text
ffmpeg
libgl1
libglib2.0-0
```

Also confirm `requirements.txt` uses `opencv-python-headless`, not `opencv-python`.

### Model file is missing

The app checks these sources in order:

1. `models/best.pt`
2. `models/last.pt`
3. `HF_MODEL_REPO_ID` plus `HF_MODEL_FILENAME`
4. `yolov8n.pt` fallback demo mode

Fallback mode is only for deployment verification. It is not the fine-tuned Penalty-Kick Distancer model.

### Video processing is slow

Use shorter clips, increase frame stride, lower max processed frames, or upload lower-resolution video. CPU Basic is the target and does not include GPU acceleration.

### Output video does not play

Check the build logs for FFmpeg/OpenCV errors. The app writes `.mp4` with OpenCV's `mp4v` codec and raises a clear error if `cv2.VideoWriter` cannot initialize.

### No ball detected

Lower the confidence threshold or use the fine-tuned `Ball` checkpoint instead of fallback demo mode. The app returns a no-detection summary instead of crashing.
