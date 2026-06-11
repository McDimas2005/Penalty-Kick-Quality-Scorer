# Deploy To Hugging Face Spaces

This project is structured as a Gradio Space. It is CPU-compatible by default, with frame stride and max-frame controls to keep inference practical.

## Steps

1. Create a Hugging Face account.
2. Create a new Space.
3. Choose the Gradio SDK.
4. Clone the Space repository.
5. Copy the project files into the Space repository.
6. Add model weights locally or upload them to a separate model repository.
7. Configure Space variables if using a separate model repository.
8. Commit and push.
9. Watch the Space build logs.
10. Test with a short video.

## Commands

```bash
pip install -U huggingface_hub
hf auth login
git clone https://huggingface.co/spaces/<username>/penalty-kick-distancer
cd penalty-kick-distancer
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

