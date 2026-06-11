"""YOLO model loading with local and Hugging Face Hub strategies."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


class ModelLoadError(RuntimeError):
    """Raised when no usable YOLO model can be loaded."""


@dataclass(frozen=True)
class LoadedModel:
    model: object
    model_path: str
    source: str
    fallback_demo_mode: bool = False


def _candidate_local_paths(model_path: str | None) -> list[Path]:
    if model_path:
        return [Path(model_path)]
    return [Path("models/best.pt"), Path("models/last.pt")]


def _download_from_hub() -> Path | None:
    repo_id = os.getenv("HF_MODEL_REPO_ID")
    filename = os.getenv("HF_MODEL_FILENAME", "best.pt")
    if not repo_id:
        return None

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ModelLoadError(
            "HF_MODEL_REPO_ID is set, but huggingface_hub is not installed."
        ) from exc

    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


@lru_cache(maxsize=4)
def load_yolo_model(model_path: str | None = None) -> LoadedModel:
    """Load a YOLO model once and cache it for repeated Gradio requests.

    Resolution order:
    1. Explicit `model_path`, if provided.
    2. Local `models/best.pt` or `models/last.pt`.
    3. `HF_MODEL_REPO_ID` and `HF_MODEL_FILENAME`.
    4. `yolov8n.pt` fallback demo mode.
    """

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelLoadError("ultralytics is not installed. Run `pip install -r requirements.txt`.") from exc

    for candidate in _candidate_local_paths(model_path):
        if candidate.exists():
            return LoadedModel(model=YOLO(str(candidate)), model_path=str(candidate), source="local")

    hub_path = _download_from_hub()
    if hub_path is not None:
        return LoadedModel(model=YOLO(str(hub_path)), model_path=str(hub_path), source="huggingface_hub")

    try:
        return LoadedModel(
            model=YOLO("yolov8n.pt"),
            model_path="yolov8n.pt",
            source="ultralytics_pretrained",
            fallback_demo_mode=True,
        )
    except Exception as exc:
        raise ModelLoadError(
            "No custom model found in models/best.pt or models/last.pt, no Hugging Face "
            "model repo was configured, and the yolov8n.pt fallback could not be loaded."
        ) from exc

