#!/usr/bin/env python
"""Train a YOLOv8 one-class ball detector."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to YOLO dataset YAML.")
    parser.add_argument("--model", default="yolov8n.pt", help="Starting YOLO model checkpoint.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="penalty_kick_ball")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is required. Install dependencies first.") from exc

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )
    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best_path = save_dir / "weights" / "best.pt"
    print(f"Training complete. best.pt: {best_path}")


if __name__ == "__main__":
    main()

