#!/usr/bin/env python
"""Export a YOLO model checkpoint."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="models/best.pt", help="Path to .pt checkpoint.")
    parser.add_argument("--format", default="onnx", choices=["onnx", "torchscript"], help="Export format.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is required. Install dependencies first.") from exc

    model = YOLO(args.weights)
    output = model.export(format=args.format, imgsz=args.imgsz, opset=args.opset)
    print(f"Export complete: {output}")
    print(".pt remains the default deployment format unless the exported model is tested in your target runtime.")


if __name__ == "__main__":
    main()

