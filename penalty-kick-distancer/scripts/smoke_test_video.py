#!/usr/bin/env python
"""Run a local video-processing smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from penalty_kick_distancer.config import ProcessingConfig, TargetConfig
from penalty_kick_distancer.model_loader import load_yolo_model
from penalty_kick_distancer.video_processor import process_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--model", default=None, help="Optional model checkpoint path.")
    parser.add_argument("--target-x", type=int, default=960)
    parser.add_argument("--target-y", type=int, default=540)
    parser.add_argument("--target-width", type=int, default=124)
    parser.add_argument("--target-height", type=int, default=130)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_yolo_model(args.model)
    config = ProcessingConfig(
        target=TargetConfig(args.target_x, args.target_y, args.target_width, args.target_height),
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    output_video, best_frame, metrics, _records, metadata, plot_path = process_video(
        args.video,
        loaded.model,
        config,
        fallback_demo_mode=loaded.fallback_demo_mode,
    )
    print(json.dumps(metrics, indent=2))
    print(f"Output video: {output_video}")
    print(f"Best frame: {best_frame}")
    print(f"Distance plot: {plot_path}")
    print(f"Metadata keys: {list(metadata.keys())}")


if __name__ == "__main__":
    main()

