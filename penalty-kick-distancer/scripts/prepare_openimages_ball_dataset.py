#!/usr/bin/env python
"""Prepare a one-class Open Images ball dataset for YOLO training."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


BALL_CLASSES = [
    "Tennis ball",
    "Football",
    "Volleyball (Ball)",
    "Golf ball",
    "Rugby ball",
    "Ball",
    "Cricket ball",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/openimages_ball_yolo", help="YOLO export directory.")
    parser.add_argument("--yaml-path", default="data/openimages_ball_yolo/dataset.yaml", help="Dataset YAML path.")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"], help="Open Images split.")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum samples to download.")
    parser.add_argument("--dataset-name", default="open-images-v7-ball-fused", help="Local FiftyOne dataset name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
        from fiftyone import ViewField as F
    except ImportError as exc:
        raise SystemExit(
            "FiftyOne is required for dataset preparation. Install it with `pip install fiftyone`."
        ) from exc

    output_dir = Path(args.output_dir)
    yaml_path = Path(args.yaml_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading/filtering Open Images V7 detections.")
    print("Review Open Images licensing and attribution requirements before publishing derived datasets.")
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=args.split,
        label_types=["detections"],
        classes=BALL_CLASSES,
        max_samples=args.max_samples,
        dataset_name=args.dataset_name,
        overwrite=True,
    )

    view = dataset.filter_labels("ground_truth", F("label").is_in(BALL_CLASSES), only_matches=True)

    print("Remapping all selected ball-like labels to one class: Ball")
    for sample in view:
        detections = sample["ground_truth"].detections
        for detection in detections:
            detection.label = "Ball"
        sample["ground_truth"].detections = detections
        sample.save()

    print(f"Exporting YOLO dataset to {output_dir}")
    view.export(
        export_dir=str(output_dir),
        dataset_type=fo.types.YOLOv5Dataset,
        split=args.split,
        label_field="ground_truth",
        classes=["Ball"],
        include_id=True,
    )

    yaml_content = {
        "path": str(output_dir.resolve()),
        "train": f"images/{args.split}",
        "val": f"images/{args.split}",
        "nc": 1,
        "names": {0: "Ball"},
    }
    yaml_path.write_text(yaml.safe_dump(yaml_content, sort_keys=False), encoding="utf-8")
    print(f"Dataset YAML written to {yaml_path}")


if __name__ == "__main__":
    main()

