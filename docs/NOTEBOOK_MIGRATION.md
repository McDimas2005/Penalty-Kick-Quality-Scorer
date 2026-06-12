# Notebook Migration

This project keeps the original notebook method while moving it into a reproducible app structure.

## `Copy_of_OpentImagesDataset_usingFiftyOne.ipynb`

Mapped to:

```text
scripts/prepare_openimages_ball_dataset.py
```

This notebook introduced Open Images V7 download/filtering with FiftyOne and exported a YOLO-compatible dataset using multiple ball-related classes.

## `OpentImagesDataset_usingFiftyOne_fuseONEClass.ipynb`

Mapped to:

```text
scripts/prepare_openimages_ball_dataset.py
src/penalty_kick_distancer/config.py
```

This notebook contains the final one-class idea: labels such as Tennis ball, Football, Volleyball (Ball), Golf ball, Rugby ball, Ball, and Cricket ball are remapped to a single class, `Ball`.

The script preserves that by rewriting every selected detection label to `Ball` before YOLO export.

## `Ball_Detection_Fiftyone.ipynb`

Mapped to:

```text
scripts/train_yolov8_ball.py
```

The notebook fine-tuned YOLOv8 from `yolov8n.pt` using the fused one-class dataset YAML. The script exposes the same training path with CLI arguments for data path, model, epochs, image size, batch size, patience, project, and run name.

## `UsetheBall.ipynb`

Mapped to:

```text
src/penalty_kick_distancer/video_processor.py
src/penalty_kick_distancer/scoring.py
src/penalty_kick_distancer/calibration.py
src/penalty_kick_distancer/visualization.py
src/penalty_kick_distancer/tracking.py
app.py
```

The notebook's video loop is preserved as a modular pipeline:

- OpenCV reads the video.
- YOLO runs on each processed frame.
- The highest-confidence ball detection is selected by default.
- The target rectangle is drawn on the frame.
- Calibration uses `ball_diameter_cm / ball_pixel_diameter`.
- Euclidean distance is converted to centimeters.
- Grades follow the same `SSS`, `A`, `B`, `C`, `D`, `F` bands.
- Annotated output video is written to MP4.

The original notebook used top-left bounding-box coordinates for distance. The app defaults to center-point scoring because it better represents ball position, while the documentation notes the original behavior.

The notebook's area-aware idea is preserved through `Original area-aware scoring`, which selects a candidate by comparing detected ball area to target area.

