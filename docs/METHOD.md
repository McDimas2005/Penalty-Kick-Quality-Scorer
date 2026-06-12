# Method

## Problem

Penalty-Kick Distancer scores a penalty-kick video by measuring how close the detected ball is to a user-defined target coordinate. The value is not only ball detection; it is object detection combined with domain-specific scoring logic.

## Dataset

The dataset workflow comes from the original notebooks. FiftyOne is used to download/filter Open Images V7 detections for ball-related labels:

- Tennis ball
- Football
- Volleyball (Ball)
- Golf ball
- Rugby ball
- Ball
- Cricket ball

The final dataset preparation remaps all selected labels to one class: `Ball`. This makes training match the app's task, a one-class ball detector.

Open Images contains different ball-like visual sources, not only soccer balls. That diversity can help general ball detection, but it is also a limitation for soccer-specific accuracy.

## Model

The training workflow fine-tunes YOLOv8 from `yolov8n.pt`. YOLOv8 Nano is a practical default for a self-study prototype and CPU-friendly demo because it is small and fast.

The expected trained class is:

```text
0: Ball
```

## Video Inference

The app reads an uploaded video with OpenCV, extracts frame rate and dimensions, and runs YOLO inference frame-by-frame or every Nth frame. For each accepted detection, it stores:

- frame index
- timestamp
- bounding box
- confidence
- center coordinate
- distance in pixels
- distance in centimeters
- grade

The output video is annotated with the detected ball bounding box, target rectangle, distance text, grade text, coordinates, confidence, and frame index.

## Target Coordinate Scoring

The user configures:

- `x_target`
- `y_target`
- target box width
- target box height

The upgraded app uses the detected ball bounding-box center as the default scoring point. This better represents the ball location than the top-left corner. The original notebook used the top-left coordinate in its distance calculation, and that design note is preserved in the migration documentation.

## Calibration

The notebook calibration logic is preserved:

```python
calibration_factor = ball_diameter_cm / ball_pixel_diameter
distance_cm = distance_px * calibration_factor
```

The default values are:

```text
ball_diameter_cm = 22
ball_pixel_diameter = 150
```

This is an approximate single-scale conversion. It is most meaningful when camera geometry and ball depth are consistent.

## Scoring Grades

Default grade thresholds:

- `SSS`: distance <= 10 cm
- `A`: 10 < distance <= 30 cm
- `B`: 30 < distance <= 50 cm
- `C`: 50 < distance <= 100 cm
- `D`: 100 < distance <= 200 cm
- `F`: distance > 200 cm

The thresholds are configurable in code.

## Method Comparison

### Original area-aware scoring

Closest to the notebook. It selects the candidate whose detected ball area is closest to the target area and reports the corresponding distance/grade. This is useful for preserving the original prototype behavior.

### Final visible ball scoring

Uses the last high-confidence ball detection. This is simple and often reasonable for short penalty-kick clips where the final visible ball position approximates the kick result.

### Tracked ball scoring

Uses lightweight temporal smoothing over detections to reduce frame-to-frame jitter. This is the recommended default for the portfolio demo because it improves continuity without requiring a fragile tracker dependency.

### Perspective-calibrated scoring

This is included as an experimental option but does not overclaim implementation. A physically meaningful version requires field reference points and homography calibration.

## Limitations

- Pixel-to-centimeter conversion is approximate unless camera geometry is controlled.
- Open Images ball classes are not only soccer balls.
- Occlusion, motion blur, and lighting can affect detection.
- Camera perspective changes distance interpretation.
- The prototype is not full sports analytics production software.

## Future Improvements

- Add manual field reference point selection and homography.
- Train with more soccer-specific penalty-kick footage.
- Add ball tracking with explicit IDs when stable in the deployment runtime.
- Add automated target selection from goal geometry.
- Add validation metrics on a held-out soccer-ball video set.

