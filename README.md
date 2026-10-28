# Retail Video Analytics

This project implements a multi-task retail video analytics pipeline for three business questions:

1. Entrance analysis: which customers showed interest in the store, entered, or passed by?
2. Interior shelf analysis: how many interactions occurred per shelf?
3. Staff interaction analysis: how many customer sessions each staff member handled?

The solution combines object detection, multi-object tracking, and rule-based temporal logic to translate raw video into annotated output videos and summary CSVs.

## 1. Project structure

```text
retail-video-analytics/
├── configs/
│   ├── entrance_config.yaml
│   └── interior_config.yaml
├── data/
│   ├── entrance.mp4
│   └── interior.mp4
├── models/
│   ├── yolov8s.pt
│   └── yolov8n.pt
├── outputs/
│   ├── entrance_annotated.mp4
│   ├── task1_results.csv
│   ├── task3_results.csv
│   ├── interior_annotated.mp4
│   └── task2_results.csv
├── src/
│   ├── tracker.py
│   ├── entrance_task1_logic.py
│   ├── entrance_task3_logic.py
│   ├── shelf_task2_logic.py
│   ├── process_entrance.py
│   ├── process_interior.py
│   ├── setup_zones.py
│   ├── setup_exclusion_zone.py
│   ├── utils.py
│   ├── verify_config.py
│   └── __init__.py
├── Dockerfile
├── docker-entrypoint.sh
├── requirements.txt
├── .dockerignore
├── README.md
└── .gitignore
```

## 2. Methodology and task logic

### 2.1 Detection and tracking pipeline

The project uses a standard video analytics pipeline:

- Frame-by-frame person detection using YOLOv8 from Ultralytics
- Tracking with DeepSORT (`deep_sort_realtime`)
- Bounding-box filtering by confidence and minimum area
- Exclusion zones to ignore non-relevant regions (e.g. signage, architectural areas)
- Body-point approximation using the bottom-center of each bounding box for line and distance tests

This is implemented in [src/tracker.py](src/tracker.py). The key design choice is to use the bottom-center / body-ground point rather than the raw bounding-box center, because the camera is angled and the lower-body location is a more stable proxy for the person’s position when testing against a threshold line or shelf polygon.

### 2.2 Task 1: Entrance interest, entry, and pass-by logic

This logic lives in [src/entrance_task1_logic.py](src/entrance_task1_logic.py). It tracks each person through a finite-state machine:

- UNKNOWN
- INTERESTED
- ENTERED
- PASSED_BY

#### Entrance line and calibration

The entrance boundary is defined in [configs/entrance_config.yaml](configs/entrance_config.yaml):

- point_1 = (362, 166)
- point_2 = (1174, 351)

The staff seed box is also defined in the config and acts as a known point on the store side, used to orient the side-of-line test.

The side test is implemented using the signed cross product:

- side = sign((x2 - x1)(y - y1) - (y2 - y1)(x - x1))

The direction is calibrated using the known store-side point so the code knows which side of the line corresponds to the interior/store side.

#### Distance-to-line formula

The perpendicular distance from a person’s body point (implemented as the bottom-center of the bounding box) to the entrance line is computed as:

- distance = |(y2 - y1)x0 - (x2 - x1)y0 + x2y1 - y2x1| / sqrt((y2 - y1)^2 + (x2 - x1)^2)

This is the same formula used by the implementation in `_point_to_line_distance(...)` when the code evaluates the person’s position against the configured entrance line.

#### Speed calculation

Track speed is computed from the recent foot-point history:

- speed = Euclidean distance traveled / time difference
- time difference is based on the frame index and fps

#### Criteria used in current logic

The default configuration in code is:

- speed_threshold_px_per_sec = 55.0
- proximity_threshold_px = 190.0
- interest_sustain_frames = 3
- entry_confirm_frames = 4
- speed_window_frames = 8

A person is considered to have entered the interest logic when either:

- they are on the store side of the line, or
- they are within 190 px of the entrance line and moving slower than 55 px/s

To avoid single-frame noise, the system increments an interest streak and only promotes a person to INTERESTED once the streak reaches 3 consecutive frames.

After that, a person is marked ENTERED when they remain on the store side for 4 consecutive frames.

If an INTERESTED person leaves the scene without ever being confirmed as ENTERED, the final state is marked PASSED_BY.

This is intentionally conservative: a person who never shows sustained interest near the entrance is not included in the final counted metrics.

#### Why this design was chosen

The project does not use pose estimation or gaze tracking. Instead, it approximates interest by combining two observable behaviors:

- the person slows down near the entrance
- they are spatially close to the store frontage

This is a practical proxy for retail interest when only video and general-purpose detection are available.

### 2.3 Task 2: Per-shelf interest logic

This logic is implemented in [src/shelf_task2_logic.py](src/shelf_task2_logic.py).

The task is to estimate how many customer interactions occurred at each shelf zone. Each shelf is backed by a polygon defined in [configs/interior_config.yaml](configs/interior_config.yaml).

#### Shelf logic state machine

Each customer track follows a state machine:

- NOT_INTERACTING
- INTERACTING
- COOLDOWN

The logic checks the person’s body center against each shelf polygon. If a point falls inside more than one polygon, the code resolves the tie by choosing the shelf whose centroid is nearest to the point.

#### Temporal rules

The current default configuration is:

- min_dwell_seconds = 1.5
- cooldown_seconds = 1.5

The shelf count increments only after a person remains inside the same shelf polygon for the required dwell duration.

This is implemented as:

- if a person stays inside one shelf for `min_dwell_seconds * fps` frames, the state transitions to INTERACTING
- the shelf count is incremented at the moment the dwell condition is met
- when the person exits the shelf, a cooldown window is applied
- if they return to the same shelf before cooldown expires, it is treated as continuation of the same episode rather than a new interaction
- if they do not return before cooldown expires, a subsequent visit counts as a new interaction

This avoids double-counting brief pass-throughs or momentary re-entry noise.

#### Shelf area and overlay representation

The rendering in [src/process_interior.py](src/process_interior.py) draws:

- semi-transparent shelf regions
- shelf outlines in color
- per-shelf text showing the count
- a top-left summary panel titled “Shelf interactions”
- per-person association lines to the active shelf when relevant

This helps reviewers visually understand which shelf a person is currently associated with.

### 2.4 Task 3: Staff-customer interaction logic

The logic is defined in [src/entrance_task3_logic.py](src/entrance_task3_logic.py).

This task estimates staff-customer sessions by testing whether a customer remains close to a staff member for sustained frames.

#### Pairwise interaction state machine

Each pair `(staff_id, customer_track_id)` follows:

- NOT_INTERACTING
- INTERACTING

The default thresholds are:

- proximity_threshold_px = 150.0
- interaction_sustain_frames = 10
- separation_sustain_frames = 10

A session is counted when the customer remains within 150 px of the staff member for 10 consecutive frames.

The session count is incremented the moment the transition is confirmed, not when the interaction ends. This matches the project’s design: a session is a distinct event that has happened once proximity is sustained.

If the customer separates for 10 consecutive frames, the pair is re-armed so a later repeated interaction can count as a new session.

#### Staff identification

The staff member is identified using the manually seeded bounding box from [configs/entrance_config.yaml](configs/entrance_config.yaml):

- x1 = 471
- y1 = 476
- x2 = 555
- y2 = 711

The code simulates a staff ID (`staff_1`) and tracks that person as the staff instance for as long as the track remains visible. This is consistent with the project’s assumption that a staff member is treated as the same instance while they remain in view.

## 3. Technologies, models, libraries, and tools

### Core technologies

- Python 3.11
- OpenCV (`opencv-python`)
- Ultralytics YOLO
- DeepSORT via `deep_sort_realtime`
- NumPy
- PyYAML
- SciPy
- LAP
- FilterPy
- Docker

### Model usage

The tracker creates a YOLO model in [src/tracker.py](src/tracker.py):

- default model path: `models/yolov8s.pt`

The project includes multiple weights under [models](models), including `yolov8n.pt` and `yolov8s.pt`. The implementation currently defaults to `yolov8s.pt`, which is a more reliable detection model than the very small nano variant for a busy retail entrance or shelf area.

### Runtime environment

- CPU is the default runtime target in code (`device="cpu"` in the tracker)
- The code is designed to run inside Docker or directly on a Python environment with the dependencies installed

## 4. Setup and execution

### 4.1 Local environment setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# or .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 4.2 Required input files

Place input videos in the project root under [data](data):

- `data/entrance.mp4`
- `data/interior.mp4`

The project expects the tracking model at:

- `models/yolov8s.pt`

The current code does not expose a CLI model argument, so the model path is set in [src/tracker.py](src/tracker.py).

### 4.3 Run entrance pipeline

```bash
python src/process_entrance.py --video data/entrance.mp4 --skip 2
```

This generates:

- `outputs/entrance_annotated.mp4`
- `outputs/task1_results.csv`
- `outputs/task3_results.csv`

### 4.4 Run interior pipeline

```bash
python src/process_interior.py --video data/interior.mp4 --skip 2
```

This generates:

- `outputs/interior_annotated.mp4`
- `outputs/task2_results.csv`

### 4.5 Docker execution

The repository includes a production-ready Docker setup using [Dockerfile](Dockerfile) and [docker-entrypoint.sh](docker-entrypoint.sh).

Build:

```bash
docker build -t retail-video-analytics .
```

Run both tasks:

```bash
docker run --rm -v "${PWD}:/app" -w /app retail-video-analytics all
```

Run only the entrance task:

```bash
docker run --rm -v "${PWD}:/app" -w /app retail-video-analytics entrance
```

Run only the interior task:

```bash
docker run --rm -v "${PWD}:/app" -w /app retail-video-analytics interior
```

This mount (`-v "${PWD}:/app"`) ensures that data files are available inside the container and that output files remain visible on the host machine after the run.

## 5. Output files and folder structure

### Outputs produced

For the entrance workflow:

- `outputs/entrance_annotated.mp4`: annotated video with entrance line, person states, staff markers, and KPI text overlay
- `outputs/task1_results.csv`: metrics for Total Interested, Interested Entered, and Interested Passed By
- `outputs/task3_results.csv`: per-staff session counts and average sessions per staff

For the interior workflow:

- `outputs/interior_annotated.mp4`: annotated video with shelf polygons, counts, and active shelf associations
- `outputs/task2_results.csv`: shelf-wise interaction count totals

These are written by [src/process_entrance.py](src/process_entrance.py) and [src/process_interior.py](src/process_interior.py).

## 6. Design decisions and reasoning behind the approach

### Why YOLO + DeepSORT?

YOLO is used for efficient, high-quality person detection in each frame. DeepSORT is used instead of a pure motion-only tracker because it retains appearance information and can maintain identity continuity through short occlusions. This is particularly important in retail scenes where people can be temporarily hidden by fixtures, shelves, or other customers.

### Why use body-ground points for logic?

The project tests proximity and line crossing using the bottom center of each person’s bounding box. This approximates the person’s body-ground point and is more robust than using the bounding-box center in a scene with perspective distortion.

### Why use temporal streaks?

The logic deliberately uses sustained thresholds rather than single-frame triggers. This reduces false positives caused by detection jitter, temporary occlusion, or a person walking quickly past a threshold line without actually engaging with the retail area.

### Why use polygons for shelves?

Shelf interaction is inherently spatial: a customer is not simply “near” a shelf; they are physically within a defined product zone. Using polygons lets the logic capture the real shelf footprint and avoid overcounting people who are simply in the same general area.

### Why use CSV outputs in addition to video?

Video makes it easy to inspect the scene visually, while the CSVs allow objective quantification and later reporting. This is aligned with the project’s requirement to provide both an annotated visual record and machine-readable results.

## 7. Assumptions, limitations, and failure cases

### Assumptions

- The camera is static or effectively static during the clip
- People are mostly visible in the frame and roughly similar in scale
- The door entrance and shelf regions have been manually configured in the YAML files
- The staff member is identifiable by a manually seeded bounding box

### Limitations

- The logic is based on motion and proximity heuristics, not true gaze, attention, or purchase intent
- People occluded for long periods may lose identity continuity and produce undercounting or ID swaps
- If a shelf polygon is poorly configured, counts can drift or become overly sensitive
- The current entrance logic is tuned to the specific camera geometry and may need retuning for a different angle or resolution
- The default tracker uses CPU and may be slower than optimized GPU deployments

### Known failure cases

- Rapidly crossing shoppers may not meet the dwell or proximity thresholds and may remain UNKNOWN or be undercounted
- People standing in the same region but not interacting can still satisfy proximity if thresholds are too loose
- Temporary loss of the staff tracker can interrupt interaction counting for all subsequent sessions until staff is re-established
- Duplicate or overlapping detections can create confusion if the tracker mislabels a person near a shelf or entrance line

## 8. Recommended workflow for reproducibility

For consistent results:

1. Confirm the input videos exist in the project’s [data](data) folder
2. Confirm the weights exist under [models](models)
3. Verify the entry and shelf polygons in the YAML files match the actual camera view
4. Run the entrance processor and interior processor separately if required
5. Inspect the generated annotated videos for validation before using the CSV values for reporting

## 9. Summary

This project converts raw retail videos into actionable business metrics using a combination of detection, tracking, and rule-based state machines. The strongest elements of the design are the explicit temporal thresholds, domain-specific spatial definitions for the entrance and shelves, and the separation between visual annotation and tabular KPI output.

The implementation is intentionally interpretable: a reviewer can inspect the code and the YAML configuration to explain why a particular customer counted as interested, entered, passed by, or interacted with a shelf or staff member.

